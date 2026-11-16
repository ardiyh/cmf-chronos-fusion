"""CPU protocol tests; the endpoint double replaces only Chronos inference."""

import contextlib
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments import run_revision as legacy


class RandomEndpoint:
    def predict(self, context, prediction_length, num_samples, **kwargs):
        shape = (1, num_samples, prediction_length)
        noise = torch.rand(shape) + float(np.random.random()) + random.random()
        return context[:, -1:].reshape(1, 1, 1) + noise


class FixedEndpoint:
    def predict(self, context, prediction_length, num_samples, **kwargs):
        assert num_samples == 3
        return torch.tensor([0.0, 2.0, 100.0]).reshape(1, 3, 1).repeat(
            1, 1, prediction_length)


class PeerReviewControlsTests(unittest.TestCase):
    def setUp(self):
        name = "experiments.run_peer_review_controls"
        self.assertIsNotNone(importlib.util.find_spec(name),
                             "peer-review control runner not implemented")
        self.controls = importlib.import_module(name)
        self.df = pd.DataFrame({
            "covariate": [5., 2., 8., 4., 99., -7.],
            "target": [10., 12., 14., 16., 13., 13.],
        })
        self.args = dict(df=self.df, origin=4, context_len=4, pred_len=2,
                         feature_cols=["covariate", "target"], target_col="target",
                         seasonality=2, device_map="cpu", pipeline=RandomEndpoint(),
                         fit_seed=42, sampler_seed=1729, num_samples=9,
                         adapter_kind="identity", mode="target_units")

    def evaluate(self, **overrides):
        return self.controls.evaluate_control_origin(**(self.args | overrides))

    def test_sampler_is_independent_of_rng_consumed_during_fitting(self):
        baseline = self.evaluate(adapter_kind="ridge")
        real_fit = legacy.cmf.fit_adapter

        def fitting_with_extra_random_work(*args, **kwargs):
            torch.rand(103)
            np.random.random(47)
            random.random()
            return real_fit(*args, **kwargs)

        with patch.object(legacy.cmf, "fit_adapter", fitting_with_extra_random_work):
            changed = self.evaluate(adapter_kind="ridge")
        np.testing.assert_array_equal(baseline["samples"], changed["samples"])

    def test_identity_replay_and_independent_sampler_seeds(self):
        first = self.evaluate()
        self.evaluate(adapter_kind="mlp", adapter_kwargs={
            "mlp_hidden": 4, "mlp_epochs": 2, "mlp_lr": 0.01})
        replay = self.evaluate(fit_seed=999)
        independent = self.evaluate(sampler_seed=1730)
        np.testing.assert_array_equal(first["samples"], replay["samples"])
        self.assertFalse(np.array_equal(first["samples"], independent["samples"]))

    def test_global_rng_states_are_restored(self):
        torch.manual_seed(701)
        np.random.seed(702)
        random.seed(703)
        expected_torch = torch.rand(3)
        expected_numpy = np.random.random(3)
        expected_python = random.random()
        torch.manual_seed(701)
        np.random.seed(702)
        random.seed(703)
        self.evaluate(adapter_kind="mlp", adapter_kwargs={
            "mlp_hidden": 4, "mlp_epochs": 2, "mlp_lr": 0.01})
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        self.assertEqual(random.random(), expected_python)

    def test_rng_states_are_restored_after_endpoint_failure(self):
        class BrokenEndpoint(RandomEndpoint):
            def predict(self, *args, **kwargs):
                super().predict(*args, **kwargs)
                raise RuntimeError("endpoint failed")

        torch.manual_seed(88)
        expected = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "endpoint failed"):
            self.evaluate(pipeline=BrokenEndpoint())
        torch.testing.assert_close(torch.random.get_rng_state(), expected)

    def test_output_inverse_transform_applies_to_every_sample(self):
        result = self.evaluate(mode="target_standardized", pipeline=FixedEndpoint(),
                               num_samples=3)
        # Past target [10,12,14,16]: mean 13, population variance 5.
        expected = np.array([13, 13 + 2 * np.sqrt(5), 13 + 100 * np.sqrt(5)])
        np.testing.assert_allclose(result["samples"], np.repeat(expected[:, None], 2, axis=1),
                                   rtol=1e-6)
        np.testing.assert_allclose(result["median_pred"], 13 + 2 * np.sqrt(5), rtol=1e-6)
        self.assertAlmostEqual(result["MASE"], np.sqrt(5) / 2, places=5)
        self.assertGreater(result["MASE_mean"], result["MASE"])
        np.testing.assert_allclose(result["q_preds"][:, 3], result["median_pred"])

    def test_fused_standardization_has_its_own_inverse(self):
        result = self.evaluate(mode="fused_standardized", pipeline=FixedEndpoint(),
                               num_samples=3)
        # This control matches the historical sample-std + 1e-5 transform.
        expected = 13 + (np.sqrt(20 / 3) + 1e-5) * np.array([0., 2., 100.])
        np.testing.assert_allclose(result["samples"][:, 0], expected, rtol=1e-6)

    def test_primary_point_metric_is_median_and_intervals_use_target_units(self):
        result = self.evaluate(pipeline=FixedEndpoint(), num_samples=3)
        self.assertEqual(result["MASE"], 11 / 4)
        self.assertEqual(result["MASE_mean"], 21 / 4)
        self.assertEqual(result["coverage_50"], 1.0)
        self.assertEqual(result["coverage_90"], 1.0)
        self.assertAlmostEqual(result["width_50"], 50.0)
        self.assertAlmostEqual(result["width_90"], 90.0)

    def test_future_values_cannot_change_samples_or_input_transform(self):
        changed = self.df.copy()
        changed.loc[4:, :] = 10000
        for mode in ["target_units", "target_standardized", "fused_standardized"]:
            with self.subTest(mode=mode):
                original = self.evaluate(adapter_kind="ridge", mode=mode)
                perturbed = self.evaluate(adapter_kind="ridge", mode=mode, df=changed)
                np.testing.assert_array_equal(original["samples"], perturbed["samples"])
                self.assertEqual(original["transform_offset"], perturbed["transform_offset"])

    def test_zero_future_is_retained_and_incomplete_context_is_rejected(self):
        zero = self.df.copy()
        zero.loc[4:, "target"] = 0
        self.assertIsNotNone(self.evaluate(df=zero))
        with self.assertRaisesRegex(ValueError, "complete context"):
            self.evaluate(origin=3)
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            self.evaluate(origin=5)

    def test_nonfinite_samples_fail_loudly(self):
        class InvalidEndpoint(FixedEndpoint):
            def predict(self, *args, **kwargs):
                return super().predict(*args, **kwargs) * float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.evaluate(pipeline=InvalidEndpoint(), num_samples=3)

    def test_seed_schedule_is_stable_across_execution_order(self):
        key = dict(dataset="toy", target="target", horizon=2, origin=4,
                   repetition=0, base_seed=1234)
        seed = self.controls.sampler_seed_for_origin(**key)
        other = self.controls.sampler_seed_for_origin(**(key | {"origin": 6}))
        self.assertEqual(seed, self.controls.sampler_seed_for_origin(**key))
        self.assertNotEqual(seed, other)
        self.assertNotEqual(seed, self.controls.sampler_seed_for_origin(
            **(key | {"repetition": 1})))

    def test_prediction_archive_roundtrips_without_pickle_and_refuses_overwrite(self):
        result = self.evaluate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.npz"
            self.controls.write_prediction(path, result)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["samples"], result["samples"])
                np.testing.assert_array_equal(saved["y_true"], result["y_true"])
                np.testing.assert_array_equal(saved["input_history"], result["input_history"])
                metadata = json.loads(str(saved["metadata_json"]))
                self.assertEqual(metadata["origin_idx"], 4)
                self.assertEqual(metadata["sampler_seed"], 1729)
                self.assertEqual(metadata["mode"], "target_units")
            with self.assertRaises(FileExistsError):
                self.controls.write_prediction(path, result)


def grid_frame():
    """60 hourly rows: split 36/48, so H=2 has six eligible test origins 48..58."""
    t = np.arange(60, dtype=float)
    target = 10 + np.sin(t / 3) + 0.1 * t
    target[52:54] = 0.0  # origin 52 has an all-zero realized horizon
    return pd.DataFrame({"covariate": np.cos(t / 5) + 0.05 * t, "target": target},
                        index=pd.date_range("2026-01-01", periods=60, freq="h"))


GRID_LABEL = "synthetic RandomEndpoint (test double)"


class PeerReviewGridTests(unittest.TestCase):
    def setUp(self):
        self.controls = importlib.import_module("experiments.run_peer_review_controls")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.out = self.tmp / "run"
        self.args = dict(df=grid_frame(), dataset="toy", feature_cols=["covariate", "target"],
                         target_col="target", seasonality=2, horizons=[2],
                         adapters=["identity", "ridge"],
                         modes=["target_units", "target_standardized"], repetitions=2,
                         base_seed=11, fit_seed=42, max_origins=3, output_dir=self.out,
                         pipeline=RandomEndpoint(), endpoint_label=GRID_LABEL,
                         device_map="cpu", context_len=8, num_samples=5)

    def run_grid(self, **overrides):
        return self.controls.run_grid(**(self.args | overrides))

    def rows(self, out=None):
        with ((out or self.out) / "losses.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_rows_share_one_sampler_seed_per_origin_and_repetition(self):
        manifest = self.run_grid()
        rows = self.rows()
        # 3 origins x 2 repetitions x (2 adapters x 2 modes + identity replay)
        self.assertEqual(len(rows), 30)
        self.assertEqual(manifest["n_records"], 30)
        for origin in (48, 52, 58):
            seeds = []
            for rep in (0, 1):
                group = [r for r in rows
                         if int(r["origin_idx"]) == origin and int(r["repetition"]) == rep]
                self.assertEqual({r["sampler_seed"] for r in group}, {str(
                    self.controls.sampler_seed_for_origin(
                        dataset="toy", target="target", horizon=2, origin=origin,
                        repetition=rep, base_seed=11))})
                seeds.append(group[0]["sampler_seed"])
            self.assertNotEqual(seeds[0], seeds[1])
        self.assertEqual({r["fit_seed"] for r in rows}, {"42"})

    def test_grid_record_equals_direct_single_origin_evaluation(self):
        self.run_grid()
        row = next(r for r in self.rows() if r["adapter"] == "ridge"
                   and r["mode"] == "target_standardized"
                   and r["origin_idx"] == "52" and r["repetition"] == "1")
        direct = self.controls.evaluate_control_origin(
            df=grid_frame(), origin=52, adapter_kind="ridge", context_len=8, pred_len=2,
            feature_cols=["covariate", "target"], target_col="target", seasonality=2,
            device_map="cpu", pipeline=RandomEndpoint(), fit_seed=42,
            sampler_seed=int(row["sampler_seed"]), mode="target_standardized",
            num_samples=5)
        with np.load(self.out / row["prediction_file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["samples"], direct["samples"])
            metadata = json.loads(str(saved["metadata_json"]))
        self.assertEqual(float(row["MASE"]), direct["MASE"])
        self.assertEqual((metadata["dataset"], metadata["repetition"], metadata["replay"]),
                         ("toy", 1, 0))

    def test_identity_replay_reuses_the_seed_and_records_its_discrepancy(self):
        self.run_grid()
        rows = self.rows()
        replays = [r for r in rows if r["replay"] == "1"]
        self.assertEqual(len(replays), 6)
        for replay in replays:
            first = next(r for r in rows if r["replay"] == "0" and r["adapter"] == "identity"
                         and r["mode"] == "target_units"
                         and r["origin_idx"] == replay["origin_idx"]
                         and r["repetition"] == replay["repetition"])
            self.assertEqual((replay["adapter"], replay["mode"]), ("identity", "target_units"))
            self.assertEqual(replay["sampler_seed"], first["sampler_seed"])
            self.assertEqual(float(replay["replay_max_abs_diff"]), 0.0)
            self.assertEqual(replay["MASE"], first["MASE"])

    def test_full_eligible_grid_differs_from_capped_pilot_and_keeps_zero_future(self):
        small = dict(adapters=["identity"], modes=["target_units"], repetitions=1)
        full = self.run_grid(max_origins=None, **small)
        capped = self.run_grid(output_dir=self.tmp / "capped", **small)
        self.assertEqual(full["grid_kind"], "full_eligible")
        self.assertEqual(full["origins"]["2"], [48, 50, 52, 54, 56, 58])
        self.assertEqual(capped["grid_kind"], "capped_pilot")
        self.assertEqual(capped["origins"]["2"], [48, 52, 58])
        self.assertEqual(capped["n_eligible"]["2"], 6)
        self.assertIn("52", {r["origin_idx"] for r in self.rows()})

    def test_existing_output_directory_is_refused_before_any_work(self):
        self.out.mkdir()
        (self.out / "keep.txt").write_text("evidence")
        with self.assertRaises(FileExistsError):
            self.run_grid()
        self.assertEqual(list(self.out.iterdir()), [self.out / "keep.txt"])

    def test_manifest_records_data_hash_provenance_and_fixed_hyperparameters(self):
        self.run_grid(repetitions=1)
        saved = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["endpoint_label"], GRID_LABEL)
        self.assertEqual(saved["sampling_args"], self.controls.SAMPLING_ARGS)
        self.assertEqual(saved["adapter_hyperparameters"]["ridge"], {"ridge_lam": 0.01})
        self.assertIn("no validation tuning", saved["hyperparameter_selection"])
        env = saved["environment"]
        self.assertEqual(env["packages"]["torch"].split("+")[0],
                         torch.__version__.split("+")[0])
        source = Path(self.controls.__file__)
        self.assertEqual(env["source_sha256"]["experiments/run_peer_review_controls.py"],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        changed = grid_frame()
        changed.iloc[40, 0] += 1.0
        other = self.run_grid(df=changed, repetitions=1, output_dir=self.tmp / "changed")
        self.assertNotEqual(saved["data_sha256"], other["data_sha256"])

    def test_noise_summary_separates_replay_from_independent_repetitions(self):
        self.run_grid(repetitions=3)
        summary = json.loads((self.out / "identity_noise_summary.json").read_text())
        self.assertEqual(summary["endpoint_label"], GRID_LABEL)
        by_h = summary["by_horizon"]["2"]
        self.assertEqual(by_h["same_seed_replay"]["n_pairs"], 9)
        self.assertEqual(by_h["same_seed_replay"]["max_abs_sample_diff"], 0.0)
        reps = by_h["independent_repetitions"]
        rows = [r for r in self.rows() if r["adapter"] == "identity"
                and r["mode"] == "target_units" and r["replay"] == "0"]
        for rep, value in enumerate(reps["case_mean_mase"]):
            self.assertAlmostEqual(value, np.mean(
                [float(r["MASE"]) for r in rows if r["repetition"] == str(rep)]))
        self.assertEqual(len(reps["case_mean_mase"]), 3)
        self.assertGreater(reps["relative_range_pct"], 0.0)

    def test_grid_requires_identity_control_and_rejects_seed_collisions(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_grid(adapters=["ridge"])
        with patch.object(self.controls, "sampler_seed_for_origin", return_value=7):
            with self.assertRaisesRegex(ValueError, "collision"):
                self.run_grid()
        self.assertFalse(self.out.exists())


class PeerReviewCliTests(unittest.TestCase):
    def setUp(self):
        self.controls = importlib.import_module("experiments.run_peer_review_controls")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "pilot"
        self.argv = ["--dataset", "toy", "--horizons", "2", "--adapters", "identity",
                     "ridge", "--modes", "target_units", "--repetitions", "1",
                     "--context-len", "8", "--num-samples", "5", "--device", "cpu",
                     "--output-dir", str(self.out)]

    def main(self, argv, **loaders):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            return self.controls.main(argv, **loaders)

    def test_help_exits_without_loading_model_or_data(self):
        stream = io.StringIO()
        with patch.object(self.controls, "_load_chronos", side_effect=AssertionError), \
                contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as exit_:
            self.controls.main(["--help"])
        self.assertEqual(exit_.exception.code, 0)
        self.assertIn("--all-eligible", stream.getvalue())
        self.assertIn("no future mask", stream.getvalue())

    def test_grid_kind_must_be_explicit(self):
        with self.assertRaises(SystemExit) as exit_:
            self.main(self.argv)
        self.assertEqual(exit_.exception.code, 2)
        with self.assertRaises(SystemExit):
            self.main(self.argv + ["--max-origins", "3", "--all-eligible"])

    def test_existing_output_is_refused_before_loading(self):
        self.out.mkdir()
        refuse = dict(load_data=lambda name: self.fail("data loaded"),
                      load_pipeline=lambda *args: self.fail("model loaded"))
        with self.assertRaises(SystemExit) as exit_:
            self.main(self.argv + ["--max-origins", "3"], **refuse)
        self.assertEqual(exit_.exception.code, 2)

    def test_injected_loaders_run_a_capped_pilot(self):
        calls = []

        def load_pipeline(model_id, revision, device):
            calls.append((model_id, revision, device))
            return RandomEndpoint(), {"model_id": model_id, "requested_revision": revision,
                                      "resolved_commit": None}

        self.main(self.argv + ["--max-origins", "3"],
                  load_data=lambda name: (grid_frame(), ["covariate", "target"], "target", 2),
                  load_pipeline=load_pipeline)
        self.assertEqual(calls, [("amazon/chronos-t5-base", None, "cpu")])
        saved = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(saved["grid_kind"], "capped_pilot")
        self.assertEqual(saved["run_info"]["checkpoint"]["model_id"], "amazon/chronos-t5-base")
        self.assertEqual(saved["n_records"], 9)  # 3 origins x (2 adapters + identity replay)

    def test_adapter_specs_are_validated_and_split_is_recorded(self):
        with self.assertRaises(SystemExit) as exit_:
            self.main(self.argv + ["--max-origins", "3", "--adapters", "identity", "krr_nope"])
        self.assertEqual(exit_.exception.code, 2)
        self.main(self.argv + ["--max-origins", "3", "--split", "val"],
                  load_data=lambda name: (grid_frame(), ["covariate", "target"], "target", 2),
                  load_pipeline=lambda *args: (RandomEndpoint(), {}))
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["split"]["name"],
                         "val")


def lag_frame():
    """target(t) = 2 * covariate(t - 1) + 1, so one past covariate lag explains it."""
    t = np.arange(80, dtype=float)
    covariate = np.sin(t / 2) + 0.3 * np.cos(t / 5)
    target = np.r_[0.0, 2 * covariate[:-1] + 1]
    return pd.DataFrame({"covariate": covariate, "target": target})


class MechanismControlTests(unittest.TestCase):
    """D3/D4 controls: covariate-only, target-only, smoothing, residual, lagged, tuning."""

    def setUp(self):
        self.controls = importlib.import_module("experiments.run_peer_review_controls")
        self.args = dict(df=grid_frame(), origin=48, context_len=8, pred_len=2,
                         feature_cols=["covariate", "target"], target_col="target",
                         seasonality=2, device_map="cpu", pipeline=RandomEndpoint(),
                         fit_seed=42, sampler_seed=1729, num_samples=5, mode="target_units")

    def history(self, adapter, **overrides):
        args = self.args | {"adapter_kind": adapter} | overrides
        return self.controls.evaluate_control_origin(**args)["input_history"]

    def test_parse_adapter_specs(self):
        parse = self.controls.parse_adapter
        self.assertEqual(parse("krr_covonly:krr_lam=0.01"), ("krr", "covonly", {"krr_lam": 0.01}))
        self.assertEqual(parse("ridge_lagged:lag_hours=1+24"),
                         ("ridge", "lagged", {"lag_hours": (1.0, 24.0)}))
        self.assertEqual(parse("ridge_lagged:lag_hours=12")[2], {"lag_hours": (12.0,)})
        self.assertEqual(parse("residual:alpha=0.5,ridge_lam=1"),
                         ("residual", "", {"alpha": 0.5, "ridge_lam": 1}))
        for bad in ("krr_nope", "identity_covonly", "ridge:krr_lam=1", "ewma",
                    "residual:alpha=0.5", "ridge_lagged", "ewma:alpha=0", "lagged",
                    "ridge:ridge_lam"):
            with self.subTest(spec=bad), self.assertRaises(ValueError):
                parse(bad)

    def test_feature_variants_route_through_the_retained_fitting_code(self):
        for spec, features in (("ridge_covonly", ["covariate"]),
                               ("krr_targetonly", ["target"])):
            with self.subTest(spec=spec):
                base = spec.split("_")[0]
                expected, _, _ = legacy._prepare_single_origin(
                    grid_frame(), 48, base, 8, 2, features, "target", "cpu", 42,
                    legacy.ADAPTER_CONFIGS[base].copy(), include_zero_future=True)
                np.testing.assert_array_equal(self.history(spec), expected[0].numpy())
        self.assertFalse(np.allclose(self.history("ridge_covonly"), self.history("ridge")))

    def test_ewma_smooths_the_target_history_only(self):
        args = dict(df=pd.DataFrame({"covariate": [5., 2., 8., 4., 99., -7.],
                                     "target": [10., 12., 14., 16., 13., 13.]}),
                    origin=4, context_len=4, seasonality=2)
        np.testing.assert_allclose(self.history("ewma:alpha=0.5", **args),
                                   [10, 11, 12.5, 14.25], rtol=1e-6)
        np.testing.assert_allclose(self.history("ewma:alpha=1", **args),
                                   self.history("identity", **args), rtol=1e-6)

    def test_residual_adapter_reduces_to_identity_without_smoothing_and_uses_covariates(self):
        np.testing.assert_allclose(self.history("residual:alpha=1,ridge_lam=1"),
                                   self.history("identity"), rtol=1e-6)
        changed = grid_frame()
        # Reverse the covariate in time; an affine change would be undone by standardization.
        changed.iloc[40:48, 0] = changed.iloc[40:48, 0].to_numpy()[::-1]
        spec = "residual:alpha=0.5,ridge_lam=0.01"
        self.assertFalse(np.allclose(self.history(spec), self.history(spec, df=changed)))

    def test_one_past_covariate_lag_reconstructs_a_lagged_target(self):
        args = dict(df=lag_frame(), origin=60, context_len=32, seasonality=24)
        insample = lag_frame()["target"].to_numpy()[28:60]
        lagged = self.history("linear_lagged:lag_hours=1", **args)
        np.testing.assert_allclose(lagged, insample, atol=1e-3)
        same_time = self.history("linear_covonly", **args)
        self.assertGreater(np.abs(same_time - insample).max(), 0.05)
        with self.assertRaisesRegex(ValueError, "before the context"):
            self.history("linear_lagged:lag_hours=1", df=lag_frame(), origin=32,
                         context_len=32, seasonality=24)
        with self.assertRaisesRegex(ValueError, "one step"):
            self.history("linear_lagged:lag_hours=0.25", **args)

    def test_new_controls_never_read_the_future(self):
        changed = grid_frame()
        changed.iloc[48:, :] = 1e4
        for spec in ("ewma:alpha=0.3", "residual:alpha=0.5,ridge_lam=0.1",
                     "ridge_lagged:lag_hours=12", "krr_covonly", "mlp_targetonly:mlp_epochs=2"):
            with self.subTest(spec=spec):
                np.testing.assert_array_equal(self.history(spec), self.history(spec, df=changed))

    def test_validation_split_and_selection(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "val"
        candidates = ["residual:alpha=1,ridge_lam=1", "residual:alpha=1,ridge_lam=2",
                      "ewma:alpha=0.5", "ewma:alpha=1"]
        manifest = self.controls.run_grid(
            df=grid_frame(), dataset="toy", feature_cols=["covariate", "target"],
            target_col="target", seasonality=2, horizons=[2],
            adapters=["identity", *candidates], modes=["target_units"], repetitions=2,
            base_seed=11, fit_seed=42, max_origins=None, output_dir=out,
            pipeline=RandomEndpoint(), endpoint_label=GRID_LABEL, context_len=8,
            num_samples=5, split="val")
        self.assertEqual(manifest["split"]["name"], "val")
        self.assertEqual(manifest["origins"]["2"], [36, 38, 40, 42, 44, 46])
        with (out / "losses.csv").open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["replay"] == "0"]
        scores = {c: np.mean([float(r["MASE"]) for r in rows if r["adapter"] == c])
                  for c in candidates}
        selected = self.controls.select_on_validation(out, candidates)["by_horizon"]["2"]
        self.assertEqual(selected["scores"], scores)
        # The two residual specs give identical forecasts, so the tie keeps list order.
        self.assertEqual(scores[candidates[0]], scores[candidates[1]])
        expected = min(candidates, key=lambda c: (scores[c], candidates.index(c)))
        self.assertEqual(selected["selected"], expected)
        self.assertNotEqual(selected["selected"], candidates[1])

    def test_selection_refuses_a_test_split_run(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "test"
        self.controls.run_grid(
            df=grid_frame(), dataset="toy", feature_cols=["covariate", "target"],
            target_col="target", seasonality=2, horizons=[2], adapters=["identity"],
            modes=["target_units"], repetitions=1, base_seed=11, fit_seed=42,
            max_origins=2, output_dir=out, pipeline=RandomEndpoint(),
            endpoint_label=GRID_LABEL, context_len=8, num_samples=5)
        with self.assertRaisesRegex(ValueError, "validation"):
            self.controls.select_on_validation(out, ["identity"])


if __name__ == "__main__":
    unittest.main()
