# Dataset preparation

`retained-inputs.zip` contains the six inputs used in the reported experiments.
Extract it into a directory and verify the files with `shasum -a 256 -c
SHA256SUMS` from that directory. Set `CMF_DATA_ROOT` to the directory after
supplying the OPSD and Solar-AL files using the reconstruction instructions in
the repository README. ECL and SDWPF retain the experiment timestamp encoding
in this archive. The reconstruction scripts produce files with nanosecond
timestamps and the same value bytes on x86-64.

The [raw-file checksums](RAW_SHA256SUMS) record the nine inputs used in the
x86-64 reconstruction. Run `shasum -a 256 -c ../../data_prep/RAW_SHA256SUMS`
from `WORK/raw` to verify the downloads made with the README example.

| Dataset | Source and preparation | Files in the archive |
|---|---|---|
| ETTh1, ETTh2, ETTm1 and ETTm2 | [ETDataset at commit 1d16c8f4](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small), by Haoyi Zhou and collaborators. The benchmark CSVs are read directly. Their names are lowercased to match the loader. | `ETT/*.csv`, CC BY-ND 4.0. The upstream license text is included. |
| ECL | [ElectricityLoadDiagrams20112014](https://doi.org/10.24432/C58C86), Artur Trindade, UCI, 2015. `preprocess_ecl.py` selects 2012–2014, sums the quarter-hour readings into hourly observations, applies its missing-value filter, and writes the processed client series to HDF5. | `ECL/ecl.h5`, derived from the CC BY 4.0 source. |
| SDWPF | [SDWPF dataset](https://doi.org/10.6084/m9.figshare.24798654), Jingbo Zhou and collaborators, Figshare. From the KDD Cup subset, download [`sdwpf_245days_v1.csv`](https://figshare.com/ndownloader/files/46005813) and [`sdwpf_baidukddcup2022_turb_location.csv`](https://figshare.com/ndownloader/files/46005807). `preprocess_sdwpf.py` pivots active power (`Patv`) by timestamp and turbine, averages duplicate entries, and sets missing and negative readings to zero. | `SDWPF/sdwpf.h5`, derived from the CC BY 4.0 source. |
| OPSD-wind | [Open Power System Data, Time Series, 2020-10-06](https://doi.org/10.25832/time_series/2020-10-06), based on ENTSO-E Transparency. `preprocess_opsd.py --variable wind` selects the 2015–2019 hourly wind series, linearly interpolates up to six consecutive missing readings, then sets remaining missing and negative values to zero. | Rebuild from the source. The source package does not specify a redistribution license for these series. |
| Solar-AL | [LSTNet data at commit 7f402f18](https://github.com/laiguokun/multivariate-time-series-data/tree/7f402f185cc2435b5e66aed13a3b560ed142e023/solar-energy), from NREL's Alabama solar data. `preprocess_solar.py` reads the 137 source columns and assigns a ten-minute index from 2006-01-01. | Rebuild from the source. The LSTNet dataset repository does not specify a redistribution license for these files. |

The license for each included dataset follows its source terms ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) or [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)). The repository's MIT license covers the software.
