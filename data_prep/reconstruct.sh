#!/bin/bash
# Rebuild the CMF h5 datasets from public raw files with the unchanged paper-2
# preprocessing scripts, then check the rebuilt frame hashes against preflight.
#   reconstruct.sh WORK_DIR STC_CODE_DIR CMF_CODE_DIR EXPERIMENT_DATA_DIR PREFLIGHT_JSON
# Pass - for EXPERIMENT_DATA_DIR when the original experiment files are unavailable.
# PREP_PYTHON runs the preprocessing scripts (pandas 2.2, requirements/data_prep.txt);
# PYTHON runs the comparison with the CMF loader (requirements/lgbm.txt).
# Raw downloads land in WORK_DIR/raw/<dataset>/; nothing outside WORK_DIR is written.
set -euo pipefail
# Resolve every argument to an absolute path first: the script changes directory
# before running the preprocessing and comparison steps.
WORK=$(mkdir -p "$1" && cd "$1" && pwd) STC=$(cd "$2" && pwd) CMF=$(cd "$3" && pwd)
PREFLIGHT=$(cd "$(dirname "$5")" && pwd)/$(basename "$5")
COMPARE_ARGS=(--cmf-code "$CMF" --rebuilt "$WORK/rebuilt/data" --preflight "$PREFLIGHT"
              --output "$WORK/comparison.json")
if [[ "$4" != - ]]; then
  CMF_DATA=$(cd "$4" && pwd)
  COMPARE_ARGS+=(--experiment "$CMF_DATA")
fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY="${PYTHON:-python3}"
PREP="${PREP_PYTHON:-$PY}"
RAW=$WORK/raw OUT=$WORK/rebuilt
mkdir -p $RAW/{ett,solar,ecl,opsd} $OUT/scripts $OUT/data/{SolarAl,ECL,OPSD,SDWPF}

for f in ETTh1 ETTh2 ETTm1 ETTm2; do   # ETDataset commit 1d16c8f4f943005d613b5bc962e9eeb06058cf07
  curl -sSfL -o $RAW/ett/$f.csv \
    https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/$f.csv
done
curl -sSfL -o $RAW/solar/solar_AL.txt.gz \
  https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/7f402f185cc2435b5e66aed13a3b560ed142e023/solar-energy/solar_AL.txt.gz
curl -sSfL -o $RAW/ecl/LD2011_2014.txt.zip \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
curl -sSfL -o $RAW/opsd/time_series_60min_singleindex.csv \
  https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
# SDWPF (Baidu KDD Cup 2022): sdwpf_245days_v1.csv and the turbine location file
# must be placed in $RAW/sdwpf/ by hand (see data_prep/DATASETS.md).
(cd "$RAW" && shasum -a 256 ett/{ETTh1,ETTh2,ETTm1,ETTm2}.csv \
  solar/solar_AL.txt.gz ecl/LD2011_2014.txt.zip opsd/time_series_60min_singleindex.csv \
  sdwpf/{sdwpf_245days_v1,sdwpf_baidukddcup2022_turb_location}.csv > RAW_SHA256SUMS)

cp $STC/scripts/preprocess_{solar,ecl,opsd,sdwpf}.py $OUT/scripts/
# preprocess_solar.py reads its .gz-named input as plain text, so decompress first.
gunzip -c $RAW/solar/solar_AL.txt.gz > $OUT/data/SolarAl/solar_AL.txt.gz
(cd $RAW/ecl && unzip -oq LD2011_2014.txt.zip)
ln -sf $RAW/ecl/LD2011_2014.txt $OUT/data/ECL/
ln -sf $RAW/opsd/time_series_60min_singleindex.csv $OUT/data/OPSD/opsd_raw.csv
ln -sf $RAW/sdwpf/sdwpf_245days_v1.csv $RAW/sdwpf/sdwpf_baidukddcup2022_turb_location.csv $OUT/data/SDWPF/
cd $WORK
$PREP -I $OUT/scripts/preprocess_solar.py > $OUT/solar.log 2>&1
$PREP -I $OUT/scripts/preprocess_ecl.py > $OUT/ecl.log 2>&1
$PREP -I $OUT/scripts/preprocess_opsd.py --variable wind > $OUT/opsd.log 2>&1
$PREP -I $OUT/scripts/preprocess_sdwpf.py > $OUT/sdwpf.log 2>&1
mkdir -p $OUT/data/ETT
cp $RAW/ett/ETTh1.csv $OUT/data/ETT/etth1.csv; cp $RAW/ett/ETTh2.csv $OUT/data/ETT/etth2.csv
cp $RAW/ett/ETTm1.csv $OUT/data/ETT/ettm1.csv; cp $RAW/ett/ETTm2.csv $OUT/data/ETT/ettm2.csv
"$PY" "$HERE/compare_frames.py" "${COMPARE_ARGS[@]}"
