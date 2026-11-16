#!/bin/bash
# Rebuild the CMF h5 datasets from public raw files with the unchanged paper-2
# preprocessing scripts, then compare each frame with the experiment frames.
#   reconstruct.sh WORK_DIR STC_CODE_DIR CMF_CODE_DIR CMF_DATA_DIR PREFLIGHT_JSON
# Raw downloads land in WORK_DIR/raw/<dataset>/; nothing outside WORK_DIR is written.
set -euo pipefail
WORK=$1 STC=$2 CMF=$3 CMF_DATA=$4 PREFLIGHT=$5
PY="${PYTHON:-python3}"
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
# must be placed in $RAW/sdwpf/ by hand (registration-gated download).
(cd $RAW && shasum -a 256 ett/* solar/* ecl/* opsd/* sdwpf/* > RAW_SHA256SUMS)

cp $STC/scripts/preprocess_{solar,ecl,opsd,sdwpf}.py $OUT/scripts/
# preprocess_solar.py reads its .gz-named input as plain text, so decompress first.
gunzip -c $RAW/solar/solar_AL.txt.gz > $OUT/data/SolarAl/solar_AL.txt.gz
(cd $RAW/ecl && unzip -oq LD2011_2014.txt.zip)
ln -sf $RAW/ecl/LD2011_2014.txt $OUT/data/ECL/
ln -sf $RAW/opsd/time_series_60min_singleindex.csv $OUT/data/OPSD/opsd_raw.csv
ln -sf $RAW/sdwpf/sdwpf_245days_v1.csv $RAW/sdwpf/sdwpf_baidukddcup2022_turb_location.csv $OUT/data/SDWPF/
cd $WORK
$PY -I $OUT/scripts/preprocess_solar.py > $OUT/solar.log 2>&1
$PY -I $OUT/scripts/preprocess_ecl.py > $OUT/ecl.log 2>&1
$PY -I $OUT/scripts/preprocess_opsd.py --variable wind > $OUT/opsd.log 2>&1
$PY -I $OUT/scripts/preprocess_sdwpf.py > $OUT/sdwpf.log 2>&1
mkdir -p $OUT/data/ETT
cp $RAW/ett/ETTh1.csv $OUT/data/ETT/etth1.csv; cp $RAW/ett/ETTh2.csv $OUT/data/ETT/etth2.csv
cp $RAW/ett/ETTm1.csv $OUT/data/ETT/ettm1.csv; cp $RAW/ett/ETTm2.csv $OUT/data/ETT/ettm2.csv
$PY "$(dirname "$0")/compare_frames.py" --cmf-code $CMF --rebuilt $OUT/data --experiment $CMF_DATA \
  --preflight $PREFLIGHT --output $WORK/comparison.json
