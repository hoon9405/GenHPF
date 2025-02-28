dest=$1
data=$2

echo "dest" $dest
echo "data" $data

python3 main.py \
--dest $dest \
--ehr mimiciii \
--first_icu \
--data $data/mimiciii/ \
--readmission --readmission --mortality --los_3day --los_7day --long_term_mortality \
--final_acuity --imminent_discharge --diagnosis --creatinine --bilirubin --platelets --wbc \
--emb_type textbase --feature "all_features" \