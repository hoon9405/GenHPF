#!/usr/bin/env bash
# shellcheck disable=SC2002,SC2086,SC2046
export $(cat .env | xargs)

CONFIG_NAME=$1
shift
COHORT_NAME=$1
shift

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python \
  $EVENT_STREAM_PATH/scripts/build_dataset.py \
  --config-path=$(pwd)/configs \
  --config-name=$CONFIG_NAME \
  "hydra.searchpath=[$EVENT_STREAM_PATH/configs]" cohort_name=$COHORT_NAME "$@"

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python \
  $EVENT_STREAM_PATH/scripts/convert_to_ESDS.py \
  "dataset_dir=$PROJECT_DIR/$COHORT_NAME" \
  "ESDS_save_dir=$PROJECT_DIR/$COHORT_NAME/ESDS" \
  "do_overwrite=True"