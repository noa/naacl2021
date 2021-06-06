#! /usr/bin/env bash

set -f
set -u
set -e

SCRIPT_DIR=../../scripts
TEST_DIR=/REPLACE/WITH/PATH/TO/RECORDS
EXPT_DIR=/REPLACE/WITH/PATH/TO/MODEL/CHECKPOINT

python ${SCRIPT_DIR}/fit.py \
       --flagfile ${EXPT_DIR}/flags.cfg \
       --mode rank \
       --num_cpu 12 \
       --num_queries -1 \
       --expt_dir ${EXPT_DIR} \
       --results_filename test_results.txt \
       --train_tfrecord_path ${TEST_DIR}/queries*.tf \
       --valid_tfrecord_path ${TEST_DIR}/targets*.tf \
       --expt_config_path reddit.json

# eof
