#! /usr/bin/env bash

set -f
set -u
set -e

SCRIPT_DIR=../../scripts
TEST_DIR=/expscratch/nandrews/aleem/reddit/train
EXPT_DIR=${1}
VAL_LEN=${2}

python ${SCRIPT_DIR}/fit.py \
       --flagfile ${EXPT_DIR}/flags.cfg \
       --mode link \
       --num_cpu 12 \
       --num_queries -1 \
       --expt_dir ${EXPT_DIR} \
       --results_filename test_results.txt \
       --train_tfrecord_path ${TEST_DIR}/queries*.tf \
       --valid_tfrecord_path ${TEST_DIR}/targets*.tf \
       --linking_queries /expscratch/nandrews/aleem/reddit_1mil_all_authors/linking/novel/queries*.tf \
       --linking_targets /expscratch/nandrews/aleem/reddit_1mil_all_authors/linking/novel/targets*.tf \
       --linking_test_targets /expscratch/nandrews/aleem/reddit_1mil_all_authors/linking/1k_50k_test/test_targets*.tf \
       --expt_config_path reddit.json \
       --episode_len 100 \
       --min_episode_len 100 \
       --min_val_len ${VAL_LEN} \
       --max_val_len ${VAL_LEN} 

# eof
