#! /usr/bin/env bash

set -e
set -u
set -f  # disable pathname expansion

# Replace the following directory with your desired working directory:
JOBS_DIR=/tmp/models
FRAMEWORK=custom

# CLI parsing
programname=${0}
flagfile=${1}
if [ ! -f "${flagfile}" ]; then
    echo "${flagfile} does not exist"
    exit
fi

shift

# Training scripts
TRAINER=`realpath ../../scripts/fit.py`
REV=`git rev-parse --short HEAD`
TIME=`date +%Y%m%dT%H%M`
RAND=`openssl rand -hex 3`
JOB_NAME=q_${REV}_${TIME}_${RAND}
JOB_DIR=${JOBS_DIR}/${JOB_NAME}
mkdir -p ${JOB_DIR}

if [ ! -d "${JOB_DIR}" ]; then
    mkdir -p ${JOB_DIR}
fi

COMMON="--framework ${FRAMEWORK}"
COMMON="${COMMON} --expt_dir ${JOB_DIR}"
COMMON="${COMMON} --num_cpu ${NUM_PROC}"
COMMON="${COMMON} --fit_verbosity 2"

# Run Training
python ${TRAINER} ${COMMON} --flagfile ${flagfile} --mode fit
python ${TRAINER} ${COMMON} --flagfile ${JOB_DIR}/flags.cfg --mode rank
