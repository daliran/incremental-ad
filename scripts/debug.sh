#!/bin/bash

PARTITION="all_serial"
ACCOUNT="tesi_ddellacasaventurelli01"
TIME="01:00:00"
GRES="gpu:1"
NODE="ailb-login-02"

USE_IMMEDIATE=1
IMMEDIATE_TIMEOUT=10
PIN_NODE=1

SRUN_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --time=${TIME} --gres=${GRES}"

if [ "${USE_IMMEDIATE}" -eq 1 ]; then
    SRUN_ARGS="${SRUN_ARGS} -Q --immediate=${IMMEDIATE_TIMEOUT}"
fi

if [ "${PIN_NODE}" -eq 1 ]; then
    SRUN_ARGS="${SRUN_ARGS} -w ${NODE}"
fi

srun ${SRUN_ARGS} --pty /bin/bash
