#!/bin/bash
#SBATCH --job-name=gams3_instruct_sft
#SBATCH --time=1-00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH=$(realpath "$0")
fi

# convert the --key=value arguments to variables
for argument in "$@"
do
  if [[ $argument == *"="* ]]; then
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare ${v,,}="${value}"
   fi
  fi
done

# time of running script
DATETIME=`date "+%Y%m%d-%H%M"`
version="${DATETIME}" # if version is not set, use DATETIME as default

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-Training/instruction_tuning

# model input, output
MODEL_NAME_OUTPUT=GaMS3-12B-SFT-Instruct

# TODO: Modify model paths if necessary
MODEL_INPUT_PATH=${WORK_DIR}/models/hf_models/GaMS3-12B
MODEL_OUTPUT_PATH=${WORK_DIR}/models/hf_models/${MODEL_NAME_OUTPUT}
mkdir -p ${MODEL_OUTPUT_PATH}

# TODO: Modify data path if necessary
DATA_DIR=$SCRIPT_DIR/data

# container
CONTAINER_DIR=$WORK_DIR/containers
NEMO_VERSION=25.09
CONTAINER_FILENAME=nemo_${NEMO_VERSION}.sqsh

# set experiment output dir
EXPERIMENTS_DIR=$SCRIPT_DIR/experiments
EXPERIMENT_NAME=gams3_sft_instruct
mkdir -p ${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${version}
WANDB_RUN_NAME=GaMS3_SFT_BS64

OUTPUT_DIR=${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${version}

# set run name
if [ "${version}" == "${DATETIME}" ]; then
  RUN_NAME=${version}
else
  RUN_NAME=${version}_R${DATETIME}
fi

# If you resume from an existing run, uncomment command below and set the path to your checkpoint
#CKPT_PATH=${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${version}/checkpoint-123

RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
  echo "Resuming from checkpoint: $CKPT_PATH"
  RESUME_ARG="--resume_from_checkpoint $CKPT_PATH"
fi

# backup this script
cp -rp ${SBATCH} ${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.sbatch

# execution script the script
SCRIPT=${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.sh
touch $SCRIPT
chmod a+x $SCRIPT

IS_DISTRIBUTED=$([ 1 -lt $SLURM_JOB_NUM_NODES ] && echo " distributed over $SLURM_JOB_NUM_NODES nodes" || echo " on 1 node")


# setup the command
echo """#!/bin/bash

pip install transformers==4.57.3
pip install deepspeed==0.18.2
pip install trl==0.25.1

# using '$(basename $SBATCH)' -> $RUN_NAME.sbatch, running $SLURM_NPROCS tasks$IS_DISTRIBUTED

# ------------------------------------------------------------
# mount points:
# ------------------------------------------------------------
# /model_input_path: '$MODEL_INPUT_PATH'
# /model_output_path: '$MODEL_OUTPUT_PATH'
# /script: '$SCRIPT_DIR'
# /experiments: '$EXPERIMENTS_DIR'
# /data: '$DATA_DIR'
# /output: '$OUTPUT_DIR'
# ------------------------------------------------------------

# print info
echo -e \"\"\"
# starting at \$(date)
# running process \$SLURM_PROCID on \$SLURMD_NODENAME
\"\"\"

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONUNBUFFERED=TRUE

# TODO: Add your WandB API key
export WANDB_API_KEY=
export WANDB_PROJECT=GaMS3-SFT-Instruction
export WANDB_DIR=/output

export TOKENIZERS_PARALLELISM=false

# train
python3 -m torch.distributed.run \\
  --nproc_per_node $SLURM_GPUS_ON_NODE \\
  --nnodes $SLURM_NNODES \\
  --node_rank \$SLURM_PROCID \\
  --master_addr $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \\
  --master_port 9901 \\
  /script/gemma_sft.py  \\
    --experiment_dir=/output \\
    --model_input_path=/model_input_path \\
    --model_output_path=/model_output_path \\
    --run_name=${WANDB_RUN_NAME} \\
    $RESUME_ARG


echo \"# completed at \$(date)\"
  """ >> $SCRIPT


# run with enroot
srun \
  --cpu-bind=verbose \
  --output="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.txt" \
  --container-mounts ${EXPERIMENTS_DIR}:/experiments,${DATA_DIR}:/data,${MODEL_INPUT_PATH}:/model_input_path,${MODEL_OUTPUT_PATH}:/model_output_path,${SCRIPT_DIR}:/script,${OUTPUT_DIR}:/output \
  --container-image ${CONTAINER_DIR}/${CONTAINER_FILENAME} \
  /experiments/${EXPERIMENT_NAME}/${RUN_NAME}.sh
