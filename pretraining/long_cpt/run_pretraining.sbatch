#!/bin/bash
#SBATCH --job-name=gams3_cpt_long
#SBATCH --time=7-00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH=$(realpath "$0")
fi

# allow passing a version, for resume training
if [ $# -gt 1 ] || [[ "$0" == *"help" ]] || [ -z "${SLURM_JOB_ID:-}" ] || [[ $# -eq 1 && "$1" != "--version="* ]]; then
  echo -e "\nUsage: sbatch ${SBATCH##*$PWD/} [--version=<version>] \n"
  exit 1
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
version=${version:-$DATETIME} # if version is not set, use DATETIME as default

# TODO: Add path to the root dir
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/GaMS3-12B-Training/long_cpt

# TODO: Add path to your container dir
CONTAINER_DIR=
NEMO_VERSION=25.09

# TODO: Add path to the data dir
DATA_DIR=

# experiment name
EXPERIMENT_NAME=gams3_12b_cpt_long

# set dir
EXPERIMENT_DIR=${WORK_DIR}/nemo_experiments
mkdir -p ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${version}

# set run name
if [ "${version}" == "${DATETIME}" ]; then
  RUN_NAME=${version}
else
  RUN_NAME=${version}_R${DATETIME}
fi

# backup this script
cp -rp ${SBATCH} ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.sbatch

# execution script the script
SCRIPT=${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.sh
touch $SCRIPT
chmod a+x $SCRIPT

IS_DISTRIBUTED=$([ 1 -lt $SLURM_JOB_NUM_NODES ] && echo " distributed over $SLURM_JOB_NUM_NODES nodes" || echo " on 1 node")
FIX=""
# fixing $FIX

# setup the command
echo """#!/bin/bash

# using '$(basename $SBATCH)' -> $RUN_NAME.sbatch, running $SLURM_NPROCS tasks$IS_DISTRIBUTED
if [ \"\$SLURM_LOCALID\" == \"0\" ]; then
  sed -i 's/\"torch\"/\"nemo\",\"torch\"/g' \$(/usr/bin/python -c 'import torch.utils.collect_env; print(torch.utils.collect_env.__file__)')
else
  sleep 5
fi

# print info
echo -e \"\"\"
# starting at \$(date)
# running process \$SLURM_PROCID on \$SLURMD_NODENAME
\$(nvidia-smi | grep Version | sed -e 's/ *| *//g' -e \"s/   */\n# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\" -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(nvidia-smi -L | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(nproc | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   vCPU: /g\")
\$(python -c 'import multiprocessing; print(multiprocessing.cpu_count())' | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   mp.CPU: /g\")
\$(python -c 'import os; print(os.cpu_count())' | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   os.CPU: /g\")
\$(echo \"---\" | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import nemo; print(f\"nemo: {nemo.__version__}\")' | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import torch; print(f\"torch: {torch.__version__}\")' | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(echo \"---\" | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import torch; print(torch.__config__.parallel_info())' | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(echo \"---\" | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import torch, torch.utils.collect_env; torch.utils.collect_env.main()' | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(echo \"---\" | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(env | grep SLURM | sort | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(echo \"---\" | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(env | grep -v SLURM | sort | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(cat /etc/nccl.conf | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\"\"\"

# set unbuffered python for realtime container logging
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# export errors
export HYDRA_FULL_ERROR=1

# debug distribured
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export CUDA_DEVICE_MAX_CONNECTIONS=1

# TODO: add your WandB API key
export WANDB_API_KEY=

export TOKENIZERS_PARALLELISM=false

# train
python /script/gemma3_pretraining.py \\
  --model_path=/models/gams3_12b_cpt_base \\
  --enable_activation_checkpointing \\
  --wandb_name=gams3_12b_TP8_B200 \\
  --wandb_project="GaMS3_CPT-long" \\
  --version=$version \\
  --experiment_dir=/experiments/$EXPERIMENT_NAME \\
  --use_sequence_parallelism \\
  --num_nodes=$SLURM_JOB_NUM_NODES \\
  --num_gpus_per_node=${SLURM_GPUS_ON_NODE} \\
  --seq_length=131072 \\
  --log_every_n_steps=5 \\
  --max_steps=2374 \\
  --val_check_interval=60 \\
  --global_batch_size=64 \\
  --limit_val_batches=23 \\
  --warmup_steps=500 \\
  --hold_steps=1000 \\
  --min_lr=5e-6 \\
  --max_lr=1e-6 \\
  --tp_size=8

echo \"# completed at \$(date)\"
  """ >> $SCRIPT


# run with enroot
srun \
  --cpu-bind=verbose \
  --output="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}.txt" \
  --container-mounts ${EXPERIMENT_DIR}:/experiments,${DATA_DIR}:/data,${WORK_DIR}/models/nemo_models:/models,$SCRIPT_DIR:/script,$SCRIPT_DIR/scripts_to_bind/gemma3.py:/opt/NeMo/nemo/collections/llm/gpt/model/gemma3.py \
  --container-image ${CONTAINER_DIR}/nemo_${NEMO_VERSION}.sqsh \
    /experiments/${EXPERIMENT_NAME}/${RUN_NAME}.sh
