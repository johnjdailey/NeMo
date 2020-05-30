#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

if [ ! -f setup.py ]; then
  echo "Script should be runned from project root."
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=32    # Default is 32.
NUM_GPU=8     # Default is 8.
WS=stan       # Workspace name.
WORKSPACE=/ws # Workspace mount point.
RESULT=/result

# ---------------------------------------------------- SAVE STATE ----------------------------------------------------
echo "Updating source code..."
# Choose run id.
name_id=$1
num_id=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13)
id="${name_id}"_"${num_id}"
# Choose tmp dir to save current state of the project.
tmp_dir=/tmp/nemos/"${id}"
echo "Tmp dir: ${tmp_dir}"
# Save current state.
mkdir -p "${tmp_dir}"
rsync -r . "${tmp_dir}" --exclude .git --filter=":- .gitignore"
ngc workspace upload "${WS}" --source "${tmp_dir}" --destination nemos/"${id}"

# -------------------------------------------------- CHOOSE COMMAND --------------------------------------------------
script=examples/asr/quartznet.py
config=examples/asr/configs/quartznet15x5-libritts-durs.yaml
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${script} \
--num_epochs=150 \
--eval_batch_size=32 \
--model_config=${config} \
--checkpoint_dir=${RESULT} \
--checkpoint_save_freq=10000 \
--train_dataset=/manifests/libritts/train-all.json \
--eval_datasets \
/manifests/libritts/dev-clean.json \
/manifests/libritts/dev-other.json \
/manifests/libritts/test-clean.json \
/manifests/libritts/test-other.json
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 58106:/data/libritts \
  --datasetid 58404:/manifests/libritts \
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
