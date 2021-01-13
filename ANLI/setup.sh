#!/usr/bin/env bash

function runexp {

export GLUE_DIR=anli_data
export TASK_NAME=${1}

custom=${2}   # Custom name
mname=${3}    # Model name
lr=${4}       # Learning rate for model parameters
bsize=${5}    # Batch size
seqlen=${6}    # Maximum sequence length
ts=${7}      # Number of training steps (counted as parameter updates)
ws=${8}      # Learning rate warm-up steps
seed=${9}    # Seed for randomness
wd=${10}      # Weight decay
beta=${11}    # regularizer coefficient
version=${12} # mi estimator version
hdp=${13}     # Hidden layer dropouts for ALBERT
adp=${14}     # Attention dropouts for ALBERT
alr=${15}      # Step size of gradient ascent
amag=${16}     # Magnitude of initial (adversarial?) perturbation
anorm=${17}    # Maximum norm of adversarial perturbation
asteps=${18}   # Number of gradient ascent steps for the adversary
alpha=${19}   # alpha for controlling local robust regularizer
cl=${20}   # lower threshold
ch=${21}   # higher threshold

if [ -z "${20}" ] ;then
    cl=0.5
fi

if [ -z "${21}" ] ;then
    ch=0.9
fi

expname=${custom}-${mname}-${TASK_NAME}-sl${seqlen}-lr${lr}-bs${bsize}-ts${ts}-ws${ws}-wd${wd}-seed${seed}-beta${beta}-alpha${alpha}--cl${cl}-ch${ch}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-version${version}

max=-1
for file in ${expname}/checkpoint-*
do
  fname=$(basename ${file})
  num=${fname:11}
  [[ $num -gt $max ]] && max=$num
done

if [ $max -eq -1 ] ; then
 echo "Train from stratch"
else
  FILE="${expname}/checkpoint-$max/eval_hist.bin"
  if [[ -f "$FILE" ]]; then
      echo "$FILE exists."
      mname="${expname}/checkpoint-$max" && echo "Resume Training from checkpoint $mname"
  else
      echo "$FILE does not exists."
      if [ $max -eq 100 ]; then
        max=-1
        echo "Train from stratch"
      else
        max=$(($max-100))
        FILE="${expname}/checkpoint-$max/eval_hist.bin"
        echo "use $FILE instead."
        mname="${expname}/checkpoint-$max" && echo "Resume Training from checkpoint $mname"
      fi
  fi
fi


port=$(($RANDOM + 1024))
echo "Master port: ${port}"
python -m torch.distributed.launch --nproc_per_node=1 --master_port ${port} ./run_anli.py \
  --model_name_or_path ${mname} \
  --task_name $TASK_NAME \
  --do_train  \
  --do_eval \
  --data_dir $GLUE_DIR \
  --max_seq_length ${seqlen} \
  --per_device_train_batch_size ${bsize} \
  --learning_rate ${lr} \
  --max_steps ${ts} \
  --warmup_steps ${ws} \
  --weight_decay ${wd} \
  --seed ${seed} \
  --beta ${beta} \
  --logging_dir ${expname} \
  --output_dir ${expname} \
  --version ${version}  --evaluate_during_training\
  --logging_steps 500 --save_steps 500 \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp}  --overwrite_output_dir  \
  --adv_lr ${alr} --adv_init_mag ${amag} --adv_max_norm ${anorm} --adv_steps ${asteps} --alpha ${alpha} \
   --cl ${cl} --ch ${ch}
}




function evalexp {
#export NCCL_DEBUG=INFO
#export NCCL_IB_CUDA_SUPPORT=0
#export NCCL_P2P_DISABLE=0
#export NCCL_IB_DISABLE=1
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_NET_GDR_READ=0
#export NCCL_SHM_DISABLE=0

export GLUE_DIR=anli_data
export TASK_NAME="anli-full"

mname=${2}    # Model name
custom=${1}   # Custom name
lr=5e-3       # Learning rate for model parameters
bsize=32    # Batch size
seqlen=128    # Maximum sequence length
ts=0      # Number of training steps (counted as parameter updates)
ws=0      # Learning rate warm-up steps
seed=42    # Seed for randomness
wd=1e-5      # Weight decay
beta=0    # regularizer coefficient
version=3 # mi estimator version
hdp=0     # Hidden layer dropouts for ALBERT
adp=0     # Attention dropouts for ALBERT
alr=0      # Step size of gradient ascent
amag=0     # Magnitude of initial (adversarial?) perturbation
anorm=0    # Maximum norm of adversarial perturbation
asteps=1   # Number of gradient ascent steps for the adversary
alpha=0   # alpha for controlling local robust regularizer
cl=0   # lower threshold
ch=0   # higher threshold

expname=${custom}-load
port=$(($RANDOM + 1024))
echo "Master port: ${port}"

python -m torch.distributed.launch --nproc_per_node=1 --master_port ${port}  ./run_anli.py \
  --model_name_or_path ${mname} \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR \
  --max_seq_length ${seqlen} \
  --per_device_train_batch_size ${bsize} \
  --learning_rate 3e-5 \
  --max_steps ${ts} \
  --warmup_steps ${ws} \
  --weight_decay ${wd} \
  --seed ${seed} \
  --beta ${beta} \
  --logging_dir ${expname} \
  --output_dir ${expname} \
  --version ${version} \
  --logging_steps 100 --save_steps 100 \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp}  --evaluate_during_training \
  --adv_lr ${alr} --adv_init_mag ${amag} --adv_max_norm ${anorm} --adv_steps ${asteps} --overwrite_output_dir \
  --alpha ${alpha} --cl ${cl} --ch ${ch}

tail -n +1 "${expname}"/eval_results*.txt > "${expname}"/eval_results.txt
cat "${expname}"/eval_results.txt
}

