#!/usr/bin/env bash
function runstandardsquad {

export SQUAD_DIR=squad_data

custom=${1}   # Custom name
gpu=${2}      # number of GPU
bsize=${3}    # Batch size
beta=${4}    # regularizer coefficient
version=${5} # mi estimator version
mname=${6}
#beta=${11}    # regularizer coefficient
#version=${12} # mi estimator version
#hdp=${13}     # Hidden layer dropouts for ALBERT
#adp=${14}     # Attention dropouts for ALBERT
#alr=${15}      # Step size of gradient ascent
#amag=${16}     # Magnitude of initial (adversarial?) perturbation
#anorm=${17}    # Maximum norm of adversarial perturbation
#asteps=${18}   # Number of gradient ascent steps for the adversary
export seqlen=384
export lr=3e-5

if [[ ${mname} == *"roberta"* ]]; then
  model_type=roberta
else
  model_type=bert
fi

expname=${custom}-${mname}-sl${seqlen}-lr${lr}-bs${bsize}-beta${beta}-version${version}

max=-1
for file in ${expname}/checkpoint-*
do
  fname=$(basename ${file})
  num=${fname:11}
  [[ $num -gt $max ]] && max=$num
done

if [ $max -eq -1 ]
then echo "Train from stratch"
else mname="${expname}/checkpoint-$max" && echo "Resume Training from checkpoint $mname"
fi

python -m torch.distributed.launch --nproc_per_node=${gpu} ./run_squad_standard.py \
    --model_type ${model_type} --evaluate_during_training --overwrite_output_dir  \
    --model_name_or_path ${mname} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 --save_steps 5000 \
    --output_dir ${expname} --data_dir ${SQUAD_DIR}\
    --per_gpu_eval_batch_size=${bsize}   \
    --per_gpu_train_batch_size=${bsize}  \
    --beta ${beta} \
    --version ${version} \
&& echo "add sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifyAll.json \
${expname}/predictions_add_sent.json \
&& echo "add one sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifySample.json \
${expname}/predictions_add_one_sent.json
}




function runsquad {

export SQUAD_DIR=squad_data

custom=${1}   # Custom name
gpu=${2}      # number of GPU
bsize=${3}    # Batch size
beta=${4}    # regularizer coefficient
version=${5} # mi estimator version
hdp=${6}     # Hidden layer dropouts for ALBERT
adp=${7}     # Attention dropouts for ALBERT
alr=${8}      # Step size of gradient ascent
amag=${9}     # Magnitude of initial (adversarial?) perturbation
anorm=${10}    # Maximum norm of adversarial perturbation
asteps=${11}   # Number of gradient ascent steps for the adversary
mname=${12}
alpha=${13}
cl=${14}
ch=${15}
if [ -z "${14}" ] ;then
    cl=0.5
fi

if [ -z "${15}" ] ;then
    ch=0.9
fi
#beta=${11}    # regularizer coefficient
#version=${12} # mi estimator version
#hdp=${13}     # Hidden layer dropouts for ALBERT
#adp=${14}     # Attention dropouts for ALBERT
#alr=${15}      # Step size of gradient ascent
#amag=${16}     # Magnitude of initial (adversarial?) perturbation
#anorm=${17}    # Maximum norm of adversarial perturbation
#asteps=${18}   # Number of gradient ascent steps for the adversary
export seqlen=384
export lr=3e-5
#export mname=bert-large-uncased-whole-word-masking

#mname=${3}    # Model name
#lr=${4}       # Learning rate for model parameters
#seqlen=${6}    # Maximum sequence length
#ts=${7}      # Number of training steps (counted as parameter updates)
#ws=${8}      # Learning rate warm-up steps
#seed=${9}    # Seed for randomness
#wd=${10}      # Weight decay
if [[ ${mname} == *"roberta"* ]]; then
  model_type=roberta
else
  model_type=bert
fi

#expname=${custom}-${mname}-${TASK_NAME}-sl${seqlen}-lr${lr}-bs${bsize}-ts${ts}-ws${ws}-wd${wd}-seed${seed}-beta${beta}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-version${version}
expname=${custom}-${mname}-sl${seqlen}-lr${lr}-bs${bsize}-beta${beta}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-alpha${alpha}-cl${cl}-ch${ch}-version${version}

max=-1
for file in ${expname}/checkpoint-*
do
  fname=$(basename ${file})
  num=${fname:11}
  [[ $num -gt $max ]] && max=$num
done

if [ $max -eq -1 ]
then echo "Train from stratch"
else mname="${expname}/checkpoint-$max" && echo "Resume Training from checkpoint $mname"
fi

python -m torch.distributed.launch --nproc_per_node=${gpu} ./run_squad.py \
    --model_type ${model_type} --evaluate_during_training --overwrite_output_dir  \
    --model_name_or_path ${mname} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 --save_steps 5000 \
    --output_dir ${expname} --data_dir ${SQUAD_DIR}\
    --per_gpu_eval_batch_size=${bsize}   \
    --per_gpu_train_batch_size=${bsize}  \
    --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
    --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
    --beta ${beta}  --alpha ${alpha}  --cl ${cl} --ch ${ch} \
    --version ${version} \
&& echo "add sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifyAll.json \
${expname}/predictions_add_sent.json \
&& echo "add one sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifySample.json \
${expname}/predictions_add_one_sent.json
}

function runptsquad {

export SQUAD_DIR=${PT_DATA_DIR}

custom=${1}   # Custom name
gpu=${2}      # number of GPU
bsize=${3}    # Batch size
beta=${4}    # regularizer coefficient
version=${5} # mi estimator version
hdp=${6}     # Hidden layer dropouts for ALBERT
adp=${7}     # Attention dropouts for ALBERT
alr=${8}      # Step size of gradient ascent
amag=${9}     # Magnitude of initial (adversarial?) perturbation
anorm=${10}    # Maximum norm of adversarial perturbation
asteps=${11}   # Number of gradient ascent steps for the adversary
mname=${12}
alpha=${13}
cl=${14}
ch=${15}
if [ -z "${14}" ] ;then
    cl=0.5
fi

if [ -z "${15}" ] ;then
    ch=0.9
fi
#beta=${11}    # regularizer coefficient
#version=${12} # mi estimator version
#hdp=${13}     # Hidden layer dropouts for ALBERT
#adp=${14}     # Attention dropouts for ALBERT
#alr=${15}      # Step size of gradient ascent
#amag=${16}     # Magnitude of initial (adversarial?) perturbation
#anorm=${17}    # Maximum norm of adversarial perturbation
#asteps=${18}   # Number of gradient ascent steps for the adversary
export seqlen=384
export lr=3e-5
export mname=bert-large-uncased-whole-word-masking

#mname=${3}    # Model name
#lr=${4}       # Learning rate for model parameters
#seqlen=${6}    # Maximum sequence length
#ts=${7}      # Number of training steps (counted as parameter updates)
#ws=${8}      # Learning rate warm-up steps
#seed=${9}    # Seed for randomness
#wd=${10}      # Weight decay


#expname=${custom}-${mname}-${TASK_NAME}-sl${seqlen}-lr${lr}-bs${bsize}-ts${ts}-ws${ws}-wd${wd}-seed${seed}-beta${beta}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-version${version}
expname=${PT_OUTPUT_DIR}/${custom}-${mname}-sl${seqlen}-lr${lr}-bs${bsize}-beta${beta}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-alpha${alpha}-cl${cl}-ch${ch}-version${version}

max=-1
for file in ${expname}/checkpoint-*
do
  fname=$(basename ${file})
  num=${fname:11}
  [[ $num -gt $max ]] && max=$num
done

if [ $max -eq -1 ]
then echo "Train from stratch"
else mname="${expname}/checkpoint-$max" && echo "Resume Training from checkpoint $mname"
fi

python -m torch.distributed.launch --nproc_per_node=${gpu} ./run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking  --overwrite_output_dir  \
    --do_train \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 --save_steps 100 \
    --output_dir ${expname} --data_dir ${SQUAD_DIR}\
    --per_gpu_eval_batch_size=${bsize}   \
    --per_gpu_train_batch_size=${bsize}  \
    --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
    --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
    --beta ${beta}  --alpha ${alpha}  --cl ${cl} --ch ${ch}  \
    --version ${version} \
&& echo "add sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifyAll.json \
${expname}/predictions_add_sent.json \
&& echo "add one sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifySample.json \
${expname}/predictions_add_one_sent.json
}

function evalsquad {


export SQUAD_DIR=squad_data

custom=${1}   # Custom name
gpu=${2}      # number of GPU
bsize=${3}    # Batch size
beta=${4}    # regularizer coefficient
version=${5} # mi estimator version
hdp=${6}     # Hidden layer dropouts for ALBERT
adp=${7}     # Attention dropouts for ALBERT
alr=${8}      # Step size of gradient ascent
amag=${9}     # Magnitude of initial (adversarial?) perturbation
anorm=${10}    # Maximum norm of adversarial perturbation
asteps=${11}   # Number of gradient ascent steps for the adversary
mname=${12}
alpha=${13}
#beta=${11}    # regularizer coefficient
#version=${12} # mi estimator version
#hdp=${13}     # Hidden layer dropouts for ALBERT
#adp=${14}     # Attention dropouts for ALBERT
#alr=${15}      # Step size of gradient ascent
#amag=${16}     # Magnitude of initial (adversarial?) perturbation
#anorm=${17}    # Maximum norm of adversarial perturbation
#asteps=${18}   # Number of gradient ascent steps for the adversary
export seqlen=384
export lr=3e-5
#export mname=bert-large-uncased-whole-word-masking

#mname=${3}    # Model name
#lr=${4}       # Learning rate for model parameters
#seqlen=${6}    # Maximum sequence length
#ts=${7}      # Number of training steps (counted as parameter updates)
#ws=${8}      # Learning rate warm-up steps
#seed=${9}    # Seed for randomness
#wd=${10}      # Weight decay
if [[ ${mname} == *"roberta"* ]]; then
  model_type=roberta
else
  model_type=bert
fi

#expname=${custom}-${mname}-${TASK_NAME}-sl${seqlen}-lr${lr}-bs${bsize}-ts${ts}-ws${ws}-wd${wd}-seed${seed}-beta${beta}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-hdp${hdp}-adp${adp}-version${version}
expname=${custom}-${mname}-load

#python -m torch.distributed.launch --nproc_per_node=${gpu} ./run_squad.py \
python run_squad.py \
    --model_type ${model_type} \
    --model_name_or_path ${mname} \
    --do_eval \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 100 --save_steps 5000 \
    --output_dir ${expname} --data_dir ${SQUAD_DIR}\
    --per_gpu_eval_batch_size=${bsize}   \
    --per_gpu_train_batch_size=${bsize}  \
    --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
    --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
    --beta ${beta} --alpha ${alpha} \
    --version ${version} \
&& echo "add sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifyAll.json \
${expname}/predictions_add_sent.json \
&& echo "add one sent" && python2 eval_adv_squad.py squad_data/sample1k-HCVerifySample.json \
${expname}/predictions_add_one_sent.json
}