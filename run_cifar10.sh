#!/bin/sh
#Settings
extra_components=$1
model_name=$2
arch=$3
gpu=$4
clamp=$5
epoch=100
gamma=1
batch_size=128

if [[ ${clamp} -eq  1 ]]
then
    nohup python main.py -m ${model_name} -a ${arch} -d cifar10 -c all ${extra_components} -e ${epoch} -g ${gpu} --clamp -b ${batch_size} --gamma ${gamma}> ${model_name}_clamp_cifar10.out &
else 
    nohup python main.py -m ${model_name} -a ${arch} -d cifar10 -c all ${extra_components} -e ${epoch} -g ${gpu} -b ${batch_size} --gamma ${gamma} > ${model_name}_cifar10.out &
fi

