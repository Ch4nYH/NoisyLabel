#!/bin/sh
#Settings
extra_components=$1
model_name=$2
gpu=$3
clamp=$4
epoch=100
gamma=0.05

if [[ $clamp -eq  1 ]]
then
    nohup python main.py -m ${model_name} -d cifar10 -c all ${extra_components} -e 100 -g ${gpu} --clamp > ${model_name}_clamp_cifar10.out &
else 
    nohup python main.py -m ${model_name} -d cifar10 -c all ${extra_components} -e 100 -g ${gpu} > ${model_name}_cifar10.out &
fi

