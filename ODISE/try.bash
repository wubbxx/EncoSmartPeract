#!/bin/bash
# 加载 conda
source ~/miniconda3/etc/profile.d/conda.sh

for env in $(conda env list | awk '{print $1}' | grep -v "#")
do
    echo "Environment: $env"
    conda activate $env
    python --version
done

