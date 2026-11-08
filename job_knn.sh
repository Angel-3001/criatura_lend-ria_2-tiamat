#!/bin/bash
#SBATCH --job-name=tiamat_knn_angel
#SBATCH --output=Angel-Tiamat_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
 
# Se você usar um ambiente virtual (conda ou venv), ative-o aqui antes:
# source /caminho/para/seu/venv/bin/activate
# ou conda activate nome_do_ambiente

# Execução correta chamando o python:
python3 /home/angel2610002/aprendizado_de_maquina/k-NN/Otimizacao_Optuna_kNN.py