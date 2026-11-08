#!/bin/bash
#SBATCH --job-name=tiamat_stacking_angel
#SBATCH --output=Angel-Tiamat_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

# Execução correta chamando o python:
python3 /home/angel2610002/aprendizado_de_maquina/Stacking/Otimizacao_Optuna_Stacking.py