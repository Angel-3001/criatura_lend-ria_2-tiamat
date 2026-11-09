#!/bin/bash
#SBATCH --job-name=Angel-Tiamat
#SBATCH --output=Angel-Tiamat_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=16:00:00
 
# Execução do script Python no ambiente virtual configurado
/home/abraham2610048/env_analise/bin/python /home/abraham2610048/aprendizado_maquina/job.py