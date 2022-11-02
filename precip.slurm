#!/bin/bash
#SBATCH   --partition=normal             # submit   to the normal(default) partition
#SBATCH   --job-name=cmip6test             # name the job
#SBATCH   --output=cmip.out        # write stdout/stderr   to named file
#SBATCH   --error=cmip.err      
#SBATCH   --time=0-02:00:00             # Run for max of 02 hrs, 00 mins, 00 secs
#SBATCH   --nodes=1               # Request N nodes
#SBATCH   --ntasks=1
#SBATCH   --cpus-per-task=48
#SBATCH   --mem=192GB             # Request nGB RAM per core



~/.conda/envs/testenv/bin/python run_precip.py
