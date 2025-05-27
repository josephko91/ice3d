#!/bin/bash
#PBS -A UCLB0047
#PBS -N move_files
#PBS -o ./out/move-files/out-01.log 
#PBS -e ./err/move-files/err-01.log
#PBS -m abe
#PBS -q main  
#PBS -l select=1:ncpus=1:mem=10GB
#PBS -l walltime=04:00:00
#PBS -V

