#!/bin/bash
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/fma_local.py .
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/fits_sep_adjust.parquet .
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json .
source /afs/cern.ch/work/a/aradosla/private/my_env/bin/activate
python fma_local.py > outputfma_python.txt 2> errorfma_python.txt
cp outputfma_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/outputfma_python.txt
cp errorfma_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/errorfma_python.txt
