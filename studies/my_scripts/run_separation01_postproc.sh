#!/bin/bash
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/postproc_per_turn.py .
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/fits_sep_adjust.parquet .
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json .
source /afs/cern.ch/work/a/aradosla/private/example_DA_study_mine/miniforge/bin/activate
python postproc_per_turn.py > outputpostproc_python.txt 2> errorpostproc_python.txt
cp outputfma_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/outputpostproc_python.txt
cp errorpostproc_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/errorpostproc_python.txt
