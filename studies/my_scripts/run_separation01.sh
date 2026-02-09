#!/bin/bash
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/separation_adjust.py .
cp /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json .
source /afs/cern.ch/work/a/aradosla/private/example_DA_study_mine/miniforge/bin/activate
python separation_adjust.py > output_python.txt 2> error_python.txt
cp output_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/output_python.txt
cp error_python.txt /afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/error_python.txt
