import os

path_scans = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/all_optics_2024_reverted"

for folder in os.listdir(path_scans):
    if "collider" in folder:
        path_folder = os.path.join(path_scans, folder)
        os.chdir(path_folder)
        os.system("python 1_build_distr_and_collider.py")
        path_xtrack = os.path.join(path_folder, "xtrack_0000")
        os.chdir(path_xtrack)
        os.system("python 2_configure_and_track.py")
