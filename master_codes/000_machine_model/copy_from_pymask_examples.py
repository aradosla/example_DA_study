from shutil import copyfile
import pymask

example_folder = pymask._pkg_root.parent.joinpath(
   'python_examples/hl_lhc_collisions_python/')

files_to_copy = ['000_pymask.py',
                'optics_specific_tools.py']#, 'config.yaml']

destination_path = '.'

for filename in files_to_copy:
    copyfile(example_folder.joinpath(filename),
        f'{destination_path}/{filename}')


