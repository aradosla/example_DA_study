import pandas as pd
import subprocess
from io import StringIO
import tree_maker as tm

def get_jobs():
    out = subprocess.Popen(('ps -ef '
                             '| grep "run.sh" '
                             '| grep -v grep'),
    stdout = subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    if stdout == b'':
        return pd.DataFrame()
    else:
        my_df = pd.read_csv(StringIO(stdout.decode('UTF-8')), sep=' ')
        my_df = my_df.set_index('COMMAND')
        my_df.index.name = None
        return my_df
