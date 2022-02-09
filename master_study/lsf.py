import pandas as pd
import subprocess
from io import StringIO
import tree_maker as tm

def bjobs():
    out = subprocess.Popen(['bjobs','-o', ('jobid '
                                           'stat '
                                           'command '
                                           'user '
                                           'from_host '
                                           'exec_host '
                                           'job_name '
                                           'submit_time '
                                           'queue '
                                           'project '
                                           'application '
                                           'mem '
                                           'delimiter=","')],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    if stdout == b'No unfinished job found\n':
        return pd.DataFrame()
    else:
        my_df = pd.read_csv(StringIO(stdout.decode('UTF-8')), sep=',')
        my_df = my_df.set_index('COMMAND')
        my_df.index.name = None
        return my_df

def create_df(node):
    """
    Creating a dataframe and its attributes.
    Here its attributes are; node, path, x_values, y_values and my_colors.
    """
    my_df = pd.DataFrame([node]+list(node.descendants),
                         columns=['handle']).copy()
    my_df['name'] = my_df['handle'].apply(lambda x:x.name)
    my_df['path'] = my_df['handle'].apply(lambda x:x.get_abs('path'))
    my_df['status'] = my_df['handle'].apply(get_status)
    return my_df
