# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import json
import logging
import os
import time

# Import third-party modules
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

# Import user-defined modules
import xmask as xm

import xobjects as xo
import xtrack as xt
import matplotlib.ticker as ticker
sns.set_theme(style="ticks")
import glob
# %%
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_try_gpu/**/*')
# %%
df_all = pd.DataFrame()
for file in files:
    df = pd.read_parquet(file)
    df_all = pd.concat([df_all, df], ignore_index=True)
    
# %%
for turn in np.unique(df_all['at_turn']):
    print(turn)
    df = df_all[df_all['at_turn'] == turn]
    #plt.figure()
    sns.histplot(data=df, x='x_norm', y='px_norm', bins=100)
    

# %%
