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
# %%
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_try_gpu/**/*')
# %%
df_all = pd.DataFrame()
for file in files:
    df = pd.read_parquet(file)
    df_all = pd.concat([df_all, df], ignore_index=True)
    
# %%
counter = 0
color = ['r', 'y', 'b', 'g', 'c', 'm', 'k', 'black', 'orange', 'purple', 'brown', 'pink', 'gray']
for turn in np.unique(df_all['at_turn']):
    print(turn)
    counter = counter + 1
    df = df_all[df_all['at_turn'] == turn]
    #plt.figure()
    sns.histplot(data=df, x='x_norm', y='px_norm', bins=100, color = color[counter])
    

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume df_all is already defined and loaded with the necessary data
# Example: df_all = pd.read_csv('your_data.csv')

# Initialize counter and color list
counter = 0
color = ['r', 'y', 'b', 'g', 'c', 'm', 'k', 'black', 'orange', 'purple', 'brown', 'pink', 'gray']

# Get the unique 'at_turn' values
unique_turns = np.unique(df_all['at_turn'])

# Create a figure with subplots
fig, axes = plt.subplots(1, len(unique_turns), figsize=(20, 5), sharex=True, sharey=True)

# Loop over each unique turn and plot histograms
for i, turn in enumerate(unique_turns):
    print(turn)
    df = df_all[df_all['at_turn'] == turn]
    sns.jointplot(data=df, x='x_norm', y='px_norm', color=color[counter], ax=axes[i])
    axes[i].set_title(f'Turn {turn}')
    counter = (counter + 1) % len(color)  # Use modulo to cycle through colors if needed
    plt.title(f'Turn {turn}')
    axes[i].set_xlim(-1.2, 1.2)
plt.tight_layout()
plt.show()


# %%

# %%
