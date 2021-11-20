import json
import yaml
import time
import pandas as pd
import xtrack as xt
import xpart as xp
import sys
import help_functions as hf
import tree_maker

with open('config.yaml','r') as fid:
    config=yaml.safe_load(fid)

tree_maker.tag_json.tag_it(config['log_file'], 'started')

with open(config['xline_json']) as fid:
    dd=json.load(fid)

WW = dd['WW_finite_diffs']
invWW = dd['WWInv_finite_diffs']
closed_orbit = dd['particle_on_tracker_co']

p_co = xp.Particles.from_dict(dd['particle_on_tracker_co'])
line = xt.Line.from_dict(dd)


egeom_1 = config['epsn_1'] / p_co.gamma0 / p_co.beta0
egeom_2 = config['epsn_2'] / p_co.gamma0 / p_co.beta0

particle_df=pd.read_parquet(config['particle_file'])

init_canonical_6D, A1_A2_in_sigma, number_of_particles = hf.from_normal_to_physical_space(
    particle_df,
    egeom_1=egeom_1,
    egeom_2=egeom_2,
    ptau_max=config['ptau_max'],
    W=WW,
    invW=invWW,
)

pp = hf.add_to_closed_orbit(init_canonical_6D, p_co, particle_id=particle_df['particle_id'].values)

line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)
line.merge_consecutive_multipoles(inplace=True)

tracker = xt.Tracker(line=line)
particles = xp.Particles(**pp.to_dict())
pd.DataFrame(particles.to_dict()).to_parquet('input_particles.parquet')

num_turns = config['n_turns']
a=time.time()
tracker.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
b=time.time()

print(f'Elapsed time: {b-a} s')
print(f'Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us')

pd.DataFrame(particles.to_dict()).to_parquet('output_particles.parquet')
tree_maker.tag_json.tag_it(config['log_file'], 'completed')
