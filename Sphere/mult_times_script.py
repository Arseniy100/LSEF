# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:34:13 2022
last modified: Oct 2022

@author: Arseniy Sotskiy
"""
import os

os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '4' # export NUMEXPR_NUM_THREADS=6


import numpy as np
import typing as tp

print(1)

def is_seed_done(seed: int, done_seeds: str):
    try:
        with open(done_seeds, 'r') as done:
            if str(seed) in done.read().split():
                return True
    except FileNotFoundError:
        with open(done_seeds, 'a+') as done:
            pass
    return False

def are_all_seeds_done(all_seeds: tp.List[int], done_seeds: str):
    try:
       with open(done_seeds, 'r') as done:
           if len(done.read().split()) == len(all_seeds):
               return True
    except FileNotFoundError:
        with open(done_seeds, 'a+') as done:
            pass
    return False


params_for_cycles = [ # dx must be (n_lat - 1) in configs
    # ('e_size', [5, 10, 20, 40, 80], 'int'),
    # ('kappa_default', [1, 1.5, 2, 3, 4], 'float'), # [1, 1.5, 2, 3, 4]
    ('mu_NSL', [1, 2, 3, 5, 10], 'int'), # [1, 2, 3, 5, 10]
    # ('n_max', [49], 'int')
    # ('obs_std_err', [0.5, 1, 2], 'float'), 
    # ('obs_per_grid_cell',  [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 4], 
    # 'float'),
    #   ('q_tranfu', [2, 3], 'int'),
    #   ('nc2_multiplier', [1, 1.25, 1.5, 1.75, 2], 'float'),
    #   ('halfwidth_multiplier', [0.7, 1, 1.3, 1.6], 'float'),

      #  ('nn_i_multiplier', [0,1,2,3], 'int'),
      #  ('batch_size', [4,8,16,32,64,128], 'int'),
      #  ('lr', [1e-1, 1e-2, 1e-3, 1e-4], 'float'),
      #  ('momentum', [0.1, 0.2, 0.3, 0.4, 0.6, 0.8], 'float'),
      #  ('activation_name', 
      #   ['SquareActivation', 'AbsActivation', 'ExpActivation'], 'str'),
      #  ('non_linearity', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'], 'str'),
      # ('n_epochs', [1,2,3,4,5,6,7,8], 'int')
    ]

# params_for_cycles = [
#     # ('w_ensm_hybr', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'float'),
#     ('w_ensm_hybr', [0.1], 'float'),
#     ] # nohup python nn_train.py kappa_default 2 float END &



# params_for_cycles = [
#     ('n_training_spheres_B', [10], 'int'), # 10, 20, 40, 100,  200, 500
#     ]


# halfwidth_multiplier_grid = [0.7, 1, 1.3]
# e_size_grid = [5,10,20,40,80]
# halfwidth_multiplier_grid = [0.7, 1]
# e_size_grid = [5]
is_nn_trained_once: int = 1
# expq_bands_params

# permition_to_begin = input('long os script will run; type y if yes: ')
permition_to_begin = 'y'
if permition_to_begin.lower() != 'y':
    print('operation aborted')
else:
    # if is_nn_trained_once:
        # os.system("python nn_train.py is_nn_trained 1 int END")   

    with open('all_seeds.txt', 'r') as all_seeds:
        all_seeds = [int(seed) for seed in all_seeds.read().split()]
        print(all_seeds)
    
        
    for param_name, param_values, param_type in params_for_cycles:
        for val in param_values:
            current_seeds_file = f'all seeds {param_name} {val}.txt'
            if are_all_seeds_done(all_seeds, current_seeds_file):
                continue
            os.system(
                f"python nn_train.py {param_name} {val} {param_type}" + \
                    f" is_nn_trained {1 - is_nn_trained_once} int END"
                )
            for seed in all_seeds:
                if is_seed_done(seed, current_seeds_file):
                    continue
                print(f"python Main.py my_seed_mult {seed} int" +\
                    f" {param_name} {val} {param_type} END")
                os.system(
                    f"python Main.py my_seed_mult {seed} int" +\
                    f" {param_name} {val} {param_type} END"
                    )
                
    # for halfwidth_multiplier in halfwidth_multiplier_grid:
    #     for e_size in e_size_grid:
    #         os.system(f"python Main.py halfwidth_multiplier " +\
    #                   f"{halfwidth_multiplier} float " +\
    #                   f"e_size {e_size} int END")

# os.system("python Main.py END")


# for i in range(100):
#     print(np.random.randint(424242), end=' ')

# with open('ff.txt', 'a+') as f:
#     print(f.read())
#     f.write('a\n')
# git 

