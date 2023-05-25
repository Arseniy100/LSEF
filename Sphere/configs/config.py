# -*- coding: utf-8 -*-

"""
last modified: Feb 2023

@author: Arseniy Sotskiy
"""

import sys
from numpy import pi

R_km = 6370  # radius of the Earth, km

# program works in 2 modes: 
# linux server (we assume that it can do computations with bigger arrays)
# and pc (less computational speed)

sys_path_with_linux = \
    '/RHM-Lustre3.2/software/general/plang/Anaconda3-2019.10'

if sys_path_with_linux in sys.path:
    is_Linux = True
else:
    is_Linux = False

if is_Linux:
    print('working on super-computer')
    n_max = 49
else:
    print('working on personal computer')
    n_max = 49
    
angular_distance_grid_size = n_max * 10 + 1
my_seed_mult = 5757  # multiplicator for random seed

n_lat: int = n_max + 2
# our_n_max = (n_max+1)


dx: float = 2 * pi / (2 * (n_lat - 1)) # in radians dx must be (n_lat - 1)

k_coarse: int = 2  # should divide n_max + 1

rho_grid_size: int = 10 * n_max

e_size: int = 20 # size of the ensemble 

band_1_width: int = 5

statio: bool = False
gamma_fixed: bool = False
kappa_default: float = 2.

mu_NSL: float = 3. # 3.

L_0: float = 2500. # default length scale for lcz matrix
        #750 #3500  #2500 for statio n_x=60  #250 for DLSM n_x=360        

SDxi_med = 1       # median of the statio fld S(x)  
SDxi_add = SDxi_med /10 # minimal SDxi
lambda_add = dx/3    # minimal lambda(x): accounts for the grid resolution
lambda_med = dx*3  # median lambda(x): the desired median length scale 
gamma_add = 1.0      # minimal gamma(x): avoid too low gamma ==> weird crf
gamma_med = 4.0    # S1: 3.0 yields almost AR2 cvf (S2: gamma=3.1 yields AR2)

# !!!!!!!!! dx * / 2, not 3 

is_best_c_loc_needed: bool = True # c_loc for lcz matrix
# c_loc: int = 2750
# c_loc = 1000 + 2000 / 6 * lamb_mult / dx
c_loc = None

# observations of the anls:
n_lon: int = (n_lat - 1) * 2
npoints: int = n_lat * n_lon - 2 * n_lon + 2
obs_per_grid_cell: float = 0.5

obs_std_err: float = 1. # 0.5, 0.7, 1, 1.3, 1.5

n_seeds: int = 1

n_training_spheres: int = 33 # for nn dataset
n_training_spheres_B: int = 33 # for B_static
is_nn_trained: int = 1

w_smoo: float = 1e-4 # for neural network 

w_ensm_hybr: float = 0.5 #  for hybrid B_mean
hybrid_type: str = 'mean' # or 'med'
threshold_coeff: float = 0. # for sparsing matrix

# bands:
halfwidth_multiplier: float = 1
nc2_multiplier: float = 1.5

q_tranfu: int = 3 # 2, 3

n_iterations: int = 1



nn_i_multiplier = 0
momentum = 0.1
lr = 1e-1
batch_size = 16
activation_name = 'ExpActivation'
non_linearity = 'ReLU'
n_epochs = 3

# if is_Linux:
#     cycle_mode_on: bool = True
# if not is_Linux:
#     cycle_mode_on: bool = True
info = ''
folder_descr = " main" 
n_params = 0

try:
    while True:
        cycle_param_name = sys.argv[n_params * 3 + 1]
        if cycle_param_name == 'END':
            break
        cycle_param_value = sys.argv[n_params * 3 + 2]
        cycle_param_type = sys.argv[n_params * 3 + 3]
        n_params += 1
        assert cycle_param_name in globals(), 'param does not exist'
        exec(f'{cycle_param_name} = {cycle_param_type}("{cycle_param_value}")')
        # e_size = int(10)
        # globals()[cycle_param_name] = cycle_param_value # !!!!! 
        # be very accurate with this!!!!!!!!!
        folder_descr += f" {cycle_param_name} {cycle_param_value}" 
    
        info = f' {cycle_param_name} {cycle_param_value}'
except IndexError:
    pass


n_obs: int = int(obs_per_grid_cell * npoints)
# print(n_obs)
if statio:
    # parameters for the processes V[x], L[x], gamma[x] such that
    kappa_SDxi = 1.# V[x] = V*g(log(kappa_V)*WN_V(x,NS_width))
    kappa_lamb = 1.  # L[x] = L_min + L*g(log(kappa_L)*WN_L(x,NS_width))
    kappa_gamma = 1. # gamma[x] = gamma_add + gamma_mult*g(log(kappa_gamma)*WN_gamma(x,NS_width))
else:
    kappa_SDxi = kappa_default # V[x] = V*g(log(kappa_V)*WN_V(x,NS_width))
    kappa_lamb = kappa_default  # L[x] = L_min + 
                                # L*g(log(kappa_L)*WN_L(x,NS_width))
    if gamma_fixed:
        kappa_gamma = 1.
    else:
        kappa_gamma = kappa_default
        # gamma[x] = gamma_add + 
        # gamma_mult*g(log(kappa_gamma)*WN_gamma(x,NS_width))

expq_bands_params = {
    'n_max': n_max, 'nband': 6, 
    'halfwidth_min': 1.5  * n_max/30 * halfwidth_multiplier, 
    'halfwidth_max': 1.2 * n_max / 4 * halfwidth_multiplier,
    'nc2': nc2_multiplier * n_max / 22, 
    'q_tranfu': q_tranfu, 'rectang': False
    }


# net_name = f'net_l2'
net_name = 'net_gridded'


