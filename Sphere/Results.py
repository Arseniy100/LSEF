# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:07:23 2022
Last modified: Mar 2022

@author: Arseniy Sotskiy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from collections import namedtuple

from functions.tools import draw_surface

# ident = 

beautiful_names_dict = {
    't': 'True-B',
    'm': 'Mean-B',
    's': 'EnKF-B',
    'h': 'Hybrid-B',
    
    'obs_std_err': 'Std of observation error',
    'n_obs': 'Number of observations',
    'e_size': 'Ensemble size',
    'kappa_default' : r'Non-stationarity strength $\varkappa$',
    'mu_NSL': r'Non-stationarity length parameter $\mu_{\rm NSL}$',
    'obs_per_grid_cell': 'Number of observations per grid point',
    'activation_name': 'Output activation function of the net',
    
    'NN net_gridded59': 'LSEF-B',
    'NN net_l2_sqrtsqrt49': 'LSEF-B',
    'NN net_gridded29': 'LSEF-B NN',
    'NN net5_log29': 'LSEF2-B NN',
    'NN net_l249': 'LSEF-B L2 loss',
    'SVShape_modal': 'LSEF-B no NN'
    }

def beautiful_name(name: str) -> str:
    if name in beautiful_names_dict.keys():
        return beautiful_names_dict[name]
    return name


def rescale_error(err: np.array, t: np.array) -> np.array:
    # print('err', err, 't', t)
    # min_err_t_len = min(len(err), len(t))
    # # print('min_err_t_len', min_err_t_len)
    # err = err[-min_err_t_len:]
    # t = t[-min_err_t_len:]
    # # print((err - t) / t)
    return (err - t) / t


def find_conf_bootstrap(sample: np.array, rescale: np.array=None, k: int=10000,
                        alpha: float=0.1, root: bool=False) -> float:
    if np.var(sample) == 0:
        return 0
    ids = np.random.choice(len(sample), (len(sample), k)) 
    # print(sample)
    bootstrap_sample = np.array(sample)[ids].mean(axis=0)
    if root:
        bootstrap_sample = np.sqrt(bootstrap_sample)
    if rescale is not None:
        bootstrap_rescale = np.array(rescale)[ids].mean(axis=0)
        if root:
            bootstrap_rescale = np.sqrt(bootstrap_rescale)
        bootstrap_sample = rescale_error(bootstrap_sample, bootstrap_rescale)
    # plt.hist(bootstrap_sample, bins=20)
    # plt.show()
    return (np.quantile(bootstrap_sample, 1 - alpha / 2) - \
        np.quantile(bootstrap_sample,  alpha / 2) ) / 2

if __name__ == '__main__':
    print(find_conf_bootstrap(np.arange(10), k=30000))
    print(np.arange(10).std() / np.sqrt(10))
    


    
#%%

R_s_R_m = 'R_s R_m.txt'
# R_s_R_m = 'a.txt'
# each line is as folows: '{name}: R_{..} = {..}'
           # for example: 'SVShape: R_s = 0.07667631449130448'


# predictors_names = ['net_deep', 'net2', 'net3', 'SVShape']

Param_grid = namedtuple('Param_grid', ['name', 'grid'])

    

def draw_results_1d(params_for_cycles: tuple,
                    source_folder: str,
                    error_types: list,
                    predictors_names: list,
                    all_err_colors: dict,
                    all_err_styles: dict,
                    predictor_marker: str='l',
                    rescale: bool=True):
    all_err_types = all_err_colors.keys()

    for param_name, param_values, param_type in params_for_cycles:
        print('========\n'*5)
        print(param_name)
        print('========\n'*2)
    
        
        error_results = dict()
        
        
        for param_val in param_values:
            error_results[param_val] = dict()
            for err_type in error_types:
                error_results[param_val][err_type] = list()
            for name in predictors_names:
                error_results[param_val][name] = list()
            for folder_name in os.listdir(source_folder):
                # print(folder_name)
                # continue
                source = source_folder + folder_name
                name_val_str = f'{param_name} {param_val}'
                if os.path.isdir(source) and \
                    name_val_str == folder_name[-len(name_val_str):]:
                    # print(folder_name)
                    # try:
                    #     param_val = float(folder_name.split()[-1])
                    # except ValueError:
                    #     param_val = str(folder_name.split()[-1])
                    # if param_val not in param_values:
                    #     continue
                    
                    
                    if not os.path.isfile(source + '\\' + R_s_R_m):
                        print('no file in {source}')
                        continue
                    with open(source + '\\' + R_s_R_m) as errors_file:
                        for line in errors_file:
                            print(line)
                            
                            # print(line.split())
                            err = line.split()[-1]
                            
                            if err != 'None':
                                err = float(err)
                                if err > 1:
                                    print(source)
                                    print(line)
                                for err_type in error_types:
                                    if f' {err_type} ' in line:
                                        error_results[param_val][err_type].append(err)
                                        
                                if f' {predictor_marker} ' in line:
                                    for predictor in predictors_names:
                                        if predictor in line:
                                            error_results[param_val][predictor].append(err)
        
        for err_type in error_types:
            for param_val in param_values:
        #         print(err_type, param_val)
                # delete duplicates
                error_results[param_val][err_type] = \
                    np.array(
                        error_results[param_val][err_type][::len(predictors_names)]
                        )
        for err_type in all_err_types:
            print(err_type)
            
            for param_val in param_values:
                # rescale: (a - t) / t
                err = np.array(error_results[param_val][err_type])
                t = np.array(error_results[param_val]['t'])
                # print('\n'*7)
                # print(err)
                # print(t)
                # print(error_results)
                
                # print(err_type, param_val, err)
                # print(param_val, err, t)
                if len(t) > 0:
                    min_err_t_len = 20
                    # err = err[min_err_t_len:]
                    # t = t.mean()
                    # t = t[min_err_t_len:]
                    
                    # if rescale:
                    #     pass
                        # if err_type == 't':
                            # print(err)
                            # print(t)
                            # print((err - t) / t)
                        # error_results[param_val][err_type] = rescale_error(err, t)
                        
                        # print(err_type, err[error_results[param_val][err_type] > 1])
                        # print('true', t[:len(err)][error_results[param_val][err_type] > 1])
                print(f'{param_name} {param_val} - {len(err)} points')
                # print(error_results[param_val][err_type])
 
        
        
        
        # for param_val in param_values:
        #     error_results[param_val]['t'] = np.zeros(
        #         len(error_results[param_val]['t'])
        #         )
            
        # print('\n'*3, error_results)
                            
        mean_results = dict()
        std_results = dict()
        for err_type in all_err_types:
            mean_results[err_type] = np.sqrt([
                np.mean(np.array(
                    error_results[param_val][err_type]
                    )**2)\
                    for param_val in param_values
                ])

            # std_results[err_type] = np.array([
            #     np.std(error_results[param_val][err_type]) \
            #         for param_val in param_values
            #     ]) / np.sqrt(len(error_results[param_val][err_type]))
            # print(err_type)
            # print(len(error_results[param_val][err_type]))
            # print(len(error_results[param_val]['t']))
            std_results[err_type] = np.array([
                find_conf_bootstrap(error_results[param_val][err_type],
                                    rescale=error_results[param_val]['t']) \
                    for param_val in param_values
                ]) 
        if rescale:
            for err_type in all_err_types:
                mean_results[err_type] = rescale_error(mean_results[err_type], 
                                                       mean_results['t'])
                
        # print('\n' * 10, 
        #       error_results[param_val][err_type],
        #       '\n'*3,
        #       std_results)
        # assert False, 'ok'
        print(mean_results)
    
        fig, ax = plt.subplots(figsize=(10,10))
        xticks = np.arange(len(param_values))
        # ax.plot(x, y_est, '-')
        # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
        # ax.plot(x, y, 'o', color='tab:brown')
        t = mean_results['t']
        for err_type in all_err_types:
            color = all_err_colors[err_type]
            linestyle = all_err_styles[err_type]
            
            err = mean_results[err_type]
            # print(err_type, err)
            # err_rescaled = (err - t) / t
            ax.plot(
                # np.arange(2), [err[0], err[0]], # if only one param val
                xticks, err, 
                label=beautiful_name(err_type), 
                color=color, linestyle=linestyle
                )
            ax.fill_between(xticks, 
                            # err_rescaled - std_results[err_type] / t, 
                            # err_rescaled + std_results[err_type] / t, 
                            err - std_results[err_type], 
                            err + std_results[err_type], 
                            alpha=0.2, color=color)
            # for i, param_val in enumerate(param_values):
            #     # if err_type == net_2_name:
            #     ax.scatter(
            #         [i for _ in \
            #           error_results[param_val][err_type]],
            #                 error_results[param_val][err_type],
                           
            #                 color=color, s=1)
            
                # print(param_val,'m', np.sort(error_results[param_val]['m']))
        ax.grid(True)
        # for p in param_values:
        #     for q in param_values:
        #         print(p, q, np.max(np.abs(
        #             np.sort(error_results[p]['m']) - \
        #                 np.sort(error_results[q]['m'])
        #             )))
        ax.set(xticks=xticks,
               xticklabels=param_values)
        ax.set_xlabel(f'{beautiful_name(param_name)}', fontsize=20)
        # ax.set_xlabel(f'iteration', fontsize=20)
        ax.set_ylabel('($RMSE - RMSE_{True B})\quad /\quad RMSE_{True B}$', fontsize=20)
        ax.legend(loc = 'best', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        plt.title(r'$S^2$ : Analysis error RMSEs', 
                     fontsize=22)
        # fig.suptitle('Analysis RMSE relative to optimal analysis RMSE', 
        #              fontsize=20)
        fig.savefig(f'{source_folder}{param_name}.png')
        fig.savefig(f'{source_folder}{param_name}.pdf')
        plt.show()
        # print(error_results[param_val]['m'])
        # print(mean_results)
        
    return error_results


# from mult_times_script import params_for_cycles
# params_for_cycles = [
    # ('e_size', [5,10,20,40,80], 'int'),
    # ('kappa_default', [1, 1.5, 2, 3, 4], 'float'),
    # ('mu_NSL', [1, 2, 3, 5, 10], 'int'),
    
    # ('obs_std_err', [0.5, 1, 2], 'float'), 
    # ('obs_per_grid_cell',  [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2], 
    #   'float'),
    # ('q_tranfu', [2, 3], 'int'),
    # ('nc2_multiplier', [0.5, 0.75, 1, 1.25, 1.5], 'float'),
    # ('nn_i_multiplier', [0,1,2,3], 'int'),
    # ]

params_for_cycles = [
    ('e_size', [5,10,20,40,80], 'int'),
    ('kappa_default', [1, 1.5, 2, 3, 4], 'float'),
    ('mu_NSL', [1, 2, 3, 5, 10], 'int'),
    
    ('obs_std_err', [0.5, 1, 2], 'float'), 
    ('obs_per_grid_cell',  [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 4], 
    # ('obs_per_grid_cell',  [0.1], 
    'float'),
      ('q_tranfu', [2, 3], 'int'),
      ('nc2_multiplier', [1, 1.25, 1.5, 1.75, 2], 'float'),
      ('halfwidth_multiplier', [0.7, 1, 1.3, 1.6], 'float'),
#     ]

# params_for_cycles = [
       ('nn_i_multiplier', [0,1,2,3], 'int'),
       ('batch_size', [4,8,16,32,64,128], 'int'),
       ('lr', [1e-1, 1e-2, 1e-3, 1e-4], 'float'),
       ('momentum', [0.1, 0.2, 0.3, 0.4, 0.6, 0.8], 'float'),
       ('activation_name', 
        ['SquareActivation', 'AbsActivation', 'ExpActivation'], 'str'),
       ('non_linearity', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'], 'str'),
      ('n_epochs', [1,2,3,4,5,6,7,8], 'int')
    ]

# params_for_cycles = [
#     ('e_size', [10, 11, 12, 13, 14, 15, 16, 17,
#                 18,19,20,2s1,22,23], 'int')
#     ]




# results_dir = 'images\\2022_03_04_21_28_31 main 20 iters\\'
source_folder = r"images\\Перебор final__\\"
# net_name = 'NN net_gridded59' 
# net_name = 'NN net_gridded29' 
# net_name = 'NN net_gridded49' 
# net_name = 'NN net_gridded47' 
# net_name = 'NN net7_anls_loss'
# net_two_name = 'NN net5_log29' 
# net_2_name = 'NN net_l229'
net_name = 'NN net_l2_sqrtsqrt49' 
net_3_name = 'NN net_l2_sqrtsqrt_12049' 
net_4_name = 'NN net_l2_sqrt49' 


print(source_folder)

predictors_names = [
    # 'SVShape_modal',
    # net_name,
    # net_two_name,
    # net_3_name,
    net_name,
    net_3_name,
    net_4_name,
    ]
error_types = ['t', 'm', 's', 'h']
all_err_colors = { # exactly these errors are plotted, be careful
    'm': 'cornflowerblue',
    's': 'olive',
    'h': 'peru',
    # 'SVShape_modal': 'tomato',
    net_name: 'purple',
    # net_two_name: 'purple',
    # net_3_name: 'purple',
    # net_2_name: 'purple',
    # net_2_name: 'green', 
    # net_3_name: 'blue', 
    # net_4_name: 'red',
    't': 'black', # must be last so that no division by zero
    }


all_err_styles = {
    'm': 'dotted', 
    's': 'dashed',
    'SVShape_modal': 'solid',
    net_name: 'solid', 
    # net_two_name: 'dashed',
    # net_2_name: 'dotted',
    # net_2_name: 'dashdot',
    # net_3_name: 'dashdot',
    # net_4_name: 'dashdot',
    'h': 'dashdot',
    't': 'solid', # must be last so that no division by zero
    
    }


params_for_cycles = [
    # ('n_max', [49], 'int'),
    # ('w_ensm_hybr', [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0], 'float')
    # ('w_ensm_hybr', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'float')
    # ('w_ensm_hybr', [0.1, 0.3, 0.5], 'float')
        ('e_size', [5, 10, 20, 40, 80], 'int'),
        # ('e_size', [5, 10, 20], 'int'),
        ('kappa_default', [1, 1.5, 2, 3, 4], 'float'),
        ('mu_NSL', [1, 2, 3, 5, 10], 'int') # [1, 2, 3, 5, 10]
    ]




if __name__ == '__main__':
    
    error_results = draw_results_1d(params_for_cycles,
                    source_folder,
                    error_types,
                    predictors_names,
                    all_err_colors,
                    all_err_styles,
                    rescale=True)
#%%

# for param_name, param_values, param_type in params_for_cycles:
#     for err_type in all_err_colors.keys():
#     # for err_type in [net_4_name]:
#         fig, ax = plt.subplots(figsize=(5,5))
#         for i, param_val in enumerate(param_values):
            
#             ax.hist(error_results[param_val][err_type],
#                     # color=all_err_colors[err_type],
#                     bins=25,
#                     alpha = 1 - i * 0.2,
#                     label=f'{beautiful_name(param_name)} = {param_val}')
#         plt.grid()
#         # ax.set_xlabel(f'{beautiful_name(param_name)} = {param_val}', 
#         #               fontsize=10)
#         plt.title(f'{beautiful_name(err_type)} err histogram')
#         plt.legend(loc='best')
#         plt.show()
    #%%
    # print(np.argmax(error_results[0.0]['m']))
    
    
    # for param in ['lambx', 'gammax', 'true_field', 'tr_nn', 'tr_sample']:
    #     predictors_names = [param]
    # for pred in ['t', net_name]:
    #     colors = {
    #         pred: all_err_colors[pred],
    #     }

    #     draw_results_1d(params_for_cycles,
    #                     source_folder,
    #                     error_types,
    #                     predictors_names,
    #                     colors,
    #                     rescale=False)
        
# source_folder = r"images\\Перебор\\"
# error_types = ['t', 'l', 's', 'h']

# if __name__ == '__main__':
#     draw_results_2d(halfwidth_multiplier_grid,
#                     e_size_grid,
#                     source_folder,
#                     error_types,
#                     predictors_names,
#                     all_err_colors,
#                     predictor_marker='l',
#                     truth_marker='t')



# error_types = ['R_s', 'R_m']

# error_results = dict()

# for err_type in error_types:
#     error_results[err_type] = dict()
#     for name in predictors_names:
#         error_results[err_type][name] = list()
    
# # print(error_results)
# # error_results['R_s']['net1'].append(1)
# # print(error_results)



# with open(results_dir + R_s_R_m) as file:
#     for line in file:
#         err = float(line.split()[-1])
#         for err_type in error_types:
#             for name in predictors_names:
#                 if name in line:
#                     break
#             if err_type in line:
#                 break
#         error_results[err_type][name].append(err)
#         # print(error_results)


                
# error_results_mean = dict()

# for err_type in error_types:
#     error_results_mean[err_type] = dict()
#     for name in predictors_names:
#         error_results_mean[err_type][name] = np.mean(
#             error_results[err_type][name]
#             )
# print('mean:')
# print(pd.DataFrame(error_results_mean))

# error_results_median = dict()

# for err_type in error_types:
#     error_results_median[err_type] = dict()
#     for name in predictors_names:
#         error_results_median[err_type][name] = np.median(
#             error_results[err_type][name]
#             )

# print('\nmedian:')
# print(pd.DataFrame(error_results_median))


# def draw_results_2d(param_1: Param_grid,
#                     param_2: Param_grid,
#                     source_folder: str,
#                     error_types: list,
#                     predictors_names: list,
#                     all_err_colors: dict,
#                     predictor_marker: str='l',
#                     truth_marker: str='t'):
#     results = []
    # colors = []
    # for predictor in predictors_names:
    #     colors.append(all_err_colors[predictor])
    #     error_results = dict()
    #     for err_type in error_types:
    #         error_results[err_type] = np.zeros((len(param_1.grid), 
    #                                              len(param_2.grid))).tolist()
        
    #     for i, val_1 in enumerate(param_1.grid):
    #         for j, val_2 in enumerate(param_2.grid):
    #             for err_type in error_types:
    #                 error_results[err_type][i][j] = list()
    #             for folder_name in os.listdir(source_folder):
    #                 source = source_folder + folder_name
    #                 if os.path.isdir(source) and \
    #                     f'{param_1.name} {val_1} {param_2.name} {val_2}' \
    #                         in folder_name:
    #                             with open(source + '\\' + R_s_R_m) as errors_file:
    #                                 for line in errors_file:
    #                                     if predictor not in line:
    #                                         continue
    #                                     # print(line.split())
    #                                     err = float(line.split()[-1])
    #                                     for err_type in error_types:
    #                                         if f' {err_type} ' in line:
    #                                             error_results[err_type][i][j].append(err)
    #     result = ((np.array(error_results[predictor_marker]) -\
    #               np.array(error_results[truth_marker])) / \
    #         np.array(error_results[truth_marker])).mean(axis=2)
    #     # print(result.shape)
    #     print(result)
    #     results.append(result)
    #     draw_surface(results, param_1.grid, param_2.grid,
    #                  title=predictor, 
    #                  xlabel=param_1.name, 
    #                  ylabel=param_2.name, 
    #                  color=colors,
    #                  label=predictors_names,
    #                  n_surfaces=2)
    #     # print(error_results)
                            
    



                