# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:18:17 2022

@author: Arseniy Sotskiy
"""

import os
import shutil

# source_folder = r"images\\Перебор\\"
source_folder = r"R from MD\\packages_for_torch\\"
destination_folder = r"images\\Перебор\\"

# fetch all files


print(os.listdir(source_folder))
# os.chdir(source_folder)
print(
    sorted(os.listdir(source_folder), 
           key=lambda f: os.path.getmtime(source_folder + f))
)

# for folder_name in os.listdir(source_folder):
#     print(folder_name)
    # current_folder = source_folder + folder_name + '\\'
    # if os.path.isdir(current_folder):
    #     if 'net_train' in folder_name:
    #         continue
    #     print(f'copying from {folder_name}...')
    #     for file_name in os.listdir(current_folder):
    #         print(f'    {file_name}')
    #         if file_name[-3:] != 'png':
    #             continue
    #         # print(f'    {file_name}')
    #         #     # construct full file path
    #         source = current_folder + file_name
    #         destination = destination_folder + file_name
    #         # move only files
    #         if os.path.isfile(source):
    #             shutil.copy2(source, destination)
    #             print('        Moved:', file_name)

# for file_name in os.listdir(source_folder):
    
    
'Rcpp_1.0.8.3.tar.gz', 'R6_2.5.1.tar.gz', 'withr_2.5.0.tar.gz', 'rlang_1.0.2.tar.gz', 'bit_4.0.4.tar.gz', 'bit64_4.0.5.tar.gz', 'magrittr_2.0.3.tar.gz', 'coro_1.0.2.tar.gz', 'ps_1.7.1.tar.gz', 'processx_3.6.1.tar.gz', 'callr_3.7.0.tar.gz', 'glue_1.6.2.tar.gz', 'cli_3.3.0.tar.gz', 'ellipsis_0.3.2.tar.gz', 'torch_0.8.0.tar.gz'
