"""
Created on March 26 2020
Last modified: Mar 2020

@author: Arseniy Sotskiy
"""

import matplotlib.pyplot as plt
import numpy as np
import pyshtools

# pyshtools.utils.figstyle(rel_width=0.7)

from RandomProcesses import RandStatioProcOnS2, get_spectrum_from_data
from functions.tools import draw_1D, draw_2D
from functions.process_functions import make_bands, band_pass_filter
from configs import n_max, gamma, lamb, V, angular_distance_grid_size
from Band import Band




#%% WHITE NOISE

NSW = 2
seed = None

spectrum_white_noise = np.zeros((n_max + 1))
spectrum_white_noise[0:NSW] = np.ones((NSW))

white_noise = RandStatioProcOnS2(spectrum_white_noise, n_max, 
                                 angular_distance_grid_size)
realiz_grid = white_noise.generate_one_realization(draw=True, seed = seed)
plt.title('random realization of white noise, NSW = {}'.format(NSW))
plt.show()