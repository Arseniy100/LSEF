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
from configs import n_max, V_mult, lamb_mult, gamma_mult, angular_distance_grid_size
from Band import Band




#%%

spectrum = get_spectrum_from_data(V_mult, lamb_mult, gamma_mult, n_max)
print(len(spectrum))

rand_process = RandStatioProcOnS2(spectrum, n_max, angular_distance_grid_size)
realiz_field = rand_process.generate_one_realization(draw=True)
plt.show()

print(realiz_field.nlon, realiz_field.nlat)
nlon, nlat = realiz_field.nlon - 1, realiz_field.nlat - 1

i_lon_test, i_lat_test = int(nlon / 2), int(nlat / 2)  # test point




#%%

field_filtered = band_pass_filter(realiz_field, Band(10, 25), draw=True)

bands = make_bands(5, 3, n_max)

fields_filtered = []
realiz_field.plot(show=False)
for band in bands:
    field_filtered = band_pass_filter(realiz_field, band, draw=False)
    field_filtered.plot(show=False)
    fields_filtered.append(field_filtered)



#%%
# check:

field_sum_of_filtered = np.sum([field for field in fields_filtered])
diff = realiz_field - field_sum_of_filtered
diff.plot(show=False)
print('max difference: ', diff.max())



