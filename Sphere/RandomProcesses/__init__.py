# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:15:32 2019

@author: Arseniy Sotskiy

#### Note that in python
FFT:
$$F[f](m) = \sum _{k = 0} ^{n-1} f  e^{\frac{-2 \pi i k m}{n}}$$
iFFT:
$$f[F](k) = \frac{1}{n} \sum _{m = -n/2 + 1} ^{n/2} F  e^{\frac{2 \pi i k m}{n}}$$

So we need to DIVIDE  the FORWARD python fft by n

and to MULTIPLY the INVERSE python fft by n.

"""

#from configs import *
#import configs


from RandomProcesses.Circle import RandStatioProcOnS1
# from RandomProcesses.ESM import proc_ESM, get_spectrum_from_ESM_data
from RandomProcesses.Sphere import RandStatioProcOnS2, get_spectrum_from_data
from RandomProcesses.DLSM_S2 import *

#import numpy as np
#from scipy.fftpack import ifft
#import matplotlib.pyplot as plt
#import random












