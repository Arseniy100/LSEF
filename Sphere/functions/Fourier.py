# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:14:01 2019

@author: Arseniy Sotskiy
"""

##### Note that in python
#FFT:
#$$F[f](m) = \sum _{k = 0} ^{n-1} f  e^{\frac{-2 \pi i k m}{n}}$$
#iFFT:
#$$f[F](k) = \frac{1}{n} \sum _{m = -n/2 + 1} ^{n/2} F  e^{\frac{2 \pi i k m}{n}}$$
#So we need to DIVIDE  the FORWARD python fft by n
#and to MULTIPLY the INVERSE python fft by n.


from numpy.fft import ifft, fft


def FFT(array, n=None, axis=-1, norm=None):
    if n is None:
        n = len(array)
    return fft(array, n, axis, norm)/n


def iFFT(array, n=None, axis=-1, norm=None):
    if n is None:
        n = len(array)
    return ifft(array, n=None, axis=-1, norm=None)*n
