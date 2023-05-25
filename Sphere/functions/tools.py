# -*- coding: utf-8 -*-
"""
Last modified on Mar 2022
@author: Arseniy Sotskiy

Simple functions for plotting,
doing different operations with matrices, etc.

List of functions:
    
    _time()
    
    mkdir(folder_descr: str, is_Linux: bool=False) -> str
    
    draw_3D(matrix: np.array,
            title: str='', xlabel: str='', ylabel: str='',
            save_path:str=None,
            rotate: bool=False)
    
    def draw_surface(
        matrix: np.array, x_grid: np.array=None, y_grid: np.array=None,
        title: str='', xlabel: str='', ylabel: str='', color='blue',
        wireframe: bool=True, label: str='',
        save_path:str=None,
        n_surfaces = 1
        )

    draw_2D(matrix: np.array,
            title: str='', xlabel: str='', ylabel: str='',
            save_path:str=None, white_zero=False)

    draw_1D(array: np.array, length: int=None,
            title:str='', xlabel: str='', ylabel: str='',
            multiple: bool=False,
            save_path:str=None)
    
    convert_modal_spectrum_to_variance(modal_spectrum: np.array,
                                       n_max: int) -> np.array
    
    def convert_variance_spectrum_to_modal(variance_spectrum: np.array,
                                       n_max: int) -> np.array
    
    def svd_pseudo_inversion(Omega: np.ndarray, V_e: np.ndarray,
                         U=None, d=None, V=None, precomputed:bool=False,
                         nSV_discard:int=0, *args, **kwargs) -> np.ndarray
    
    convert_spectrum2fft(spectrum: np.array) -> np.array
    
    convert_aligned_matrix_to_diag(aligned_matrix: np.array) -> np.array
    
    convert_diag_matrix_to_aligned(diag_matrix: np.array) -> np.array
    
    make_crm_from_cvm(cov_matrix: np.array) -> np.array
    
    sigmoid(z: float, b: float = 1) -> float
    
    filter_fft(array: np.array, n_2: int, deg: int=8) -> np.array
    
    integrate_gridded_function_on_S2(grid: pyshtools.SHGrid,
                                     function = identical) -> float
    
    draw_2D_1D_rows_of_matrix(B: np.array, name: str,
                              grid: LatLonSphericalGrid,
                              points=None,
                              comment: str = '') -> None
    
    draw_2D_1D_rows_of_matrix(B: np.array, name: str,
                              grid: LatLonSphericalGrid,
                              points: list=None,
                              comment: str = '') -> None
    
    find_sample_covariance_matrix(ensemble: np.array) -> np.array
    
    apply_inv_matrix_to_vector(matrix: np.array, 
                               vector: np.array) -> np.array
    
    make_matrix_sparse(mx: np.array, threshold_coeff: float=0.001,
                       return_n_zeros: bool=False) -> np.array
    
    show_best_R_s_with_params(param_grid: tp.List[dict]=None, 
                              R_s_grid: tp.List[float]=None, 
                              R_m_grid: tp.List[float]=None,
                              data: str=None, 
                              path: str=None, 
                              param_grid_dict: dict=dict())
"""



# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np
import pyshtools
from pyshtools import SHGrid

import typing as tp

from Grids import LatLonSphericalGrid, make_2_dim_field
# from Band import Band
#import sympy

from time import process_time, strftime
import os


#%%
def _time():
    """Returns current time in format YYYY_MM_DD_HH_MM_SS"""
    return (strftime('%Y_%m_%d_%H_%M_%S'))

print(_time())


#%%
def mkdir(folder_descr: str, is_Linux: bool=False) -> str:
    '''
    Makes directory (folder)
    dir name is current time + folder_descr
    returns path to the dir

    Parameters
    ----------
    folder_descr : str
        dir name (without current time)
    is_Linux : bool, optional
        if True, then replaces "\" with "/" for Linux
        The default is False.

    Returns
    -------
    path_to_save : str
        path to the dir ("images\dir\").

    '''
    path_to_save = 'images\\' + _time() + \
        folder_descr + r'\\'
    if is_Linux:
        path_to_save = path_to_save.replace('\\', r'/')
    path_to_save = path_to_save.replace(':', '_')
    os.mkdir(path_to_save)
    # os.rmdir(r'images\2021_12_15_15_19_07 20 training spheres, w_smoo 0.0001, ')
    print(f'made directory {path_to_save}') 
    return path_to_save

#%%
# COLORMAP = plt.cm.viridis
# COLORMAP = plt.cm.plasma
# COLORMAP = plt.cm.inferno

# COLORMAP = plt.cm.spring
# COLORMAP = plt.cm.autumn
# COLORMAP = plt.cm.spring
COLORMAP = plt.cm.seismic

def draw_3D(matrix: np.array,
            title: str='', xlabel: str='', ylabel: str='',
            save_path:str=None,
            rotate: bool=False):
    '''
    Plots 2d matrix - grid - on a sphere

    Parameters
    ----------
    matrix : np.array
        2d matrix to plot
    title : str, optional
        title of the plot. The default is ''.
    xlabel : str, optional
        label of the x-axis (horizontal). The default is ''.
    ylabel : str, optional
        label of the y-axis (vertical). The default is ''.
    save_path : str, optional
        path to save the picture. The default is None (and then no saving).
    rotate : bool, optional
        if True, then the picture will rotate. The default is False.

    Returns
    -------
    None.
    '''

    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1, projection='3d')

    u = np.linspace( 0, 2 * np.pi, matrix.shape[1])
    v = np.linspace( 0, np.pi, matrix.shape[0])

    # create the sphere surface
    XX = np.outer( np.cos( u ), np.sin( v ) )
    YY = np.outer( np.sin( u ), np.sin( v ) )
    ZZ = np.outer( np.ones( np.size( u ) ), np.cos( v ) )



    # ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
    norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1,
                    facecolors=COLORMAP( norm(matrix.T) ),
                    shade=False)
    m = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    fig.colorbar(m, shrink=0.5)
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
    if save_path:
        fig.savefig(save_path)
    if rotate:
        for angle in range(0, 360, 2):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.000001)
    else:
        plt.show()


if __name__ == '__main__':
    mx = np.arange(800).reshape((20,40))
    draw_3D(mx, title='0-99', xlabel='x', ylabel='y',
            save_path='test_3.png', rotate=False
            )


#%%

def draw_surface(
        matrix: np.array, x_grid: np.array=None, y_grid: np.array=None,
        title: str='', xlabel: str='', ylabel: str='', color='blue',
        wireframe: bool=True, label: str='',
        save_path:str=None,
        n_surfaces = 1
        ):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # Make data.
    if x_grid is None:
        x_grid = np.arange(matrix.shape[0])
    if y_grid is None:
        y_grid = np.arange(matrix.shape[1])
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    print(x_grid)
    print(x_grid.shape)
    print(y_grid)
    print(y_grid.shape)
    
    
    
    
    # Plot the surface.
    if not wireframe:
        norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
        surf = ax.plot_surface(
            x_grid, y_grid, 
            matrix.T, 
            facecolors=COLORMAP( norm(matrix.T) ),
            # cmap=cm.coolwarm,
            linewidth=0, antialiased=False
            )
        m = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
        fig.colorbar(m, shrink=0.7)
    # surf_2 = ax.plot_surface(
    #     x_grid, y_grid, 
    #     matrix*2, 
    #     # facecolors=COLORMAP( norm(matrix.T) ),
    #     cmap=cm.coolwarm,
    #     linewidth=0, antialiased=False
    #     )
    if wireframe:
        if n_surfaces == 1:
            matrix = [matrix]
            color = [color]
            label=[label]
        for i, mx in enumerate(matrix):
            surf = ax.plot_wireframe(
                x_grid, y_grid, 
                matrix[i].T, 
                color=color[i],
                label=label[i]
                # cstride=0
                # facecolors=COLORMAP( norm(matrix.T) ),
                # cmap=cm.coolwarm,
                # linewidth=0, antialiased=False
                )
    # surf_2 = ax.plot_wireframe(
    #     x_grid, y_grid, 
    #     matrix**2, 
    #     color='red',
    #     # cstride=0
    #     # facecolors=COLORMAP( norm(matrix.T) ),
    #     # cmap=cm.coolwarm,
    #     # linewidth=0, antialiased=False
    #     )
    
    
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set(title=title,
            xlabel=xlabel,
            ylabel=ylabel)
    ax.legend(loc='best')
    if save_path:
        fig.savefig(save_path)
        
    
    
    plt.show()

if __name__ == '__main__':
    mx = np.arange(200).reshape((10,20))
    draw_surface(mx, title='0-99', xlabel='x', ylabel='y',
            save_path='test_3dsurf.png',
            )

#%%

def draw_2D(matrix: np.array,
            title: str='', xlabel: str='', ylabel: str='',
            save_path:str=None, white_zero=False,
            cmap=None, save_gray: str=None):
    '''
    Plots 2d matrix

    Parameters
    ----------
    matrix : np.array
        2d matrix to plot
    title : str, optional
        title of the plot. The default is ''.
    xlabel : str, optional
        label of the x-axis (horizontal). The default is ''.
    ylabel : str, optional
        label of the y-axis (vertical). The default is ''.
    save_path : str, optional
        path to save the picture. The default is None (and then no saving).
    white_zero : bool, optional
        if True, then 0 is white on the colorbar.  
    cmap : optional
        Colormap. The default is None (and then it is as default)

    Returns
    -------
    None.

    '''
    if cmap is None:
        cmap = COLORMAP
    fig, ax = plt.subplots()
    norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
    if white_zero == True:
        norm = colors.Normalize(vmin=-np.abs(matrix).max(), 
                                vmax=np.abs(matrix).max())
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.imshow(np.real(matrix), cmap=cmap, norm=norm)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(m, cax=cax)
    # fig.colorbar(m)
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
    if save_path:
        fig.savefig(save_path)    
    plt.show()
    
    if save_gray is not None:
        img = mpimg.imread(save_path)
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)

        Avg = (R+G+B)
        grayImage = img.copy()

        for i in range(3):
            grayImage[:,:,i] = Avg
        fig, ax = plt.subplots()
        plt.imshow(grayImage)
        plt.axis('off')
        fig.savefig(save_gray) 
        plt.show()

if __name__ == '__main__':
    mx = np.arange(-200, 800).reshape((20,50))
    draw_2D(mx, title='0-99', xlabel='x', ylabel='y',
            save_path='test_2.png', white_zero=True,
            save_gray='test_2_.png'
            )
    for cmap in [plt.cm.inferno, plt.cm.BuPu,]:
        draw_2D(mx, title='0-99', xlabel='x', ylabel='y', cmap=cmap,
                save_path='test_2.png',
                save_gray='test_2_.png')

#%%


def draw_1D(array: np.array, length: int=None,
            title:str='', xlabel: str='', ylabel: str='',
            multiple: bool=False,
            save_path:str=None):
    '''
    Plots function given as an array of values
    [f[0], f[1], ..., f[length - 1]]
    Parameters
    ----------
    array : np.array
        array of function values
    length : int, optional
        lenght of the array. The default is None.
    title : str, optional
        title of the plot. The default is ''.
    xlabel : str, optional
        label of the x-axis (horizontal). The default is ''.
    ylabel : str, optional
        label of the y-axis (vertical). The default is ''.
    multiple : bool, optional
        if array is a list or tuple of different arrays to plot
    save_path : str, optional
        path to save the picture. The default is None (and then no saving).

    Returns
    -------
    None.

    '''



    fig, ax = plt.subplots()
    if multiple:
        for arr in array:
            if length is None:
                ax.plot(arr)
            else:
                ax.plot(arr[:length])
    else:
        if length is None:
            length = len(array)
        ax.plot(np.arange(length), array[:length])
    ax.grid(True)
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
    # ax.legend(loc = 'best')
    if save_path:
        fig.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    x = np.linspace(0, 3, 100)
    y = x ** 2
    y_1 = x ** 3
    draw_1D((y, y_1), length=50, multiple=True,
            title='square and cube', xlabel='x', ylabel='y',
            save_path='test_1.png'
            )


#%%

def convert_modal_spectrum_to_variance(modal_spectrum: np.array,
                                       n_max: int) -> np.array:
    '''


    Parameters
    ----------
    modal_spectrum : np.array
    n_max : int

    Returns
    -------
    variance_spectrum : np.array

    '''
    if len(modal_spectrum.shape) == 3:
        variance_spectrum = 1 / (4 * np.pi) * np.array(
            [(2*n+1) * modal_spectrum[n,:,:] for n in range(n_max + 1)]
            )
    elif len(modal_spectrum.shape) == 1:
        variance_spectrum = 1 / (4 * np.pi) * np.array(
            [(2*n+1) * modal_spectrum[n] for n in range(n_max + 1)]
            )
    return variance_spectrum


def convert_variance_spectrum_to_modal(variance_spectrum: np.array,
                                       n_max: int) -> np.array:
    '''


    Parameters
    ----------
    variance_spectrum : np.array
    n_max : int

    Returns
    -------
    modal_spectrum : np.array
    '''
    if len(variance_spectrum.shape) == 3:
        modal_spectrum = (4 * np.pi) * np.array(
            [1 / (2*n+1) * variance_spectrum[n,:,:] for n in range(n_max + 1)]
            )
    elif len(variance_spectrum.shape) == 1:
        modal_spectrum = (4 * np.pi) * np.array(
            [1 / (2*n+1) * variance_spectrum[n] for n in range(n_max + 1)]
            )
    return modal_spectrum


#%%


def svd_pseudo_inversion(Omega: np.ndarray, V_e: np.ndarray,
                         U=None, d=None, V=None, precomputed:bool=False,
                         nSV_discard:int=0, *args, **kwargs) -> np.ndarray:
    '''
    Solves the equation
    V_e = Omega * b
    using pseudo inversion:
        b = Omega^+ V_e

        Omega = U*D*V^T

    Parameters
    ----------
    V_e : np.ndarray
        1d vector.
    Omega : np.ndarray
        2d matrix.
    U, d, V: svd of Omega (if precomputed == True)
    precomputed : bool
        if True, then U,d,V are not computed from Omega
    nSV_discard : int
        number of V columns which are ommited

    Returns
    -------
    b : np.ndarray
        1d vector.

    '''
    if not precomputed:
        U, d, V_T = np.linalg.svd(Omega, full_matrices=False)
        V = V_T.T
    # print('d', d)
    b_ = np.matmul(U.T, V_e) / d
    # print('b_.shape', b_.shape)
    if nSV_discard == 0:
        b = np.matmul(V[:,:], b_[:])
    else:
        b = np.matmul(V[:,:-nSV_discard], b_[:-nSV_discard])
    return b


if __name__ == '__main__':
    shape = (5, 10)
    A = np.random.randn(shape[0], shape[1])
    print('A', A)
    x = np.random.randn(shape[1])
    print('x', x)
    b = np.matmul(A, x) + 0*np.random.normal(0, 0.01, shape[0])
    print('b', b)
    x_ = svd_pseudo_inversion(A, b, nSV_discard=0)
    print('x-x_ =', x-x_)





#%%

def convert_spectrum2fft(spectrum: np.array) -> np.array:
    '''
    Makes from the array spectrum = [a_0, a_1, ..., a_n] another array:
    spectrum_1 = [a_0, a_1, ..., a_n, a_n*, ..., a_2*, a_1*]
    such that spectrum_1[-i] = spectrum[i]*
    (* is complex conjugation)

    Parameters
    ----------
    spectrum : 1d np.array

    Returns
    -------
    1d np.array
        enlarged array
    '''

    return np.array(
        spectrum.tolist() + np.conj(spectrum).tolist()[-2:0:-1]
        )


def convert_aligned_matrix_to_diag(aligned_matrix: np.array) -> np.array:
    '''
    Shifts all rows of the aligned_matrix so that
    the first column becomes the diagonal.
    for example:
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    becomes
    array([[0, 1, 2],
           [5, 3, 4],
           [7, 8, 6]])

    Parameters
    ----------
    aligned_matrix : 2d np.array

    Returns
    -------
    diag_matrix : 2d np.array
    '''

    diag_matrix = aligned_matrix.copy()
    for i in range(len(diag_matrix)):
        diag_matrix[i,:] = np.roll(aligned_matrix[i,:], i)
    return diag_matrix


def convert_diag_matrix_to_aligned(diag_matrix: np.array) -> np.array:
    '''
    Shifts all rows of the aligned_matrix so that
    the diagonal becomes the first column.
    For example:
    array([[0, 1, 2],
           [5, 3, 4],
           [7, 8, 6]])
    becomes
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    Parameters
    ----------
    diag_matrix : 2d np.array

    Returns
    -------
    aligned_matrix : 2d np.array
    '''
    aligned_matrix = diag_matrix.copy()
    for i in range(len(diag_matrix)):
        aligned_matrix[i,:] = np.roll(aligned_matrix[i,:], -i)
    return aligned_matrix


def make_crm_from_cvm(cov_matrix: np.array) -> np.array:
    '''
    Makes correlation matrix from the covariance matrix.

    Parameters
    ----------
    cov_matrix : 2d np.array

    Returns
    -------
    corr_matrix : 2d np.array
    '''

    corr_matrix = cov_matrix.copy()
    for i in range(len(cov_matrix)):
        for j in range(len(cov_matrix)):
            corr_matrix[i,j] = \
            corr_matrix[i,j]/np.sqrt(cov_matrix[i,i]*cov_matrix[j,j])
    return corr_matrix


#%%

def sigmoid(z: float, b: float = 1) -> float:
    '''
    Logistic function.
    sigmoid(z) : (-inf, inf) -> (0, 1 + exp(b))

    Parameters
    ----------
    z : float
    b : float, optional
        The default is 1.

    Returns
    -------
    float
    '''
    return (1 + np.exp(b))/(1 + np.exp(b-z))

sigmoid(np.array([-10 ,-5, -1,0,1, 5, 10]), -10)

#%%

def filter_fft(array: np.array, n_2: int, deg: int=8) -> np.array:
    '''
    Reduces sampling noise
    (by decreasing the high frequency)
    applies ifft, decreases "tail" of the spectrum
    and applies fft

    Parameters
    ----------
    array : 1d np.array
        array of function values to be filtered
    n_2 : int
        parameter of the filter (characteristic length of the filter,
        such that the spectrum is divided by two at this point).
    deg : int, optional
        parameter of the filter (intensity). The default is 8.

    Returns
    -------
    1d np.array
    '''

    fft_array = np.fft.ifft(array)
    for n in range(len(fft_array)):
        fft_array[n] = fft_array[n] / (1 + (n/n_2)**deg)
    return np.fft.fft(fft_array)


#%%

def identical(x):
    return x


def integrate_gridded_function_on_S2(grid: pyshtools.SHGrid,
                                     function = identical) -> float:
    '''
    Computes integral of the fucntion of the grid on the sphere -
    L2-norm of this grid

    grid should be

    \integral_{S^2} function(grid(\theta, \phi)) \cos(\theta) d \theta d \phi

    Parameters
    ----------
    grid : pyshtools.SHGrid
    function: function to be integrated. The default is identical

    Returns
    -------
    norm : float
    '''
    theta = np.array(
        [[(2*i/(grid.nlat-1) - 1) * np.pi / 2 for i in range(grid.nlat)]]
        )  # from - pi to pi
    # draw_1D(theta[0])

    d_theta = np.pi / (grid.nlat - 1)
    d_phi = 2 * np.pi / (grid.nlon - 1)

    weights = np.matmul((np.cos(theta)).T, np.ones((1, grid.nlon - 1)))
    weights[0,:] = np.ones((grid.nlon - 1)) * np.sin(d_theta / 4)\
    * np.sin(d_theta / 4) / np.sin(d_theta / 2)
    weights[-1,:] = np.ones((grid.nlon - 1)) * np.sin(d_theta / 4)\
    * np.sin(d_theta / 4) / np.sin(d_theta / 2)
    # print(np.sin(d_theta / 4)\
    # * np.sin(d_theta / 4) / np.sin(d_theta / 2) )


    grid_func = function(grid.data[:,:-1])
    # print(np.multiply(grid_func, weights))
    # print(np.multiply(grid_func, weights) * d_phi *
    #                     2 * np.sin(d_theta / 2))
    integral = np.sum(np.multiply(grid_func, weights) * d_phi *
                        2 * np.sin(d_theta / 2))
                        # d_theta)
    # poles must be added !!!!!
    # ones = pyshtools.SHGrid.from_array(np.ones(grid.data.shape))

    # integral_1 = np.sum(np.multiply(ones.data[:,:-1], weights) * d_phi * d_theta)
    # print('integral_1: ', integral_1 / 4 / np.pi)
    # print(norm, norm_1)
    return integral # / integral_1 * 4 * np.pi

if __name__ == '__main__':
    # nlon = 241
    # nlat = 121
    nlon = 5
    nlat = 3
    # nlon = 9
    # nlat = 5
    # nlon = 25
    # nlat = 13
    ones = pyshtools.SHGrid.from_array(np.ones((nlat, nlon)))
    # integrate_gridded_function_on_S2(ones)
    print(integrate_gridded_function_on_S2(ones) / 4 / np.pi)



# L2_norm(ensemble_of_fields[0])


#%%

def draw_2D_1D_rows_of_matrix(B: np.array, name: str,
                              grid: LatLonSphericalGrid,
                              points: list=None,
                              comment: str = '') -> None:
    '''
    Takes one row of matrix B and transforms it to the 2D field.
    Draws 2D field, one row and one column of this field.

    Parameters
    ----------
    B : np.array
        2D matrix (B or W).
    name : str
        name of matrix.
    grid: LatLonSphericalGrid
        our coarsened grid
    points: list
        list of points to draw
    comment: str
        comment to add to the title of the picture

    Returns
    -------
    None
    '''

    if points is None:
        points = [int(lat)*grid.nlon+int(grid.nlon/2)
                  for lat in np.linspace(0, grid.nlat, 5)[:-1]]
    for point in points :
    # [grid.nlat/2, grid.nlat/3, grid.nlat/5, grid.nlat/10,]:
        my_lat = int(grid.colats[point]/np.pi*grid.nlat)
        my_lon = int(grid.lons[point]/2/np.pi*grid.nlon)

        titles = ['%s - %.0f lon = %.2f, lat=%.2f'\
                %(name, point, grid.lons[point], grid.lats[point]),
                '%s - %.0f, lon = %.2f, lat=%.2f, row %.0f'\
                %(name, point, grid.lons[point], grid.lats[point], my_lat),
                '%s - %.0f, lon = %.2f, lat=%.2f, column %.0f'\
                %(name, point, grid.lons[point], grid.lats[point], my_lon)]
        titles = [title + comment for title in titles]

        print(point, grid.lons[point], grid.lats[point])
        one_row_as_field = make_2_dim_field(field=B[int(point),:],
                                            nlat=grid.nlat, nlon=grid.nlon+1)
        draw_2D(one_row_as_field,
                title=titles[0],
                save_path=f'images/explosion/{titles[0]}.png')



        draw_1D(one_row_as_field[my_lat,:],
                title=titles[1],
                save_path=f'images/explosion/{titles[1]}.png')
        draw_1D(one_row_as_field[:,my_lon],
                title=titles[2],
                save_path=f'images/explosion/{titles[2]}.png')
        print('max is at point %.0f'%(np.argmax(one_row_as_field[:,my_lon])))



#%%

def find_sample_covariance_matrix(ensemble: np.array) -> np.array:
    '''Finds estimation of the covariance matrix

    Parameters
    ----------
    ensemble : np.array
        data; shape = (n_dimensions, sample_size)

    Returns
    -------
    S : np.array
        cov matrix of the ensemble, shape = (n_dimensions, n_dimensions)

    '''
    n_e = ensemble.shape[1]
    E = (ensemble - ensemble.mean(axis=1).reshape((-1, 1)))
    # print(E.shape)
    S = np.matmul(E, E.T) / (n_e - 1)
    return S


if __name__ == '__main__':
    B = np.array([
        [3,2,1],
        [2,3,2],
        [1,2,3]
        ])
    ensm = np.random.multivariate_normal([0,0,0], B, 3000).T
    B_est = find_sample_covariance_matrix(ensm)
    print(B_est)





#%%

def apply_inv_matrix_to_vector(matrix: np.array, 
                               vector: np.array) -> np.array:
    '''computes matrix^(-1) * vector
    
    Parameters
    ----------
    matrix : np.array
    vector : np.array

    Returns
    -------
    np.array
        matrix^(-1) * vector

    '''
    return np.linalg.solve(matrix, vector)


if __name__ == '__main__':
    for _ in range(10):
        mx = np.random.normal(size=(5, 5))
        vector = np.arange(5)

        # print(mx)
        # print(vector)
        print(np.abs(np.matmul(np.linalg.inv(mx), vector) - \
              apply_inv_matrix_to_vector(mx, vector)).max())


#%%

def make_matrix_sparse(mx: np.array, threshold_coeff: float=0.001,
                       return_n_zeros: bool=False) -> np.array:
    '''Makes matrix more sparse:
        all values v such that 
        |v| <= |matrix_ij|_max * threshold_coeff 
        become zero (|matrix_ij|_max - value with the max abs of the matrix)

    Parameters
    ----------
    mx : np.array
        matrix which will bw sparsed.
    threshold_coeff : float, optional
        if 1, then all matrix will become 0. The default is 0.001.
    return_n_zeros : bool, optional
        if True, then returns also threshold_descr: 
            string with number of new zero values. 
        The default is False.

    Returns
    -------
    mx_sparse: np.array
        sparsed matrix
    if return_n_zeros is True, then returns 
    (mx_sparse: np.array, threshold_descr: str)

    '''
    mx_sparse = mx.copy()
    to_zero_mask = np.abs(mx) <= np.abs(mx).max() * threshold_coeff
    mx_sparse[to_zero_mask] = 0
    if return_n_zeros:
        n_zeros = to_zero_mask.sum()
        zeros_frac = np.round(n_zeros / mx.size, 2)
        threshold_descr = f'threshold_coeff = {threshold_coeff}; ' + \
            f'{n_zeros} zeros ({int(zeros_frac * 100)}%)'
        return mx_sparse, threshold_descr
    return mx_sparse

if __name__ == '__main__':
    mx = np.arange(-10, 10).reshape(4,5)
    print(make_matrix_sparse(mx, 0.5, True))
    print(mx)
    print(make_matrix_sparse(mx, 0.1, True))
    print(make_matrix_sparse(mx, 1, True))






#%%

import json
import pandas as pd

data = '''
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 1.0681818181818181, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.12219098578100986
R_m = 0.2337611816668135
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 1.0681818181818181, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.12219098578100844
R_m = 0.23376118166681079
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 1.0681818181818181, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.14838462938086053
R_m = 0.2838717282093725
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 1.0681818181818181, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.19292179060896197
R_m = 0.36907489905067364
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 2.1363636363636362, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1038844153016095
R_m = 0.19873924023489073
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 2.1363636363636362, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.12722797300333785
R_m = 0.24339734326749207
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 2.1363636363636362, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.166640286520357
R_m = 0.318796268327913
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 3.2045454545454546, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.09799898554488279
R_m = 0.18747993983923686
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 3.2045454545454546, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1136660989241507
R_m = 0.21745238758926386
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 2.35, 'nc2': 3.2045454545454546, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.14837271484853304
R_m = 0.28384893475093464
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 1.0681818181818181, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.13799118800392715
R_m = 0.26398823907695
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 1.0681818181818181, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1657375287443203
R_m = 0.3170692201079776
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 1.0681818181818181, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.20537674496566427
R_m = 0.39290222828792937
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 2.1363636363636362, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1143067938072969
R_m = 0.21867808842157038
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 2.1363636363636362, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1377642602308765
R_m = 0.26355410799893086
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 2.1363636363636362, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.17751943311726492
R_m = 0.33960895060365354
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 3.2045454545454546, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.10568641633689352
R_m = 0.20218661312152703
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 3.2045454545454546, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.12434111229257515
R_m = 0.23787454658374305
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.1333333333333333, 'nc2': 3.2045454545454546, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1592506253791929
R_m = 0.3046592523325456
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 1.0681818181818181, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.16395440290543625
R_m = 0.31365795698988486
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 1.0681818181818181, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.18391192852923657
R_m = 0.35183830837297614
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 1.0681818181818181, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.2168571567966948
R_m = 0.4148651793066979
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 2.1363636363636362, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.1241421912025989
R_m = 0.23749399454257503
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 2.1363636363636362, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.14672363900661642
R_m = 0.28069412005653893
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 2.1363636363636362, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.18787495299688506
R_m = 0.359419892862296
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 3.2045454545454546, 'halfwidth_max': 17.625, 'q_tranfu': 2, 'rectang': False}
R_s = 0.11439745348340997
R_m = 0.21885152767228322
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 3.2045454545454546, 'halfwidth_max': 23.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.13397809787664083
R_m = 0.2563108749547623
params = {'n_max': 47, 'nband': 6, 'halfwidth_min': 3.9166666666666665, 'nc2': 3.2045454545454546, 'halfwidth_max': 29.375, 'q_tranfu': 2, 'rectang': False}
R_s = 0.17014413389694794
R_m = 0.32549941011776634

'''



data_20 = '''
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 22.125, 'q_tranfu': 2, 'rectang': False}
R_s = 0.3539572707628803
R_m = 0.3068152551194878
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 22.125, 'q_tranfu': 3, 'rectang': False}
R_s = 0.37168011647694116
R_m = 0.3221776727849965
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 29.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.40289585187088434
R_m = 0.3492359213639645
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 29.5, 'q_tranfu': 3, 'rectang': False}
R_s = 0.3822550610785472
R_m = 0.3313441869205113
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 36.875, 'q_tranfu': 2, 'rectang': False}
R_s = 0.45871559168746556
R_m = 0.39762127498480004
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 1.3409090909090908, 'halfwidth_max': 36.875, 'q_tranfu': 3, 'rectang': False}
R_s = 0.410185565402729
R_m = 0.3555547499395165
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 22.125, 'q_tranfu': 2, 'rectang': False}
R_s = 0.31475966590593074
R_m = 0.272838207244931
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 22.125, 'q_tranfu': 3, 'rectang': False}
R_s = 0.2976501912353524
R_m = 0.2580074684252429
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 29.5, 'q_tranfu': 2, 'rectang': False}
R_s = 0.3518898560747905
R_m = 0.3050231903213925
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 29.5, 'q_tranfu': 3, 'rectang': False}
R_s = 0.3207113787878786
R_m = 0.2779972376056778
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 36.875, 'q_tranfu': 2, 'rectang': False}
R_s = 0.4095123969222652
R_m = 0.3549712378149408
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 2.6818181818181817, 'halfwidth_max': 36.875, 'q_tranfu': 3, 'rectang': False}
R_s = 0.35487126893436716
R_m = 0.3076075218853579
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 4.0227272727272725, 'halfwidth_max': 22.125, 'q_tranfu': 2, 'rectang': False}
R_s = 0.3016904325483655
R_m = 0.26150960772732496
params = {'n_max': 59, 'nband': 6, 'halfwidth_min': 2.95, 'nc2': 4.0227272727272725, 'halfwidth_max': 22.125, 'q_tranfu': 3, 'rectang': False}
R_s = 0.29476766150612127
R_m = 0.2555088501814153
'''

def show_best_R_s_with_params(param_grid: tp.List[dict]=None, 
                              R_s_grid: tp.List[float]=None, 
                              R_m_grid: tp.List[float]=None,
                              data: str=None, 
                              path: str=None, 
                              param_grid_dict: dict=dict()):
    '''
    Prints max and min R_s and R_m for each param value from param_grid.
    
    For example: param halfwidth_max
    
    halfwidth_max np.array([1.5, 2, 2.5]) * n_max / 4
    halfwidth_max
    22.125    0.294768
    29.500    0.320711
    36.875    0.354871
    Name: R_s, dtype: float64
    halfwidth_max
    22.125    0.255509
    29.500    0.277997
    36.875    0.307608
    Name: R_m, dtype: float64

    Parameters
    ----------
    param_grid : tp.List[dict], optional
        each element of list is a dict of parameters. The default is None.
    R_s_grid : tp.List[float], optional
        array of R_s values for different params from param_grid. 
        The default is None.
    R_m_grid : tp.List[float], optional
        array of R_m values for different params from param_grid. 
        The default is None.
    data : str, optional
        String which can be read from the file and then parsed. 
        If not None, then it is parsed to get param_grid, R_s_grid, R_m_grid.
        The default is None.
    path : str, optional
        path to save the picture. The default is None (and then no saving).
    param_grid_dict : dict, optional
        Dict of descriptions of different values of parameters. For example:
        {'halfwidth_min': 'np.array([1.5, 2, 2.5]) * n_max/30', 
         'nc2': 'np.array([0.5, 1, 1.5]) * n_max / 22', 
         'halfwidth_max': 'np.array([1.5, 2, 2.5]) * n_max / 4', 
         'q_tranfu': '[2, 3]'}
        The default is dict().

    Returns
    -------
    None.

    '''
    if data is not None:
        data = data.replace("'", '"').replace('True', '1').replace('False', '0')
        data_list = data.split('\n')
        param_grid = [json.loads(s[len('params = '):]) \
                      for s in data_list if 'params' in s]
        R_s_grid = [float(s.split()[-1]) for s in data_list if 'R_s' in s]
        R_m_grid = [float(s.split()[-1]) for s in data_list if 'R_m' in s]

    else:
        error_message = 'param_grid and R_S_grid needed if data is None'
        assert param_grid is None, error_message
        assert R_s_grid is None, error_message
    R_s_grid = np.array(R_s_grid)

    df = pd.concat([pd.DataFrame(np.array(list(a.values())).reshape(1,-1),
                                 columns=a.keys())\
                                  for a in param_grid],
          ignore_index=True)
    
    df['R_s'] = R_s_grid
    if R_m_grid is not None:
        R_m_grid = np.array(R_m_grid)
        df['R_m'] = R_m_grid
    print(R_s_grid)

    if path is not None:
            with open(f'{path}best_R_s.txt', 'a') as file:
                file.write('='*10 + '\n' + '='*10 + '\n'*2)

    for feature in param_grid[0].keys():
        if feature in param_grid_dict.keys():
            print(feature, param_grid_dict[feature])
        print(df.groupby(by=feature)['R_s'].min())
        print(df.groupby(by=feature)['R_m'].min())
        if path is not None:
            with open(f'{path}best_R_s.txt', 'a') as file:
                if feature in param_grid_dict.keys():
                    file.write(feature + ': ' + str(param_grid_dict[feature]) \
                               + '\n')
                file.write(str(df.groupby(by=feature)['R_s'].min()) + '\n')
                file.write(str(df.groupby(by=feature)['R_m'].min()) + '\n')
        # df['R_s'].hist(by=df[feature], bins=4)
        # plt.legend(loc='best')
        # plt.xlabel(str(df.groupby(by=feature)['R_s'].mean()))
        # plt.show()





if __name__ == '__main__':
    param_grid_dict = {
        'halfwidth_min': 'np.array([1.5, 2, 2.5]) * n_max/30',
        'nc2': 'np.array([0.5, 1, 1.5]) * n_max / 22',
        'halfwidth_max': 'np.array([1.5, 2, 2.5]) * n_max / 4',
        'q_tranfu': '[2, 3]'
        }
    show_best_R_s_with_params(data=data_20, path='',
                              param_grid_dict=param_grid_dict)






