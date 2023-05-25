# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:42:28 2020
Last modified: Dec 2021

@author: Arseniy Sotskiy
"""

import numpy as np
import matplotlib.pyplot as plt
# from tqdm.autonotebook import tqdm
from tqdm import tqdm, trange
#%%


def flatten_field(field: np.array) -> np.array:
    '''
    Flattens 2D field on sphere to 1d array of points
    Starts from the North Pole and goes to South
    Each Pole becomes one point

    Parameters
    ----------
    field : np.array
        2D array
        1st and last row must consist of equal numbers
            (if not, then field[0,0] and [-1,0] is taken as pole value)
        1st and last columns must be equal
            (if not, then last column is ignored)

    Returns
    -------
    flattened_field : np.array
    '''
    nlat = field.shape[0] # size of shtools
    nlon = field.shape[1]
    npoints = nlat * (nlon - 1) - 2 * (nlon-1) + 2

    flattened_field = [0]*npoints
    flattened_field[0] = field[0,0]
    flattened_field[-1] = field[-1,0]
    for i in range(1, nlat - 1):
        flattened_field[(i-1) * (nlon-1) + 1 : i * (nlon - 1) + 1] = field[i,:-1]
    return np.array(flattened_field)


if __name__ == '__main__':
    # test
    field = np.array(
        [[1,1,1,1,1,1,1,1,1],
         [1,2,3,4,5,6,7,8,1],
         [11,12,13,14,15,16,17,18,11],
         [21,22,23,24,25,26,27,28,21],
         [2,2,2,2,2,2,2,2,2]]
        )
    field2 = np.array(
        [[ 1,-2,-3,-4,-5,-6,-7,-8,-9],
         [1,2,3,4,5,6,7,8,0],
         [11,12,13,14,15,16,17,18,0],
         [21,22,23,24,25,26,27,28,0],
         [2,-2,-2,-2,-2,-2,-2,-2,0]]
        )
    flattened_field = flatten_field(field)
    print(flattened_field)
    flattened_field2 = flatten_field(field2)
    print(flattened_field2)


#%%


def make_2_dim_field(field: np.array, nlat: int, nlon: int) -> np.array:
    field_2d = np.zeros((nlat, nlon))
    field_2d[0,:] = field[0]
    field_2d[-1,:] = field[-1]
    for i in range(1, nlat - 1):
        field_2d[i,:-1] = field[(i-1) * (nlon-1) + 1 : i * (nlon - 1) + 1]
    field_2d[:,-1] = field_2d[:,0]

    return field_2d


if __name__ == '__main__':
    # test
    field = np.array(
        [[1,1,1,1,1,1,1,1,1],
         [1,2,3,4,5,6,7,8,1],
         [11,12,13,14,15,16,17,18,11],
         [21,22,23,24,25,26,27,28,21],
         [2,2,2,2,2,2,2,2,2]]
        )
    nlon, nlat = field.shape
    flattened_field = flatten_field(field)
    print(flattened_field)
    field_2d = make_2_dim_field(flattened_field, nlon, nlat)
    print(field_2d)
    assert (field_2d == field).all()



#%%
def coarsen_grid(field: np.array, k: int) -> np.array:
    '''
    Makes new grid which takes every k-th point on the grid
    Suppose field.shape[0] = nlat. Then k should be such that
    k divides nlat - 1: nlat - 1 = k * (new_nlat - 1)

    Parameters
    ----------
    field : np.array
        2D np.array.
    k : int

    Returns
    -------
    new_field : np.array
        coarsed field.

    '''
    if (field.shape[0] - 1) % (2 * k) != 0:
        print('Bad k! (nlat - 1 ) % (2 * k) != 0')
        return None
    new_field = field[::k, ::k]
    return new_field

if __name__ == '__main__':
    # test
    field = np.array(
        [[1,1,1,1,1,1,1,1,1],
         [1,2,3,4,5,6,7,8,1],
         [11,12,13,14,15,16,17,18,11],
         [21,22,23,24,25,26,27,28,21],
         [2,2,2,2,2,2,2,2,2]]
        )
    field.shape
    new_field = coarsen_grid(field, 2)
    print(new_field)

    new_field = coarsen_grid(field, 4)
    print(new_field)



#%%

def interpolate_1D_slow(arr, k):
    for i in range(len(arr) - 1):
        # print(i, i - i%k)
        # print(arr[i - i%k + k], (i%k)/k,  arr[i - i%k], (k - i%k)/k)
        arr[i] = arr[i - i%k + k]*(i%k)/k + arr[i - i%k]*(k - i%k)/k
    return arr

def interpolate_1D(arr, k):
    # print(f'interpolating {arr}')
    nums = arr[::k]
    mx = np.zeros((len(nums)-1, k+1))
    mx[:,0] = nums[:-1]
    mx[:,-1] = nums[1:]

    for i in range(1, k):
        mx[:,i] = mx[:,0]*(k-i)/k + mx[:,-1]*i/k
    # print(mx)
    arr = np.append(mx[:,:-1].reshape(1,-1), nums[-1])
    # print(f'result: {arr}')
    return arr

if __name__ == '__main__':
    arr = np.array([1, 0, 0, 0, 5, 0, 0, 0, 9])
    k = 4
    print(interpolate_1D(arr, k))
    print(interpolate_1D_slow(arr, k))

#%%

def interpolate_grid(field: np.array, k: int) -> np.array:
    '''
    Makes new grid which is k*k times bigger than given
    Interpolates every row, then adds interpolated rows

    Parameters
    ----------
    field : np.array
        2D np.array.
    k : int

    Returns
    -------
    new_field : np.array
        interpolated field.

    '''

    new_field = np.zeros(np.array(field.shape) * k - (k - 1))
    new_field[::k, ::k] = field
    new_field = np.apply_along_axis(interpolate_1D, 1, new_field, k)
    new_field = np.apply_along_axis(interpolate_1D, 0, new_field, k)
    new_field
    return new_field

if __name__ == '__main__':
    # test
    field = np.array(
        [[1,1,1,1,1,1,1,1,1],
         [1,2,3,4,5,6,7,8,1],
         [11,12,13,14,15,16,17,18,11],
         [21,22,23,24,25,26,27,28,21],
         [2,2,2,2,2,2,2,2,2]]
        )
    field.shape
    small_field = coarsen_grid(field, 2)
    print(small_field)
    big_field = interpolate_grid(small_field, 2)
    print(big_field)
    # big_field = interpolate_grid(small_field, 3)
    # print(big_field)

    # new_field = coarsen_grid(field, 4)
    # print(new_field)







#%%


def grid_rho(grid, i, j):
    '''
    computes distance between points x_i and x_j on sphere

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    lon_i = grid.lons[i]
    lon_j = grid.lons[j]
    lat_i = grid.lats[i]
    lat_j = grid.lats[j]
    delta_lon = np.abs(lon_i - lon_j)
    under_arccos = np.sin(lat_i) * np.sin(lat_j) + \
            np.cos(lat_i) * np.cos(lat_j) * np.cos(delta_lon)
    under_arccos = min(under_arccos, 1)
    under_arccos = max(under_arccos, -1)
    distance = np.arccos(under_arccos)

    return distance


def make_matrix_from_column(matrix: np.array):
    '''


    Parameters
    ----------
    matrix : np.array
        2d square matrix. right column is filled,
        other part iz zeros.

    Returns
    -------
    None.

    '''
    pass



[[(j,i) for i in range(3)] for j in range(8)]


class LatLonSphericalGrid:
    '''
    class of grid on sphere
    keeps one point for each point of grid
    (unlike pyshtools.SHGrid)
    '''

    def __init__(self, nlat):
        self.nlat = nlat
        self.nlon = (self.nlat - 1) * 2
        # self.n_max = n_max
        self.dlon = 2 * np.pi / self.nlon
        self.dlat = self.dlon
        self.npoints = self.nlat * self.nlon - 2 * self.nlon + 2
        
        self.coords_2d_to_1d = make_2_dim_field(
            np.arange(self.npoints), self.nlat, self.nlon + 1
            )
        self.coords_2d_to_1d = self.coords_2d_to_1d.astype(int)
        # print(self.coords_2d_to_1d)
        
        indices =[[(j,i)
              for i in range(self.nlon + 1)] for j in range(self.nlat)]
        # print(indices)
        # print(np.array(indices).shape)
            
        # for i in range(len(indices)):
        #     print(indices[i])
        np_indices = np.array(indices, dtype=tuple)
        # for i in range(len(np_indices)):
        #     print(np_indices[i])
        self.coords_1d_to_2d = flatten_field(
            np_indices
            )
        
        # print(self.coords_1d_to_2d)

        
        
        
 

        # now making array of delta areas:
        # print('delta areas:')
        theta = np.array(
            [[(2*i/(self.nlat-1) - 1) * np.pi / 2 for i in range(self.nlat)]]
            )  # from - pi to pi
        # print('theta:', theta)
        d_theta = self.dlat
        d_phi = self.dlon
        weights = np.matmul((np.cos(theta)).T, np.ones((1, self.nlon+1)))
        weights[0,:] = np.ones((self.nlon+1)) * np.sin(d_theta / 4)\
        * np.sin(d_theta / 4) / np.sin(d_theta / 2)
        weights[-1,:] = np.ones((self.nlon+1)) * np.sin(d_theta / 4)\
        * np.sin(d_theta / 4) / np.sin(d_theta / 2)
        ones = np.ones((self.nlat, self.nlon+1))
        # print('weights:', weights)

        # plt.figure()
        # plt.imshow(weights)
        # plt.colorbar()
        # plt.show()
        areas = np.multiply(ones, weights) * d_phi * \
                            2 * np.sin(d_theta / 2)
        # print('areas:', areas)

        # print(d_phi * (np.cos(theta + np.pi/2 - d_theta/2)\
        #                - np.cos(theta + np.pi/2 + d_theta/2)))
        # print(1 - np.cos(3*d_theta/2))
        # print(np.sum(areas) / 4 / np.pi)
        # print('\n'*5)
        self.areas = flatten_field(areas)
        self.areas[0] *= self.nlon
        self.areas[-1] *= self.nlon
        self.areas_sqrt = np.sqrt(self.areas)


        colats_1D = np.array([np.linspace(0, np.pi, self.nlat)])
        lats_1D = -np.array([np.linspace(-np.pi/2, np.pi/2, self.nlat)])
        lons_1D = np.array([np.linspace(0, 2*np.pi, self.nlon+1)])
        colats = np.matmul(colats_1D.T, np.ones((1, self.nlon+1)))
        lats = np.matmul(lats_1D.T, np.ones((1, self.nlon+1)))
        lons = np.matmul(np.ones((self.nlat, 1)), lons_1D)
        # print(lats)
        # print(lons)
        self.lons = flatten_field(lons)
        self.colats = flatten_field(colats)
        self.lats = flatten_field(  lats)

        # assert False, 'should make 2D fields such that flatten is correct'
        self.rho_matrix = None
        self.rho(0,0)
        
        
        

    def rho(self, i, j): # terrible, must be parallelised
        
        if self.rho_matrix is None:
            # print()
            # self.rho_matrix = np.array([
            #     [grid_rho(self,i,j) for i in range(self.npoints)]
            #     for j in range(self.npoints)])
            # draw_2D(self.rho_matrix)
            rho_matrix_1 = np.zeros((self.npoints,self.npoints))
            # for i in tqdm(range(1, self.nlat - 1), 
            #               desc='computing rho matrix'):
            for i in range(1, self.nlat - 1):
                for j in range(0, self.npoints):
                    rho_matrix_1[j, i * self.nlon] = grid_rho(self, j,
                                                              i * self.nlon)
                    for k in range(0, j % self.nlon):
                        rho_matrix_1[j - k, i * self.nlon - k] = rho_matrix_1[j, i * self.nlon]
                    if j % self.nlon == 0 and j != 0:
                        val = rho_matrix_1[j, i * self.nlon]
                        for k in range(0, self.nlon):
                            rho_matrix_1[j - k, i * self.nlon - k] = val / 2
    
            # draw_2D(rho_matrix_1)
            rho_matrix_1 += rho_matrix_1.T
            for i in range(-1, 1):
                for j in range(0, self.npoints):
                    rho_matrix_1[j, i] = grid_rho(self, j, i)
            for j in range(-1, 1):
                for i in range(0, self.npoints):
                    rho_matrix_1[j, i] = grid_rho(self, j, i)
            # draw_2D(rho_matrix_1)
            # draw_2D(self.rho_matrix - rho_matrix_1)
    
            self.rho_matrix = rho_matrix_1
            
        return self.rho_matrix[i,j]
    
    def transform_coords_2d_to_1d(self, coord_2d):
        '''
        Parameters
        ----------
        coord_2d : tuple ot list of tuples
            2d coordinates.

        Returns
        -------
        coord_2d : int or np.array of ints
            transformed coordinates.

        '''
        coords = np.array(coord_2d, dtype=int)
        # print(self.coords_2d_to_1d)
        # print(np.array(coord_2d))
        dim = len(coords.shape)
        if dim == 1:
            coord_1d = self.coords_2d_to_1d[coords[0], coords[1]]
        elif dim == 2:
            if np.array(coord_2d).shape[0] == 2:
                coord_1d = self.coords_2d_to_1d[coords[0], coords[1]]
            else:
                coord_1d = (self.coords_2d_to_1d[coords[:,0], coords[:,1]])
        return coord_1d
    
    def transform_coords_1d_to_2d(self, coord_1d):
        coord_2d = self.coords_1d_to_2d[coord_1d]
        return coord_2d

    def __repr__(self):
        return f'LatLonSphericalGrid(nlat={self.nlat}, nlon={self.nlon},' + \
               f'npoints={self.npoints})'





if __name__ == '__main__':
    grid = LatLonSphericalGrid(5)
    print(grid)
    print(grid.areas, grid.areas.sum() / 4 / np.pi)
    print(grid.lats / np.pi)
    print(grid.colats / np.pi)
    print(grid.lons / np.pi)
    


    print(len(grid.lats))
    print(len(grid.lons))

    print(grid.rho(0, -1))
    print(grid.rho(0, 9))
    print(grid.rho(9, 10))
    
    
    for coord_1d in range(grid.npoints):
        coord_2d = grid.transform_coords_1d_to_2d(coord_1d)
        coord_1d_ = grid.transform_coords_2d_to_1d(coord_2d)
        print(coord_1d, coord_2d, coord_1d_)
    
    coords_2d = [(0,0), (0,1), (1,-1), (1,3), 
                     (3,5), (4,2), (2,6), (4,6)]
    for coord_2d in coords_2d:
        coord_1d = grid.transform_coords_2d_to_1d(coord_2d)
        print(coord_2d, coord_1d, grid.transform_coords_1d_to_2d(coord_1d))
        
    coords_1d = grid.transform_coords_2d_to_1d(coords_2d)
    print(coords_2d, coords_1d, grid.transform_coords_1d_to_2d(coords_1d))
    # draw_2D(grid.rho_matrix)
    
    coords_2d = [(0,0,1,1,3,4,2,4), (0,1,-1,3,5,2,6,6)]
    coords_1d = grid.transform_coords_2d_to_1d(coords_2d)
    print(coords_2d, coords_1d, grid.transform_coords_1d_to_2d(coords_1d))




