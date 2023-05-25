evalFuScaledArg = function(xx, f, a){

#-----------------------------------------------------------------------
# Given function f defined on the regular grid xx[1:nx],
# evaluate the function
#
# f_scaled(x) = f(x/a)
#
# on the same grid xx, ie   find  f_scaled(xx).
# If some  xx/a  happen to lie outside the range of  xx, extrapolate  f  by const.
#
#   Args
#
# xx[1:nx] - Regular grid on which the input fu  f  is defined and on which the
#            output  f_scaled  is to be evaluated.
# f[1:nx] - input function on the grid
# a - scale
#
# return:  f_scaled
# M Tsy 2021 Jan
#-----------------------------------------------------------------------

  nx = length(xx)

  tt = xx / a   # args of  f  on the scaled grid

  # intpl f_shape from the grid xx to the grid  yy

  f_scaled = LinIntpl(xx, f, tt)

  return(f_scaled)
}