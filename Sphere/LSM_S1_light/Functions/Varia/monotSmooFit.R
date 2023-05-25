
monotSmooFit = function(z, w_smoo){

#*****************************************************************************
# Draw a Monotonically Increasing "function" y[1:n] least (L2) deviating from  
# n ``observations''  z1:n] = y[1:n] + eps[1:n],
# whr eps[i]= sig * N(0,1)  (sig unknown).
#  
#    Method
# 
# Represent  
# 
#    y[i] = sum u[j=1:i]              (1)
# 
# with u[j] subject to n inequality constraints:  
# 
#    u[j] >= 0                        (2)
#    
# Eq(1) implies that
# 
#    y = A * u,                        (3)
#   
# whr 
#      |1 0 0 0 0|
#      |1 1 0 0 0|
#  A = |1 1 1 0 0|
#      |1 1 1 1 0|
#      |1 1 1 1 1| etc.
#      
# Next, we impose a smoothness constraint on u. Let
# 
#   J_s = w * (u_2 - u_1)^2 + ... + (u_n - u_{n-1})^2
#       = w *(Tu,u)
# Here
#      
#      T[1,] = [ 1 -1 0 0 0 ]
#      T[2,] = [-1 2 -1 0 0 ]
#      T[3,] = [0 -1 2 -1 0 ]
#      T[4,] = [0 0 -1 2 -1 ]
#      T[5,] = [0 0 0 -1 1 ]
# 
# d J_s/(2*w) = (u1 - u2)*du1 + (u2 - u1 + u2 - u3)* du2 +...+ (u_n - u_{n-1})*dun
#      = (T*u)^T * du
#   
# (ie T is 3-diag, with c(1,2,2,...,2,1) on the main diag & -
#    1 on the 2 adjacent diagonals).
# 
# wT := w*T
#    
# Now, we formulate the problem as
# 
#     J(u) = (Au - z, Au - z) + (wT u,u)  -->  min
#     
#  Here, we distill the ``full square'' in order to arrive at the form
#  
#    (Cu - d, Cu - d) --> min           
#  
# We have
# *******************************
#    C = sqrt(A^T A +wT)
#    d = C A^T z
# *******************************   
# and finally solve the problem
# 
#    ||C*u - d|| --> min subject to u[j]>=0           (*)
# ...........................   
# Technically, we call function  
# lsqnonneg(C, d)  from  pracma:
# ..............................................
# C, d matrix and vector such that C x -d will be minimized with x >= 0.
# ..............................................
# 
# By the way, 
# 
#          |5 4 3 2 1|
#          |4 3 2 1 0|
#  A^T A = |3 2 1 0 0|
#          |2 1 0 0 0|
#          |1 0 0 0 0| etc.
# 
# 
#    Args
# 
# z - observations
# w_smoo - weeight of the smoothness constraint
# 
# Return y.
# 
# M Tsy 2020 Nov 
#*****************************************************************************

  # deb
  # z=c(1,3,7,4,9,8)
  # w_smoo=1
  # end deb
  
  n = length(z)
  
  # build A - transform u \mapsto y
  
  A = matrix (data = 0, nrow = n , ncol = n)
  
  for (i in 1:n){
    A[i,1:i] = 1
  }
  
  # build T - smoothness of u
  
  T = matrix (data = 0, nrow = n , ncol = n)
  
  diag(T)=rep(2, n)
  T[1,1] = 1
  T[n,n] = 1
  
  for (i in 2:n){
    T[i, i-1] = -1
    T[i-1, i] = -1
  }
  
  T = T * w_smoo
  
  # 2 ways: sqrt-form (way-1) and full-form (way-2)
  #       -- for square matrices the 2 ways are equivalent. 
  #       Way-1 is faster but the mx in way-2 can be better conditioned.
  
  way=2
  if(way == 1){
    
    # C = A^T A + T 
    # d = A^T z
    
    C = t(A) %*% A  +  T
    d = drop( t(A) %*% z )
    
    
  }else if(way == 2){
    
    # G = A^T A + w T  
    # C =sqrt(G)
    # d = 1/C A^T z
    
    G2 = t(A) %*% A  +  T
    C = symm_pd_mx_sqrt(G2)$sq
    ATz = drop(t(A) %*% z)
    d = solve(C, ATz)
  }
  
  # Solve C*u = d
  
  u = lsqnonneg(C, d)$x
  y=drop( A %*% u )
  
  sqrt(crossprod(y-z, y-z) / crossprod(y, y))

  mn=min(z,y)
  mx=max(z,y)
  plot(z, ylim=c(mn,mx))
  lines(y)

  return(y)
}