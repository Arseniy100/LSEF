
createNstatioWSigma_S1 = function(nx, UU0_local, LL_u_local, crftype, nu=1){

#=========================================================================
# Generate a non-symmetric W-mx on S^1 such that
#
# B=CVM=W * W^T
#  
# - using a nstatio LSM (local-spectrum or moving-average) model.
#
# NB: CVM(i,j) = sum_k W_ik * w_jk
#     Each row x of W is generated using 
#     = a positive definite function of rho, u_0(rho)=u(x, rho), of type (crftype,nu),
#     = its maximum value UU0_local(x) = u(x,rho=0), and 
#     = its local length scale LL_u_local[i].
#     
# More specifically, 
# 
#    u(x,rho) = UU0_local(x) * u_0(rho / LL_u_local(x))
#     
# Also, calc sigma_n(x) as entries of the mx Sigma:
#    Sigma[ix,n+1] = sigma_n(x_ix)
# Here, n runs from 0 to nx-1 so that wvn=n
# 
# sigma_n(x) = sqrt(2*pi) FFT_{u -> n} {v(x,u)},
# v(x,u) is the even & pos-def fu we impose 
# (such that w(x,y) = v(x, y-x) * sqrt(dx))
#
# NB: The complex spectral coefficients produced by the R's function fft are ordered as follows:
#
# {\bf\tilde f} := {\tilde f_0}, \tilde f_1, \tilde f_2, \tilde f_3, ..., 
# \tilde f_{n/2-1}, \, {\tilde f_{n/2}},  
#    \tilde f_{-n/2+1}, \dots, \tilde f_{-3}, \tilde f_{-2}, \tilde f_{-1}  (*)
#
# that is, from wvn=0 go to the right up to $n_{max}:=nx/2$, 
# then jump to the very left to (-nmax+1) (but not to -nmax!)
# and then go up till nx=-1. 
# Thus, all wavenumbers are counted only once, running the whole circle.
#
#    Args: 
#
# nx - nu of points of the grid
# UU0_local - nstatio variance, an 1:nx vector 
# LL_u_local - nstatio len scale, an 1:nx vector (rad.)
# crftype - character variable. Can attain values:
#  "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern".
# nu ~ optional ~ used for "Bessel" & "Matern" & "Cauchy"
#      For "Bessel", nu >0.5
#      For "Bessel", nu > (d-2)/2, where the crf is defined in R^d.
#
# Return: W, Sigma[i,in], in \in [1,nx], wvn=n=in-1
# 
# M Tsy Jan 2020 
#=========================================================================
  
  #t1=proc.time()
  #=====
  #  Debug
  # nx=64
  # LL_u_local = c(1:nx)
  # LL_u_local[] = 2000
  # LL_u_local[1:(nx/2)] = 4000
  # crftype="Matern"
  #=====
  
  sq2pi=sqrt(2*pi)
  dx=2*pi/nx
  sqdx=sqrt(dx)
  
  W     = matrix(ncol = nx, nrow = nx)
  Sigma = matrix(ncol = nx, nrow = nx)
  v = c(1:nx) # init v-function (the centered w(x,y))
  
  
  omega=0.9 # for ecos: sh.be <1 on R2

  for(i in (1:nx)) {   # row
    u0=UU0_local[i]    # local u(x, rho=0) 
    L=LL_u_local[i]      # local len scale
    v[]=0
    
    for(j in (1:nx)) { # column
      d=abs(i-j)
      if ((d> floor(nx/2))) {d <- nx-d}
      rho <- dx * d   # rad.
      r=2*sin(rho/2) # chordal dist, unitless
      x=r/L
      
      crl =crfs_isotr(crftype, nu, x)
      
      # calc v(x,u)
      
      k=j-i+1          # to the right of the main diagonal (including it),
                       # k is the position of the entry wrt the main diagonal
                       # i=j corresponds to k=1
      if(k<1) k=k+nx   # to the left  of the main diagonal (periodicity)
      v[k] = u0 * crl       # v[k] has its max (center) at the main diagonal, k=0 or i=j
      
      W[i,j] = v[k] * sqdx
    }
    
    # calc sigma_n(x) = 1/sqrt(2*pi) FFT_{u -> n} {v(x,u)},
    
    Sigma[i,] = sq2pi * Re( fft(v, inverse = FALSE) ) /nx 
    
  }
  #proc.time() -t1
  #image2D(W)
  
  return(list("W"=W, "Sigma"=Sigma))
}