
Sigma2WB = function(Sigma){

  #=========================================================================
  # In a nstatio LSM (local-spectrum or moving-average) model,
  # from the mx Sigma:
  #    Sigma[ix,n+1] = sigma_n(x_ix) == sqrt(b_n*(x_ix)),
  # (Here, n runs from 0 to nx-1 so that wvn=n)
  # compute the convolution mx W and the CVM B.
  #
  # B=CVM=W * W^T
  # Each row i of W is generated from the even & positive definite function v(x_i, u),
  #  whr it is u which is changing.
  #
  #  w(x,y) = v(x, y-x) * sqrt(dx)
  #
  # v(x,u) = 1/sqrt(2*pi) FFT_{n->u} sigma_n(x)
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
  # Sigma - an nx*nx real mx
  #
  # Return: W, B.
  # 
  # M Tsy Feb 2020 Tested OK
  #=========================================================================

  nx = dim(Sigma)[1]
  sq2pi1 = 1/sqrt(2*pi)
  dx=2*pi/nx
  sqdx=sqrt(dx)
  
  W = matrix(ncol = nx, nrow = nx, data=0)
  B = matrix(ncol = nx, nrow = nx, data=0)

  for(i in (1:nx)) {   # row
 
    # v(x,u) = 1/sqrt(2*pi) FFT_{n->u} sigma_n(x)
    
    v = sq2pi1 * Re( fft(Sigma[i,], inverse = TRUE)  )
    
    # form W[i,]

    for(j in 1:nx) {   # column, the whole circle
      
      k=j-i+1          # to the right of the main diagonal (including it),
                       # k is the position of the entry wrt the main diagonal
                       # j=i corresponds to k=1
      if(k<1) k=k+nx   # to the left  of the main diagonal (periodicity)
      W[i,j] = v[k] * sqdx
    }
  }
  B=W %*% t(W)
  
  #image2D(W, main="W")
  #image2D(B, main="B")
  
  return(list("W"=W, "B"=B))
}