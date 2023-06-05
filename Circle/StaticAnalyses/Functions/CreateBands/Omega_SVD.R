
Omega_SVD = function(tranfu2, lplot){
  #------------------------------------------------------------------------------
  # Compute the matrix that relates Band Variances to the spectrum:
  # 
  #   d = Omega * y                    (*)
  #   
  # (where d (data) are the Band Variances and y is the spectrum)
  # and compute its SVD.
  # 
  # More specifically, two matrices  Omega  are computed.
  # The 1st one is for  y = b[1:nmaxp1]:  Omega_nmax,
  # The 2nd one is for  y = b[1:nx]:  Omega_S1.
  #
  #    Methodology
  #    
  # -------------------------------------------------------------------
  # (1)  Omega_nmax 
  # 
  #      Omega_nmax is a J*N mx  (N=nmaxp1)
  #  d[1:J] = sum tranfu2[1:nx] * b[ix,1:nx]
  # As length(y) = nmaxp1=N  and  b[1:nx] is an Even fu on S1(nx),
  #          Omega_nmax[,1] = tranfu2[1,]
  #   1<n<N: Omega_nmax[,n] = tranfu2[i_n,] + tranfu2[i_n_symm,] 
  #          Omega_nmax[,N] = tranfu2[N,]
  # 
  # -------------------------------------------------------------------
  # (2)  Omega_S1
  # 
  # To avoid "singularity" at n=0 & n=nmax in the  "observation operator"  Omega, 
  # we assume that the whole circle of  b_n  are unknown 
  # ( despite the fact that for the real valued field in question  xi,  b(-n)=b(n) ).
  # [ We acknowledge this latter property by averaging, at the end, the 
  # resulting  b_fit(n)  over (+n) and (-n). ]
  # Correspondingly, we require that all wvns on the circle are "observed" 
  # by the band variances.
  # To achieve this, we place  J-2  fictitious "symmetric" bands to cover 
  # negative wvns.
  #-------------------------------------------------
  # tranfu2_S1[j]  are placed symmetrically  on S1:
  # the "symmetric" bands  j and j_symm  satisfy
  # 
  #   tranfu2_S1(n,j) = tranfu2_S1(-n,j_symm)
  # or
  #   tranfu2_S1(n, j) = tranfu2_S1(n_symm, j_symm)          (1)
  # Here 
  #   j_symm(J+l) = J-l   ==>  For j=J+l=2:K ==> l=j-J  ==>  j_symm(j) = J-l = J-(j-J) = 2*J - j
  # -----------------------------------
  #   j=2:K  ==>    j_symm(j) = 2*J - j            (2)
  # -----------------------------------
  #  j=1:J  ==>  tranfu2_S1[np1=1:nx, j] =  tranfu2[np1=1:nx, j]
  #  j=2:K  ==>  tranfu2_S1[np1=1:nx, j] =  tranfu2[np1_symm(np1), j_symm]
  #  
  #    n_symm(n) = -n  
  # 
  # The upper semi-circle (convert n_symm from negative to positive -- to compute i_n_symm, 
  # using periodicity): 
  # i_n = 1:nmaxp1  <=>  n=i_n-1=0:nmax  <==>  n_symm = -n = nx-n  ==> 
  # i_n_symm = nx-n+1 = nx - i_n +2
  # 
  # The lower semi-circle (convert n from positive to negative -- to compute i_n, using periodicity):
  # i_n = nmaxp2:nx  <=>  n=i_n-1-nx (neg)  ==>  n_symm = -n = nx + 1 - i_n (pos)
  # i_n_symm = n_symm +1 = nx - i_n +2  (exactly as for the upper semi-circle!)
  #  Thus,
  #  
  #   i_n_symm(i_n) = nx - i_n +2  (except i_n=1)
  #   
  #-------------------------------------------------
  # NB: band_V_S1  are to be specified to be equal for  j and j_symm  on input:
  #   
  #   j=1:J  ==> band_V_S1[,j] = band_V[],j]
  #   j=2:K  ==> band_V_S1[,j] = band_V[,j_symm(j)]
  #
  #  NB The suffix  S1  denotes the whole circle. 
  #-------------------------------------------------
  #          Omega_S1 = t( tranfu2_S1[,] )
  # -------------------------------------------------------------------          
  # 
  #    Args
  # 
  # tranfu2[i_n=1:nx, j=1:J] - |tranfu|^2 for [i_n, J], i_n=1,...,nx,  i_n=n+1
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: Omega_nmax, Omega_S1, SVD_Omega_nmax, SVD_Omega_S1
  #         
  # M Tsy 2021 Feb
  #-------------------------------------------------------------------------------------------
  
  nx = dim(tranfu2)[1] 
  J  = dim(tranfu2)[2] ;  K = 2 * (J-1)

  nmax=nx/2
  nmaxp1=nmax+1
  nmaxp2=nmax+2
  N = nx 
  
  #-----------------------------------------------------------------
  #-----------------------------------------------------------------
  # Compute Omega_nmax 
  # (a J*nmaxp1  matrix)
  # Build the band-variance weighting mx Omega.  Omega is a J*N mx.
  #          Omega[,1] = tranfu2[1,]
  #   1<n<N: Omega[,n] = tranfu2[i_n,] + tranfu2[i_n_symm,] 
  #          Omega[,N] = tranfu2[N,]
  
  Omega_nmax = t( tranfu2[1:nmaxp1,] ) 
  Omega_nmax[,2:nmax] = Omega_nmax[,2:nmax] +  t( tranfu2[rev(nmaxp2:nx),] ) 
  
  # test Omega
  # d_true = t( apply(b_true[,1:nmaxp1], 1, function(t) Omega %*% t) )
  # max(abs(band_Vt-d_true))
  
  # SVD
  
  SVD_Omega_nmax = svd(Omega_nmax)
  
  if(lplot){
    
    UU = SVD_Omega_nmax$u
    VV = SVD_Omega_nmax$v
    sval  = SVD_Omega_nmax$d
    
    image2D(UU, main="UU_nmax")
    image2D(VV, main="VV_nmax")
    j=0
    j=j+1
    plot(UU[,j], main = paste0("UU_nmax, j=",j), type="l")
    
    j=0
    j=j+1
    plot(VV[,j], main = paste0("VV_nmax, j=",j), type="l")
    
    plot(sval, main = "sval_nmax")
    min(sval)
    sval
  }
  #-----------------------------------------------------------------
  #-----------------------------------------------------------------
  #  Compute Omega_S1
  # (a J*nx  matrix)
  #
  # 1) From  tranfu2  to  tranfu2_S1
  #   i_n_symm(i_n) = nx - i_n +2  
  
  tranfu2_S1 = matrix(0, nrow = nx, ncol = K) # [i_n, j=1:K]
  tranfu2_S1[,1:J] = tranfu2
  
  n = 0: (nx-1) ;  i_n = n +1
  n_symm = -n 
  n_symm = (n_symm + nx) %% nx
  i_n_symm = n_symm +1

  for (j in (J+1):K){
    j_symm = 2*J - j 
    tranfu2_S1[i_n, j] = tranfu2[i_n_symm, j_symm]
  }
  
  # plot(tranfu2_S1[1:nx, 1], main=paste0("tranfu2_S1"), 
  #      xlab="Wavenumber", ylab="Spectral transfer function", ylim=c(0,1), type="l")
  # if(K > 1){
  #   for (k in 2:K){
  #     lines(tranfu2_S1[1:nx, k])
  #   }
  # }
  
  #-----------------------------------------------------------------
  # Build  Omega_S1  (a J*nx mx).

  Omega_S1 = t( tranfu2_S1 ) 
  
  # # test Omega
  # image2D(Omega)
  # y = b_true[,]  #[ix, i_n]
  # d_true = t( apply(y, 1, function(t) Omega %*% t) )
  # delta = band_Vt_S1 - d_true
  # image2D(delta, main="band_Vt_S1 - d_true")
  # max(abs(delta))
  # image2D(band_Vt_S1, main="band_Vt_S1")
  # image2D(d_true, main="d_true")
  #-----------------------------------------------------------------------
  #-----------------------------------------------------------------------
  # SVD
  # Omega = UU * D * VV^T
  # d = Omega * y = UU * D * VV^T  * y     ==>
  # 
  #    UU^T * d = D * * VV^T  * y    (1)
  # 
  # From (1), expand  d  in the basis formed by the columns of  UU, getting  d_ 
  #      and  expand  y  in the basis formed by the columns of  VV, getting  y_
  #      so that
  #   
  #    d_ = D * y_
  # or
  # 
  #    d_[k] = sval[k] * y_k,           (2)
  # 
  # whr
  #    sval = diag(D)
  # 
  # The resulting y is then
  # 
  #    y = V * y_.
  #    
  # NB: Only  J  first columns of  VV  are actually used here
  # (this is automatically produced by the R's svd).
  # 
  # Solution:
  # 
  #   y_[k] = d_k / sval[k],           (3)
   
  SVD_Omega_S1 = svd(Omega_S1)
  
  if(lplot){
    UU = SVD_Omega_S1$u
    VV = SVD_Omega_S1$v
    sval  = SVD_Omega_S1$d
    
    image2D(UU, main="UU_S1")
    image2D(VV, main="VV_S1")
    j=0
    j=j+1
    plot(UU[,j], main = paste0("UU_S1, j=",j), type="l")
    
    j=0
    j=j+1
    plot(VV[,j], main = paste0("VV_S1, j=",j), type="l")
    
    plot(sval, main = "sval_S1")
    min(sval)
    sval
  }
  
  #-----------------------------------------------------------------------  
  
  return(list("Omega_nmax"=Omega_nmax,
              "SVD_Omega_nmax"=SVD_Omega_nmax, 
              "Omega_S1"=Omega_S1, 
              "SVD_Omega_S1"=SVD_Omega_S1))
}