
B2S_SVshape = function(band_V, Omega_S1, SVD_Omega_S1,
                       b_shape, moments, band_centers_n,
                       BOOTENS, band_Ve_B, 
                       nSV_discard, a_max_times, w_a_fg, true_spectra_available, lplot){
  #------------------------------------------------------------------------------
  #  Bands To Spectrum.
  # Fit the spectrum  b_fit[ix,i_n]  to the set of band variances  d=band_V[ix,1:J]   
  # given the Obs Matrix  Omega_S1[1:K, i_n=1:nx].
  # Perform at each ix independently.
  #
  #    Methodology
  #    
  # To avoid "singularity" at n=0 & n=nmax in the  "observation operator"  Omega, 
  # we assume that the whole circle of  b_n  are unknown 
  # ( despite for the real valued field in question  xi,  b(-n)=b(n) ).
  # [ We acknowledge this latter property by averaging, at the end, the 
  # resulting  b_fit(n)  over (+n) and (-n). ]
  # Correspondingly, we require that all wvns on the circle are "observed" 
  # by the band variances.
  #-------------------------------------------------
  # band_V_S1  are specified to be equal for  j and j_symm  on input:
  #   
  #   j=1:J  ==> band_V_S1[,j] = band_V[],j]
  #   j=2:K  ==> band_V_S1[,j] = band_V[,j_symm(j)]
  #
  #  NB The suffix  S1  denotes the whole circle. 
  #-------------------------------------------------
  # At ix, the observed band-j variance ("data") is  
  # 
  #   d[j=1:K] = band_V_S1[ix,j]
  # 
  # The obs eqn is then
  #  
  #    d =  Omega_S1 * y + err                                        (2)
  # 
  # Here  the error term  err  comes from both error in  d (sampling error in
  # the ensemble) and error in  d_mod  (methodological error).
  # 
  # After finding the SVD regularized solution,  b_shape  is fit to it
  # (because b_svd is wavy and non-positive ..)
  # 
  #    Args
  # 
  # band_V[ix=1:nx, j=1:J] - BAND variances at the grid point  ix
  #   (normally, estimated from the ensemble or may be the true band variances) 
  # Omega_S1[j=1:K, i_n=1:nx]  (whr i_n=n+1) - the Obs mx
  # SVD_Omega_S1 - SVD of Omega_S1: contains 3 components: u,v,d (SVD)
  # b_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*b_shape(n/a)
  # moments - which moments to equate: "01" or "12" or "012"
  # band_centers_n - centers of wave bands
  # BOOTENS - TRUE if the bootstrap band variances  band_Ve_B  are available
  # band_Ve_B - bootstrap sample of  band_V, [ix,j=1:J,iB] (O means
  #               all bands arouund the spectrral circle)
  # nSV_discard - how many trailing SV are to be discarded 
  #               to filter out sampling noise
  # a_max_times = max deviation of  a  in SHAPE in times 
  # w_a_fg - weight of the ||a-1||^2 weak constraint
  # true_spectra_available - TRUE spectra and true band variances are provided testing
  #    (if TRUE,  b_true, b_true_Ms  and  band_Vt, band_Vt_S1  are taken from the environment!)
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  #         aa, AA,
  #         rel_err_discard_nullspace_b - rel info loss in removing the 
  # Omega_S1-null.space components from  \vec b (sh.be < 20%)  
  #       
  # Required functions:
  # fitScaleMagn
  # 
  # 
  #   
  # M Tsy 2021 May
  #-------------------------------------------------------------------------------------------
  
  nx = dim(band_V)[1] 
  J  = dim(band_V)[2] ;  K = 2 * (J-1)
  if(BOOTENS) nB = dim(band_Ve_B)[3]

  nmax=nx/2
  nmaxp1=nmax+1
  N = nx # dim-ty of the unknown spectrum  y
  
  b_fit = matrix(0, nrow = nx, ncol = nx) # [ix,i_n]
  b_svd = matrix(0, nrow = nx, ncol = nx) # [ix,i_n]
  if(BOOTENS) b_svd_B = array(0, dim=c(nx,nx,nB))
  
  UU = SVD_Omega_S1$u
  VV = SVD_Omega_S1$v
  sval = SVD_Omega_S1$d
  
  #-----------------------------------------------------------------
  # From  band_V  to  band_V_S1 : add fictitious "symmetric" bands
  # 
  #   j=1:J  ==> band_V_S1[,j] = band_V[],j]
  #   j=2:K  ==> band_V_S1[,j] = band_V[,j_symm(j)]
  # Here
  #   j=2:K  ==>    j_symm(j) = 2*J - j 
  
  band_V_S1 = matrix(0, nrow = nx, ncol = K) # [ix, j=1:K]
  band_V_S1[,1:J] = band_V
  
  jj = (J+1):K
  jj_symm = 2*J - jj
  band_V_S1[, jj] = band_V[,jj_symm]
  
  if(BOOTENS){
    band_Ve_B_S1 = array(0, dim=c(nx,K,nB)) # [ix, j, iB]
    band_Ve_B_S1[,1:J,] = band_Ve_B[,1:J,]
    band_Ve_B_S1[,jj,] = band_Ve_B[,jj_symm,]
  }
  
  #-------------------------------------------------------------------
  #    Tests
  # Evaluate info loss from removing the Omega-null.space components from  \vec b.
  # Find out how well b_true can be represented in the V-space
  # (in the column basis of V).
  # That is, how large is the information loss due to the switching from 
  # the spectrum (given that is is smooth) to band variances.
  # To evaluate the loss, switch from b_n to v_j and then back.
  #    Omega = U D V^T
  # Here U,V are "tall" ie containing only those columns that 
  # correspond to non-zero singular values d_n, n=1,...,r.
  # We assume that the bands' tranfu are not linearly dependent ==> 
  # Omega is non-singular ==> r=K.
  # NB: we consider here  K  bands (K=2*J-2), with J-2 bands 
  #     being "symmetric" on the wvn circle, but we do not assume here
  #     that b(-n)=b(n),  therefore, Omega is non-singular.
  #     
  #   U is K*K
  #   V is nx*K
  #   D is K*K diagonal non-degenerate
  #   
  # NB: 
  #  range(Omega)   = range(U)  
  #  range(Omega^T) = range(V)
  #  Ker(Omega)   = orthocomplement(range(Omega^T)) = orthocomplement(range(V))
  #  Ker(Omega^T) = orthocomplement(range(Omega))   = orthocomplement(range(U))
  # 
  # 1) Expand \vec b into all  K  orthogonal vectors v_k:
  #       b = sum _b_r *v_k
  # 2) Multiply  V_tall^T *b = V_tall^T * sum _b_r *v_k
  #    Since  v_k  are orthogonal, we get
  #       V_tall^T *b = (_b_1,..., _b_K), 
  #    That is, multiplying  \vec b  by  V^T  REMOVES all v-components in  \vec b
  #    that are in the null.space of V ==  null.space of Omega.
  # Vectors in  Ker(V)=Ker(Omega)  are "invisible" by  ensm varc  \vec v == \vec e. 
  
  if(true_spectra_available){
    y = t(b_true[,]) # y[i_n, ix]
    
    nSV = K - nSV_discard
    y_restore = drop( VV[,1:nSV] %*% crossprod(VV[,1:nSV], y) )
    
    max(abs(y_restore - y))/ max(abs(y)) # 10%
    rel_err_discard_nullspace_b = mean(abs(y_restore - y))/ mean(abs(y)) # 15%
    
    # Test Omega_S1
    
    Test_Omega_S1=F
    if(Test_Omega_S1){
      Omega_b = tcrossprod(Omega_S1, b_true)
      max(abs(t(Omega_b) - band_Vt_S1)) # sh.be very small
    }
    
    Test_S2B_infoLoss=F
    if(Test_S2B_infoLoss){
      rel_err_discard_nullspace_b
      
      ix=sample(1:nx, 1)
      nm=nmax/2
      mn=min(y[,ix], y_restore[,ix])
      mx=max(y[,ix], y_restore[,ix])
      plot(y[1:nm, ix], ylim=c(mn,mx))
      lines(y_restore[1:nm, ix])
      
    }
  } else{
    rel_err_discard_nullspace_b = NULL
  }
  #-------------------------------------------------------------------
  # Pseudo-Inversion: Main SVD solver section
  # 
  # d = U D V^T y 
  # d_ = U^T d
  # y_ = V^T y
  # 
  # The pseudo-inverse solution
  # y_ = D_plus d_
  # 
  # y = V Sigma^+ U^T * d
  
  # Calc  D_plus  while discarding  nSV_discard  trailing singular values

  sval_discard_plus = c(1:K) ;  sval_discard_plus[]=0
  Knz = K - nSV_discard
  sval_discard_plus[1:Knz] = 1 / sval[1:Knz] # _plus refers to the pseudo-inv mx  Omega^+
  sval_discard_plus[sval_discard_plus == Inf] = 0
  
  # data:  d[j, ix]
  
  d = t(band_V_S1)
  
  # data in U-space:  d[j,ix]  projected on the columns of  U, at all  ix.

  d_ = crossprod(UU, d)  # d_[k,ix].  U^T*d
  
  # Solution in U-space:  y_[k,ix]
  # Mult every column of  d_[k,ix]  by  the vector sval_discard_plus[]
  # (k labels the SVecs)
  
  y_ = sweep(d_, MARGIN = 1, sval_discard_plus, `*`) # [k,ix]
  
  # The solution  y[i_n,ix]
  
  y = VV %*% y_
  b_svd = t(y)  # [ix,i_n]
  
  # for (ix in 1:nx){
  #   d = band_V_S1[ix,]
  #   d_ = drop(crossprod(UU, d)) # d  in the UU-column space
  #   y_ = d_ / sval                 # y  in the VV-column space
  #   
  #   if(BOOTENS){
  #     dB = band_Ve_B_S1[ix,,]
  #     dB_ = crossprod(UU, dB) # d  in the UU-column space
  #     yB_ = dB_ / sval                 # y  in the VV-column space
  #     yB = VV %*% yB_
  #     b_svd_B[ix,,] = yB
  #   }
  #   # Filter
  #   y_f_ = y_
  #   if(nSV_discard > 0) y_f_[(K - nSV_discard +1):K]=0 # the simplest flt
  #   y_f = drop(VV %*% y_f_)     # back to the physical space
  #   b_svd[ix,] = y_f 
  # }
 
  #-----------------------------------------------------------------------
  # Diag plots
  
  if(lplot & true_spectra_available){
    # Check whether b[i_n] can be reasonably well represented
    # by the basis of columns of VV
    # (if not, then the bands sh.be redefined)
    
    ix=sample(1:nx,1)
    
    y_true = b_true[ix,]
    y_true_ = drop(crossprod(VV, y_true))
    y_hat = drop(VV %*% y_true_)
    max(abs(y_hat - y_true)) / max(abs(y_true))
    # --> rel.err ~ 1e-2 for single-bands and 1e-3 for double (good) ==>
    #     V-space seems to be "enough" to represent y=b
    nm=nmax/3
    plot(y_true[1:nm])
    lines(y_hat[1:nm])
    
    # Check in the VV-column space
    mn=min(y_[,ix], y_true_) #;  if(BOOTENS) mn=min(mn, yB_)
    mx=max(y_[,ix], y_true_) #;  if(BOOTENS) mx=max(mx, yB_)
    plot(y_[,ix], type="p", ylim=c(mn,mx))
    lines(y_true_, lwd=2)
    # if(BOOTENS){
    #   for(iB in 1:nB){
    #     lines(yB_[,iB], col=rgb(red=0, green=1, blue=0, alpha=0.5),
    #           lwd=0.5, lty=3)
    #   }
    # }
    
    ix=sample(1:nx,1)
    nm=nmax/1
    mn=min(b_svd[ix, 1:nm], b_true[ix, 1:nm])
    mx=max(b_svd[ix, 1:nm], b_true[ix, 1:nm])
    plot(b_svd[ix, 1:nm], type="l", main="b_svd (red),  b_true", ylim=c(mn,mx), col="red")
    lines(b_true[ix, 1:nm])
    # lines(y[1:nm], col="red")
    abline(h=0)
    
    
    
    b_svd_Ms = apply(b_svd, 2, mean)
    inm=nmax/1
    mx=max(b_svd_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_svd_Ms[1:inm], main="b_svd_Ms(red), b_true_Ms", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
    lines(b_true_Ms[1:inm], lwd=2)

  }
  
  #-----------------------------------------------------------------------
  # Calc the resulting  b_fit: fit  b_shape
  # f(n) = A*g(n/a) 
  
  nn=c(0:nmax)
  
  FIT = fitScaleMagn(t(b_svd[,1:nmaxp1]), b_shape, nn, 
                     moments, a_max_times, w_a_fg, lplot)
  b_fit[,1:nmaxp1] = t(as.matrix(FIT$f_fit))
  AA = FIT$AA
  aa = FIT$aa
  
  b_fit[, (nmaxp1+1):nx] = b_fit[,rev(2:nmax)]
  
  
  sum(b_fit < 0) / sum(b_fit >= 0)
  
  if(true_spectra_available){
    norm(b_svd - b_true, type="F") / norm(b_true, type="F")
    norm(b_fit - b_true, type="F") / norm(b_true, type="F")
    
    # mean(abs(log(aa / aa_true)))
    # mean(abs(log(AA / AA_true)))
    
    # image2D(b_fit)
    # image2D(b_true)
  }
  
  #-----------------------------------------------------------------------
  #-----------------------------------------------------------------------
  # Diags
  
  if(lplot & true_spectra_available){
    b_svd_Ms = apply(b_svd, 2, mean)
    b_fit_Ms = apply(b_fit, 2, mean)
    
    inm=nx/6
    mx=max(b_svd_Ms[1:inm], b_fit_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_fit_Ms[1:inm], main="Ms: b_svd(blu), b_fit(red), b_tru", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
    lines(b_svd_Ms[1:inm], lwd=2, col="blue")
    lines(b_true_Ms[1:inm], lwd=2)
    
    
    plot(b_fit_Ms[1:inm]/b_fit_Ms[1], main="b_fit_Ms Nrmlzd(red), b_true_Ms", 
         type="l", col="red", lwd=2, ylim = c(0,1))
    lines(b_true_Ms[1:inm]/b_true_Ms[1], lwd=2)
    
    
    ix=sample(c(1:nx), 1)
    inm=nx/6
    plot(b_true[ix,1:inm]/b_true[ix,1], 
         main=paste0("b_true/b_true[1] (circ), b_fit/b_fit[1] \n ix=", ix),
         ylim=c(0,1), xlab="n+1")
    lines(b_fit[ix,1:inm]/b_fit[ix,1], col="red")
    
    
    ix=sample(c(1:nx), 1)
    bias = sum(b_fit[ix,] - b_true[ix,])
    AE_svd = sum(abs(b_svd[ix,] - b_true[ix,]))
    AE_fit = sum(abs(b_fit[ix,] - b_true[ix,]))
    
    inm=nx/6
    mn=min(b_true[ix,1:inm], b_fit[ix,1:inm], b_svd[ix,1:inm])
      # if(BOOTENS) mn=min(mn, b_fit_B[ix,1:inm,])
    mx=max(b_true[ix,1:inm], b_fit[ix,1:inm], b_svd[ix,1:inm])
      # if(BOOTENS) mx=max(mx, b_fit_B[ix,1:inm,])
    plot(b_true[ix,1:inm], 
         main=paste0("b_true(blck), b_svd(gold), b_fit(red)\n ix=", ix,
                     " bias=", signif(bias,2), "\nAE_svd=", signif(AE_svd,2), 
                     " AE_fit=", signif(AE_fit,2)), type="l",
         ylim=c(mn,mx), xlab="n+1", lwd=2)
    lines(b_svd[ix,1:inm], col="gold", lwd=1)
    lines(b_fit[ix,1:inm], col="red", lwd=2)
    lines(x=band_centers_n, y=band_V[ix,]/10)
    
    # if(BOOTENS){
    #   for(iB in 1:nB){
    #     lines(b_fit_B[ix,1:inm,iB], col=rgb(red=0, green=1, blue=0, alpha=0.5),
    #           lwd=0.5, lty=3)
    #   }
    # }
    abline(h=0)
    
    
    # scatterplot AE vs. gamma
    
    rAAE_fit = apply(b_fit - b_true, 1, function(t) sum(abs(t))) / 
           apply(b_true,         1, function(t) sum(abs(t)))
    ymx=max(rAAE_fit)
    plot(x=gamma[1:nx], y=rAAE_fit, ylim=c(0,ymx),
         main=paste0("rAAE_fit. gamma_med=", gamma_med))
    
    AL = lowess(x=gamma[1:nx], y=rAAE_fit)
    ggamma = AL$x
    rAAE_fit_smoo = AL$y
    plot(x=ggamma, y=rAAE_fit_smoo,  ylim=c(0,ymx),
         main=paste0("rel_AbsErr_smoo. gamma_med=", gamma_med))
    
    # ==> The higher gamma, the better fit (somewhat)!!
    #-------------------------
    # The worst-fit point
    
    ix_worst = which(rAAE_fit==max(rAAE_fit), arr.ind = T)
    ix=ix_worst
    
    inm=nmaxp1
    AE_svd = sum(abs(b_svd[ix,] - b_true[ix,]))
    AE_fit = sum(abs(b_fit[ix,] - b_true[ix,]))
    mn=min(b_true[ix,1:inm], b_fit[ix,1:inm], b_svd[ix,1:inm])
    # if(BOOTENS) mn=min(mn, b_fit_B[ix,1:inm,])
    mx=max(b_true[ix,1:inm], b_fit[ix,1:inm], b_svd[ix,1:inm])
    # if(BOOTENS) mx=max(mx, b_fit_B[ix,1:inm,])
    plot(b_true[ix,1:inm], 
         main=paste0("b_true(blck), b_svd(gold), b_fit(red)\n ix=", ix,
                     " bias=", signif(bias,2), "\nrel_AE_svd=", signif(AE_svd,2), 
                     "  rel_AE_fit=", signif(AE_fit,2)), type="l",
         ylim=c(mn,mx), xlab="n+1", lwd=2)
    lines(b_svd[ix,1:inm], col="gold", lwd=1)
    lines(b_fit[ix,1:inm], col="red", lwd=2)
    
    #-------------------------
    
    # d=20
    # j=4
    # mx = max(band_Vt[(ix-d): (ix+d), j], band_V[(ix-d): (ix+d), j])
    # plot(band_Vt[(ix-d):(ix+d), j], ylim=c(0,mx))
    # lines(band_V[(ix-d):(ix+d), j])
    # 
    # mx = max(b_true[(ix-d): (ix+d), j], b_fit[(ix-d): (ix+d), j])
    # plot(b_true[(ix-d):(ix+d), j], ylim=c(0,mx))
    # lines(b_fit[(ix-d):(ix+d), j])
  }
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by the estmted b_fit
  
  band_V_restored_K = t( apply( b_fit, 1, function(t) drop(Omega_S1 %*% t) ) )
  band_V_restored = band_V_restored_K[,1:J]
  
  # mx=max(band_V, band_V_restored)
  # image2D(band_V, main="band_V", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")
  
  norm(band_V_restored - band_V, "F") / norm(band_V, "F")
  
  if(lplot & true_spectra_available){
    ix=sample(1:nx,1)
    mx=max(band_V[ix,], band_V_restored[ix,], band_Vt[ix,])
    plot(band_V[ix,], main="V_e (circ), V_fit(red), V_tru(black)", ylim=c(0,mx),
         xlab = "band", ylab = "Band variances",
         sub=paste0("ix=", ix))
    lines(band_V_restored[ix,], col="red", lwd=2)
    lines(band_Vt[ix,], col="black", lwd=2)
    
  }

  #-----------------------------------------------------------------------  

  return(list("b_fit"=b_fit, # [ix, i_n] 
              "band_V_restored"=band_V_restored, # band variances restored from b_fit
              "AA"=AA, "aa"=aa,
              "rel_err_discard_nullspace_b"=rel_err_discard_nullspace_b,
              "b_svd"=b_svd)) # [ix, i_n] 
}
