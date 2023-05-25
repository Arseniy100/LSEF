
B2S_NN = function(band_V, NN, Omega, truth_available, lplot){
  #------------------------------------------------------------------------------
  #  Bands To Spectrum.
  # Fit the spectrum  f  (variable b_fit[ix,i_n])  to the set of band variances  
  #    d=band_V[ix=1:nx,1:J]   
  # Perform at each ix independently.
  #
  #    Methodology
  #    
  # Use Neural Netw (previously trained)
  # 
  # NB: Omega  is the Forward  Model: f --> d
  #     NN     is the Inverse  Model: d --> f 
  # 
  #    Args
  # 
  # band_V[ix=1:nx, j=1:J] - BAND variances at the grid point  ix
  #   (normally, estimated from the ensemble or may be the true band variances) 
  # NN - neural network (a derived data type variable)
  # Omega[j=1:J, i_n=1:(nmax+1)]  (whr i_n=n+1) - the Obs mx:
  #    d =  Omega * f + err    (whr  f  is the spectrum)
  # truth_available - TRUE if there is TRUTH available for testing
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  #         
  # Required functions:
  #
  #   
  # A Sotskiy, M Tsy 2022 May
  #-------------------------------------------------------------------------------------------
  
  nx = dim(band_V)[1]  # spatial grid size
  J  = dim(band_V)[2]  # nu of spectral bands

  nmax=nx/2            # max resolvable wvn
  
  b_fit = matrix(0, nrow = nx, ncol = nx) # [ix,i_n]
  
  #-----------------------------------------------------------------------
  # Apply NN, getting  b_fit
  
  
  
  
  #-----------------------------------------------------------------------
  # Diags
  
  if(truth_available){
    norm(b_fit - b_true, type="F") / norm(b_true, type="F")
  }
  
  if(lplot & truth_available){
    # image2D(b_fit)
    # image2D(b_true)
    
    b_fit_Ms = apply(b_fit, 2, mean)
    
    inm=nx/6
    mx=max(b_fit_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_fit_Ms[1:inm], main="Ms: b_svd(blu), b_fit(red), b_tru", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
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
  }
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by the estmted  b_fit
  # (apply the forward model to  b_fit)
  
  band_V_restored = t( apply( b_fit, 1, function(t) drop(Omega %*% t) ) )

  # mx=max(band_V, band_V_restored)
  # image2D(band_V, main="band_V", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")
  
  norm(band_V_restored - band_V, "F") / norm(band_V, "F")
  
  if(lplot & truth_available){
    ix=sample(1:nx,1)
    mx=max(band_V[ix,], band_V_restored[ix,], band_Vt[ix,])
    plot(band_V[ix,], main="V_e (circ), V_fit(red), V_tru(black)", ylim=c(0,mx),
         xlab = "band", ylab = "Band variances",
         sub=paste0("ix=", ix))
    lines(band_V_restored[ix,], col="red", lwd=2)
    lines(band_Vt[ix,], col="black", lwd=2)
    
  }
  #-----------------------------------------------------------------------  

  return(list("b_fit"=b_fit,                     # the fitted spectrum
              "band_V_restored"=band_V_restored)) # band variances restored from b_fit
}
