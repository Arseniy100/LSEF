
B2S_NN = function(band_Ve, NN, 
                  TransformSpectrum_type_x, TransformSpectrum_type_y,
                  TransformSpectrum_pow_x, TransformSpectrum_pow_y,
                  Omega_nmax, true_spectra_available, lplot){
  #------------------------------------------------------------------------------
  #  Bands To Spectrum.
  # Fit the spectrum  f  (variable b_fit[ix,i_n])  to the set of (ensm) band variances  
  #    d = band_Ve[ix=1:nx,1:J]   
  #    
  # Perform at each ix independently.
  #
  #    Methodology
  #    
  # Use Neural Netw (previously trained)
  # 
  # NB: Omega_nmax  is the Forward  Model: f --> d
  #     NN     is the Inverse  Model: d --> f 
  # 
  #    Args
  # 
  # band_Ve[ix=1:nx, j=1:J] - ensm BAND variances at the grid point  ix
  #   (normally, estimated from the ensemble or may be the true band variances) 
  # NN - neural network (a derived data type variable)
  # TransformSpectrum_type_x, TransformSpectrum_type_y - functional transformation type
  #   of NN's input and output data 
  # TransformSpectrum_pow_x, TransformSpectrum_pow_y - exponents in the power-law transforms 
  # of in and out NN data, resp.
  # Omega_nmax[j=1:J, i_n=1:(nmax+1)]  (whr i_n=n+1) - the Obs mx:
  #    d =  Omega_nmax * f + err    (whr  f  is the spectrum)
  # true_spectra_available - TRUE spectra and true band variances are provided testing
  #    (if TRUE,  b_true, b_true_Ms  and  band_Vt  are taken from the environment!)
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  #         
  # Required functions:
  #
  #   
  # M Tsy 2022 May
  #-------------------------------------------------------------------------------------------
  
  nx = dim(band_Ve)[1]  # spatial grid size
  J  = dim(band_Ve)[2]  # nu of spectral bands

  nmax=nx/2            # max resolvable wvn
  nmaxp1 = nmax +1 ;  nmaxp2 = nmax +2
  
  b_fit = matrix(0, nrow = nx, ncol = nx) # [ix,i_n]
  
  #-----------------------------------------------------------------------
  # Apply NN, getting  b_fit
  # First, transform the input data
  
  x_test = TransformSpectrum(band_Ve, type=TransformSpectrum_type_x, pow=TransformSpectrum_pow_x)
  xx = torch_tensor(x_test, dtype = torch_float())
  
  #==============
  # Apply NN
  yy = NN(xx)
  #==============
  
  # Transform the NN output back
  y_test = as.array(yy)
  b_fit[,1:nmaxp1] = TransformSpectrum(y_test, type=TransformSpectrum_type_y, 
                                       pow=TransformSpectrum_pow_y, inverse = T)
  b_fit[b_fit<0]=0
  
  # add "symmetric" wvns in  b_fit
  b_fit[,nmaxp2:nx] = b_fit[,rev(2:nmax)]

  #-----------------------------------------------------------------------
  # Diags
  
  if(true_spectra_available){
    norm(b_fit - b_true, type="F") / norm(b_true, type="F")
  }
  
  if(lplot & true_spectra_available){
    mx=max(b_fit, b_true)
    image2D(b_fit, zlim=c(0,mx))
    image2D(b_true, zlim=c(0,mx))
    
    b_fit_Ms = apply(b_fit, 2, mean)
    
    inm=nx/6
    mx=max(b_fit_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_fit_Ms[1:inm], main="Ms:  b_fit(red), b_tru", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
    lines(b_true_Ms[1:inm], lwd=2)
    
    plot(b_fit_Ms[1:inm]/b_fit_Ms[1], main="b_fit_Ms Nrmlzd(red), b_true_Ms", 
         type="l", col="red", lwd=2, ylim = c(0,1))
    lines(b_true_Ms[1:inm]/b_true_Ms[1], lwd=2)
    
    
    ix=sample(c(1:nx), 1)
    inm=nx/6
    mx=max(b_fit[ix, 1:inm], b_true[ix, 1:inm])
    plot(b_true[ix,1:inm], 
         main=paste0("b_true (circ), b_fit \n ix=", ix),
         ylim=c(0,mx), xlab="n+1")
    lines(b_fit[ix,1:inm], col="red")
    
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
  
  band_V_restored = t( apply( b_fit[,1:nmaxp1], 1, function(t) drop(Omega_nmax %*% t) ) )

  # mx=max(band_Ve, band_V_restored)
  # image2D(band_Ve, main="band_Ve", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")
  
  norm(band_V_restored - band_Ve, "F") / norm(band_Ve, "F")
  
  if(lplot & true_spectra_available){
    ix=sample(1:nx,1)
    mx=max(band_Ve[ix,], band_V_restored[ix,], band_Vt[ix,])
    plot(band_Ve[ix,], main="V_e (circ), V_fit(red), V_tru(black)", ylim=c(0,mx),
         xlab = "band", ylab = "Band variances",
         sub=paste0("ix=", ix))
    lines(band_V_restored[ix,], col="red", lwd=2)
    lines(band_Vt[ix,], col="black", lwd=2)
    
  }
  #-----------------------------------------------------------------------  

  return(list("b_fit"=b_fit,                     # the fitted spectrum
              "band_V_restored"=band_V_restored)) # band variances restored from b_fit
}
