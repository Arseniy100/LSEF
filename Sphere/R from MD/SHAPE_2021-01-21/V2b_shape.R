

V2b_shape = function(b_tofit, b_shape, tranfu2, band_Ve, moments, lplot){
  #-------------------------------------------------------------------------------------------
  # Fit the following prm mdl to b_tofit[n] 
  # 
  #    b_tofit[n] \approx A* b_shape(n/a)\equiv A*g(n/a)               (*)
  #
  #    Methodology
  #    
  # The method of moments is used to retrieve A,a from (*):
  # Assuming, for simplicity of derivation, that n is a continuous variable z,
  # we have the 0th and 1st moment eqs:
  # 
  #  I= \int b_tofit(z) dz   = A \int g(z/a) dz = Aa \int g(t) dt = Aa*G
  # 
  #  I1= \int z b_tofit(z) dz = A \int z g(z/a) dz = Aa^2 \int t g(t) dt = Aa^2*G1
  # 
  # we can also take the 2nd moment:
  # 
  # I2= \int z^2 b_tofit(z) dz = A \int z^2 g(z/a) dz = Aa^3 \int t^2 g(t) dt = Aa^3*G2
  # 
  # See function fitScaleMagnFu  for details.
  # 
  # To compute G,G1,G2,I,I1,I2 we replace the integrals by the sums.
  #     
  # NB: n >= 0 everywhere in the FITTING process, z \in [0, (nx/2)].
  # 
  #    Args
  # 
  # b_tofit[ix=1:nx, i_n=1:nx], i_n=1,...,nx,  i_n=n+1 - spectrum to be fitted
  # b_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*b_shape(n/a)
  # tranfu2[1:nx, 1:nband] - |tranfu|^2 for [i_n, band], i_n=1,...,nx,  i_n=n+1
  # band_Ve[1:nx, 1:nband] - [ix, band] BAND variances 
  #               (normally, estimated from the ensemble or may be the true band varinces) 
  #                           at the grid point x
  # moments - which moments to equate: "01" or "12" 
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  # 
  # M Tsy 2020 Sep
  #-------------------------------------------------------------------------------------------
  
  nx    = dim(b_tofit)[1]
  nband = dim(tranfu2)[2]

  nmax = nx /2
  nmaxp1 = nmax +1
  
  nn=c(0:nmax)
  xx=nn
  
  b_fit = b_tofit # init

  #-----------------------------------------------------------------------
  # Calc the resulting b_fit
  # f(n) = A*g(n/a) 

  FIT = fitScaleMagn(t(b_tofit[,1:nmaxp1]), b_shape, xx, moments, lplot)
  b_fit[,1:nmaxp1] = t(as.matrix(FIT$f_fit))
  b_Magnitudes = FIT$Magnitudes
  b_Scales = FIT$Scales
  
  b_fit[, (nmaxp1+1):nx] = b_fit[,rev(2:nmax)]
  
  #-----------------------------------------------------------------------
  # Diags
  
  if(lplot){
    inm=nx/2
    ix=sample(c(1:nx), 1)
    mx=max(b_tofit[ix,1:inm], b_fit[ix,1:inm], b_true[ix,1:inm] )
    plot(b_tofit[ix,1:inm], main=paste0("b_tofit[ix,] (blue), b_true (black) 
         b_fit (red), b_mean_j (circles)
         ix=", ix), ylim=c(0,mx),
         type="l", col="blue", xlab="n+1")
    lines(b_true[ix,1:inm])
    lines(b_fit[ix,1:inm], col="red")
    
    # image2D(b_tofit, main="b_tofit")
    # image2D(b_fit, main="b_fit")
    # image2D(b_true, main="b_true")
  }

  norm(b_fit - b_true, "F") / norm(b_true, "F")
  mean(b_fit - b_true) / mean(b_true)
  
  # uu = abs(b_fit - b_true) - abs(b_tofit - b_true)
  # ix=which(uu == max(uu), arr.ind = T)[1]
  
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by the estmted b_LSM
  # For any ix, and any band j
  # V_j_restored = sum_i_n( tranfu2[,j] * b_fit[ix,] )
  
  band_V_restored = matrix(nrow = nx, ncol = nband) # [ix, band]

  for(ix in 1:nx){
    band_V_restored[ix,] = apply( tranfu2, 2, function(t) sum(t * b_fit[ix,]) )
  }
  
  # mx=max(band_Ve, band_V_restored)
  # image2D(band_Ve, main="band_Ve", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")

  norm(band_V_restored - band_Ve, "F") / norm(band_Ve, "F")
  
  #-----------------------------------------------------------------------  

  return(list("b_fit"=b_fit,
              "band_V_restored"=band_V_restored, # band variances restored from b_fit
              "b_Magnitudes"=b_Magnitudes, # AA
              "b_Scales"=b_Scales))         # aa
}
