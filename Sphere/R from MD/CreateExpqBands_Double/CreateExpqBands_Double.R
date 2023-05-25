
CreateExpqBands_Double = function(DoubleBands, nmax, J1, 
                                  halfwidth_min_wide, halfwidth_min_narr, 
                                  nc2_wide, nc2_narr, 
                                  halfwidth_max_wide, halfwidth_max_narr, 
                                  q_tranfu=3, rectang = FALSE){
  #-------------------------------------------------------------------------------------------
  # Specify a Double (or Single) set of Bands: wide & narrow. 
  # 
  #  Method:   
  # 
  # Call  CreateExpqBands  twice: for wide & for narrow bands, concatenate the results.
  # If   DoubleBands = F,  only the WIDE bands are created.  
  # 
  #    Args
  # 
  # DoubleBands - double or single set of bands?
  # nmax - [0,nmax] is the spectral range
  # J1 - nu of passbands in the Single set of bands
  # halfwidth_min_wide - minimal halfwidth for wide bands (integer)
  # halfwidth_min_narr - minimal halfwidth for narrow bands (integer)
  # nc2_wide - center wvn of band 2  for wide bands
  # nc2_narr - center wvn of band 2  for narrow bands 
  #  halfwidth_max_wide, halfwidth_max_narr, - maximal halfwidth (integer)  
  # q_tranfu - exponent in the parametric tranfu model:  
  #            tranfu=exp(-|(n-nc)/halfwidth|^q_tranfu)
  # rectang - if TRUE, tranfu is rectangular, =1 iff |n-nc| <= halfwidth
  #           
  #    Return:
  # 
  # tranfu[i_n=1:nx, j=1:J2] - spectral transfer functions for all PASSBANDS (J2=J*2)
  # respfu[i_n=1:nx, j=1:J2] - impulse response functions for all bands
  #   (cplx valued!)
  # hhwidth[j=1:J2] - half-widths of the bands' tranfu
  # band_centers_n[j=1:J2] - centers of passbands (nnc)
  # 
  # M Tsy 2021 Feb
  #-------------------------------------------------------------------------------------------
  
  source('CreateExpqBands.R')

  nx=2*nmax
  
  dx = 2*pi/nx
  Rekm = 6370
  dx_km = dx * Rekm
  xx_km=c(0:(nx-1)) * dx_km
  
  #-----------------------------------------------------------------
  # Specify bands' tranfu & respfu etc for both wide and narrow bands
  
  #---------------------------------
  # Wide bands are created anyway
  
  BANDS_wide = CreateExpqBands(nmax, J1, 
                               halfwidth_min_wide, nc2_wide, 
                               halfwidth_max_wide, 
                               q_tranfu, rectang)
  tranfu_wide = BANDS_wide$tranfu
  respfu_wide = BANDS_wide$respfu
  band_centers_n_wide  = BANDS_wide$band_centers_n
  hhwidth_wide = BANDS_wide$hhwidth
  
  if(!DoubleBands){ # single  set
    
    #---------------------------------
    # Just take the wide-bands results
    
    J=J1 ; nband=J
    tranfu = tranfu_wide
    respfu = respfu_wide
    band_centers_n = band_centers_n_wide
    hhwidth = hhwidth_wide
    
  }else{           # double set
    
    #---------------------------------
    # Generate narrow bands in addition
    
    BANDS_narr = CreateExpqBands(nmax, J1, 
                                 halfwidth_min_narr, nc2_narr, 
                                 halfwidth_max_narr, 
                                 q_tranfu, rectang)
    tranfu_narr = BANDS_narr$tranfu
    respfu_narr = BANDS_narr$respfu
    band_centers_n_narr  = BANDS_narr$band_centers_n
    hhwidth_narr = BANDS_narr$hhwidth
    
    
    #---------------------------------
    # Concatenate wide and narrow bands
    
    J=J1*2 ; nband=J
    tranfu = cbind(tranfu_wide, tranfu_narr)
    respfu = cbind(respfu_wide, respfu_narr)
    band_centers_n = c(band_centers_n_wide, band_centers_n_narr)
    hhwidth = c(hhwidth_wide, hhwidth_narr)
  }
  
  band_centers_n
  hhwidth
  
  #-----------------------------------------------------------------
  # Plots
  
  namefile=paste0("TranfuExpqDbl", abs(DoubleBands), "J",J, 
                  "rect", abs(rectang),  "q", q_tranfu,
                  "scw", nc2_wide,
                  "hwmnw", halfwidth_min_wide,
                  "hwmxw", halfwidth_max_wide,".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(tranfu[1:nx, 1], main=paste0("Exp(-(n-nc)^q)  bands: transfer functions"), 
       xlab="Wavenumber", ylab="Spectral transfer function", ylim=c(0,1), type="l")
  if(J > 1){
    for (band in 2:J){
      lines(tranfu[1:nx, band])
    }
  }
  dev.off()
  
  respfu_re = Re(respfu) # NB: respfu is CPLX for non-symm bands!
  respfu_im = Im(respfu) 
  
  ixmax = nx/6
  
  mn = min(respfu_re[1:ixmax,])
  mx = max(respfu_re[1:ixmax,])
  namefile=paste0("Re_Resp_ExpqDbl", abs(DoubleBands), "J",J, 
                  "rect", abs(rectang),  "q", q_tranfu,
                  "scw", nc2_wide,
                  "hwmnw", halfwidth_min_wide,
                  "hwmxw", halfwidth_max_wide,".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(x=xx_km[1:ixmax], y=respfu_re[1:ixmax, 1], ylim=c(mn,mx),
       main=paste0("Exp(-(n-nc)^q) bands: Re(response functions)"), 
       xlab="Distance, km", ylab="Re(Impulse response function)", type="l")
  if(J > 1){
    for (band in 2:J){
      lines(x=xx_km[1:ixmax], y=respfu_re[1:ixmax, band])
    }
  }
  abline(h=0)
  dev.off()
  
  mn = min(respfu_im[1:ixmax,])
  mx = max(respfu_im[1:ixmax,])
  namefile=paste0("Im_Resp_ExpqDbl", abs(DoubleBands), "J",J, 
                  "rect", abs(rectang),  "q", q_tranfu,
                  "scw", nc2_wide,
                  "hwmnw", halfwidth_min_wide,
                  "hwmxw", halfwidth_max_wide,".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(x=xx_km[1:ixmax], y=respfu_im[1:ixmax, 1], ylim=c(mn,mx),
       main=paste0("Exp(-(n-nc)^q) bands: Im(response functions)"), 
       xlab="Distance, km", ylab="Im(Impulse response function)", type="l")
  if(J > 1){
    for (band in 2:J){
      lines(x=xx_km[1:ixmax], y=respfu_im[1:ixmax, band])
    }
  }
  abline(h=0)
  dev.off()
  
  #-----------------------------------------------------------------
  
  return(list(
    "tranfu" = tranfu,     # spe transfer functions
    "respfu" = respfu,     # impulse response functions
    "hhwidth" = hhwidth,               # half-widths of the bands' tranfu
    "band_centers_n" = band_centers_n))  # centers of passbands 

}
