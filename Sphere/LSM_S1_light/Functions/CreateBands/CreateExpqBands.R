
CreateExpqBands = function(nmax, nband, halfwidth_min, nc2, halfwidth_max, 
                              q_tranfu=3, rectang = FALSE){
  #-------------------------------------------------------------------------------------------
  # Bands specification.
  # Create J=nband pass bands of varying width, 
  # each with the center-band wavenumber nnc[band] ranging from 0 to nmax 
  # and specified in this function.
  # 
  # NB: On S1: Allow for non-even tranfu(n) so that the filtered signal will be cplx valued!
  #            This allows better resolution at small wvns.
  #     
  #     On S2: (1) From tranfu, take the first extent  i_n=n+1  only within the range [1, nmax+1].
  #            (2) Disregard respfu. Calculate  respfu  using Fourier-Legendre transform.
  #  
  #  Method:   
  #  
  # Call  tranfuBandpassExpq  nband times for a number of growing center-band wvn, nc. 
  #  
  # 
  # (1) the extreme bands.
  #    nnc[1] = 0,        hhwidth[1]    =halfwidth_min
  #    nnc[nband]=nmax,   hhwidth[nband]=halfwidth_max
  # 
  # (2) 2nd band: 
  #    nnc[2] = nc2,      hhwidth[2]    =halfwidth_min
  #    
  # Between the 2nd and the last bands (nc=0 & nc=nmax, resp.), 
  # specify a smooth transition, with growing both nc & halfwidth uniformly in log(n). 
  # ---
  # Specifically, for nnc,
  # let the spacings Delta_j between the center points of the 
  # adjacent bands be growing geometrically:
  # 
  # Delta1 = nc2 - 0 = nc2
  # Delta_2 = mu*Delta1
  # ...
  # Delta_{J-1} = mu^(J-2) Delta1
  # 
  # so that
  # sum Delta_j = nmax
  # sum Delta_j = Delta1 * (1 + mu + ... + mu^(J-2)) = nmax
  # 
  # We find mu by solving the eqn
  # 
  #     phi(mu) = nmax / Delta1,             (1)
  #     
  # whr 
  #  
  #     phi(mu) = sum_{j=1}^{j=J-2} mu^j     (2)
  #  
  #  (we avoid using the explicit formula for this sum since it has a singularity at mu=1).
  # We employ the simple bisection method to solve (2) since phi(mu) is monotone.
  #---
  #  As for hhwidth[], we assume they grow exponentially:
  #  
  #  hw[1] = halfwidth_min
  #  hw[2] = hw[1] * mu_hw
  #  ...
  #     hw[J] = halfwidth_min * mu_hw^(J-1)  = halfwidth_max   (3)
  #  
  # From the last eqn,
  # mu_hw^(J-1) = halfwidth_max / halfwidth_min
  # (J-1) log(mu_hw) = log(halfwidth_max / halfwidth_min)
  # 
  #   mu_hw = (halfwidth_max / halfwidth_min)^(1/(J-1))
  #-------------------------  
  #    Args
  # 
  # nmax - [0,nmax] is the spectral range
  # nband - nu of passbands
  # halfwidth_min - minimal halfwidth (integer)
  # nc2 - center wvn of band 2
  # halfwidth_max - maximal halfwidth (integer)
  # q_tranfu - exponent in the parametric tranfu model:  
  #            tranfu=exp(-|(n-nc)/halfwidth|^q_tranfu)
  # rectang - if TRUE, tranfu is rectangular, =1 iff |n-nc| <= halfwidth
  #           
  #    Return:
  # 
  # tranfu[i_n=1:nx, band=1:nband] - spectral transfer functions for all PASSBANDS 
  # respfu[i_n=1:nx, band=1:nband] - impulse response functions for all bands
  #   (cplx valued!)
  # hhwidth[band=1:nband] - half-widths of the bands' tranfu
  # band_centers_n[band=1:nband] - centers of passbands (nnc)
  # 
  # Required functions: 
  # bisection, tranfuBandpassExpq
  # 
  # M Tsy 2020 Jun
  #-------------------------------------------------------------------------------------------

  # Prelims
  
  nmaxp1=nmax +1
  nx=2*nmax
  dx = 2*pi/nx
  Rekm = 6370
  dx_km = dx * Rekm
  xx_km=c(0:(nx-1)) * dx_km
  
  J = nband
  
  nnc=c(1:nband) # init
  nncp1=nnc # init
  hhwidth = c(1:nband) # init
  
  tranfu = matrix(0, nrow=nx, ncol=nband)
  respfu = matrix(0, nrow=nx, ncol=nband)
  
  # (1) Specify nnc[] and hhwidth[] 
  
  #------------------------------
  # The two extreme bands.
  # nncp1[1] = 1,          hhwidth[1]    =halfwidth_min
  # nncp1[nband]=nmaxp1,   hhwidth[nband]=halfwidth_max
  
  band=1
  nncp1[band] = 1
  hhwidth[band] = halfwidth_min
  
  band=nband
  nncp1[band] = nmaxp1
  hhwidth[band] = halfwidth_max
  
  #-------------------------------
  # (2) 2nd band
  # nnc[2] = halfwidth_min /2 (or/3)
  
  band=2
  nncp1[band] = nc2 +1
  
  #-------------------------------
  # The other bands.

  #-----------------
  # (1) Bands centers:
  #     Solve
  #      
  # phi(mu) = nmax / Delta1
  # phi(mu) = sum_{j=1}^{j=J-2} mu^j
 
  Delta1 = nc2
  lambda = nmax / Delta1
  
  mu_nc = bisection(0.5, 10, 1e-6, function(mu) sum( mu^c(0:(J-2))  ) - lambda)
  # mu_nc
  # sum( mu_nc^c(0:(nband-2))  ) - lambda
  
  #-----------------
  # halfwidths
  # mu_hw = (halfwidth_max / halfwidth_min)^(1/(J-1))
  
  mu_hw = (halfwidth_max / halfwidth_min)^(1/(J-1))
  hhwidth[2] = halfwidth_min * mu_hw

  #-----------------
  # Calc nnc & hhw
  
  hw_prev = hhwidth[2]
  Delta_prev = Delta1
  
  for(band in 3:(nband-1)){
    Delta = Delta_prev * mu_nc
    nncp1[band] = nncp1[band-1] +  Delta
    Delta_prev = Delta

    hw = hw_prev * mu_hw
    hhwidth[band] = hw
    hw_prev = hw  
  }

  nnc[] = nncp1[] -1
  
  nnc
  hhwidth
  
  #-----------------------------------------------------------------
  #-----------------------------------------------------------------
  # Specify bands' tranfu & respfu
  
  band_centers_n = nnc
  
  for (band in 1:nband){
    
    nc = nnc[band]
    halfwidth = hhwidth[band] 
    
    BAND = tranfuBandpassExpq(nmax, nc, halfwidth, 
                              q_tranfu=q_tranfu, rectang = rectang)
    tranfu[,band] = BAND $tranfu
    respfu[,band] = BAND $respfu
    if(band == 1 | band == nband) respfu[,band] =    # tranfu is an even fu of n here
      complex(real = Re(respfu[,band]), imaginary = 0)
    
    #plot(tranfu[,band], type="l", main=paste0("tranfu. Band=", band, " band center ", nc))
    #plot(Re(respfu[,band]), type="l", main=paste0("Re(respfu). Band=", band, " band center ", nc))
    #plot(Im(respfu[,band]), type="l", main=paste0("Im(respfu). Band=", band, " band center ", nc))
    #plot(abs(respfu[,band]), type="l", main=paste0("abs(respfu). Band=", band, " band center ", nc),
    #    xlab = "Distance, mesh sizes", ylab = "respfu")
  }
  
  namefile=paste0("./Out/TranfuExpq_J",nband, "rect", abs(rectang),  "q", q_tranfu,
                  "sc", signif(nc2,2), "hwmn", signif(halfwidth_min,2), "hwmx", signif(halfwidth_max,2),".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(tranfu[1:nx, 1], main=paste0("Exp(-(n-nc)^q)  bands: transfer functions"), 
       xlab="Wavenumber", ylab="Spectral transfer function", ylim=c(0,1), type="l")
  if(nband > 1){
    for (band in 2:nband){
      lines(tranfu[1:nx, band])
    }
  }
  dev.off()
  
  respfu_re = Re(respfu) # NB: respfu is CPLX for non-symm bands!
  respfu_im = Im(respfu) 
  
  ixmax = nx/6
  
  mn = min(respfu_re[1:ixmax,])
  mx = max(respfu_re[1:ixmax,])
  namefile=paste0("./Out/Re_Resp_Expq_J",nband, "rect", abs(rectang),  "q", q_tranfu,
                  "sc", signif(nc2,2), "hwmn", signif(halfwidth_min,2), "hwmx", signif(halfwidth_max,2),".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(x=xx_km[1:ixmax], y=respfu_re[1:ixmax, 1], ylim=c(mn,mx),
       main=paste0("Exp(-(n-nc)^q) bands: Re(response functions)"), 
       xlab="Distance, km", ylab="Re(Impulse response function)", type="l")
  if(nband > 1){
    for (band in 2:nband){
      lines(x=xx_km[1:ixmax], y=respfu_re[1:ixmax, band])
    }
  }
  abline(h=0)
  dev.off()
  
  mn = min(respfu_im[1:ixmax,])
  mx = max(respfu_im[1:ixmax,])
  namefile=paste0("./Out/Im_Resp_Expq_J",nband, "rect", abs(rectang), "q", q_tranfu,
                  "sc", signif(nc2,2), "hwmn", signif(halfwidth_min,2), "hwmx", signif(halfwidth_max,2),".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(x=xx_km[1:ixmax], y=respfu_im[1:ixmax, 1], ylim=c(mn,mx),
       main=paste0("Exp(-(n-nc)^q) bands: Im(response functions)"), 
       xlab="Distance, km", ylab="Im(Impulse response function)", type="l")
  if(nband > 1){
    for (band in 2:nband){
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
