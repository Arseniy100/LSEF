

band_lbmean = function(band_V, tranfu2, lplot){
  #-------------------------------------------------------------------------------------------
  # Having the band variances, band_V,  
  # calc bands' mean spectral values,  bmean.
  # Also calc band-mean wvns,  lmean.
  # To do so, we need the bands' spectral transfer functions squared,  tranfu2.
  # NB: it appears that for each band,  bmean[j]  is proportional to  band_V[j],
  # with the proportionality constant being dependent only on  tranfu_j  and not on the
  # spectrum. Therefore, we may compute  bmean  from band variances only
  # (not using the spectrum).
  # 
  #    Method
  #    
  #    w_j[n] = |tranfu_j[n]|^2,
  # whr n runs the whole circle (0:()),
  # we have the band variances:   
  #    
  #    v_j = sum_n  w_j[n]*b[n] 
  #
  # Now, switch to the half-circle of wvns, l \in [0, nmax]
  # because the local spectrum is assumed to be an EVEN fu of wvn.
  #
  #    v_j = w_j[0] b[0] +  w_j[lmax] b[lmax] +  
  #          sum_{l=1}^{lmax-1} (w_j[l] + w_j[-l]) * b[l] = 
  #          c_j * sum_{l=0}^lmax  p_j[l]*b[l].
  #
  # whr 
  # p_j[0] = w_j[0] / c_j
  # p_j[l] = (w_j[l] + w_j[-l]) / c_j    for l \in (1, lmax-1)
  # p_j[lmax] = w_j[lmax] / c_j
  # 
  # c_j = sum( w_j[n] )
  # ...............................................
  # Thus,  v_j = c_j * sum_{l=0}^lmax p_j[l]*b[l].
  # ............................................... 
  #  
  # The band-j MEAN b can be defined as
  # 
  #    band_bmean[j] := sum_{l=0}^lmax  p_j[l]*b[l]  == v_j / c_j
  #
  # We compute band_bmean[j] and assign it to the band-mean wvns
  # ...........................................
  #    l_mean[j] = sum_{l=0}^lmax  p_j[l] * l, 
  # ...........................................
  # thus getting J values of b at l_mean[j=1:J].
  # Finally, we draw straight lines through those J points  (l_mean[j], band_bmean[j])
  #
  #    Args
  # 
  # band_V[1:nx, 1:nband] - [ix, band] BAND variances 
  #               (normally, estimated from the ensemble or may be the true band varinces) 
  #                           at the grid point x
  # tranfu2[1:nx, 1:nband] - |tranfu|^2 for [i_n, band], i_n=1,...,nx,  i_n=n+1
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: band_bmean[1:nx, 1:nband] (band-mean spectra)
  #         band_lmean[1:nband] (band-mean wvns)
  # 
  # M Tsy 2022 Aug
  #-------------------------------------------------------------------------------------------
  
  nx    = dim(tranfu2)[1]
  nband = dim(tranfu2)[2] ;  J=nband
  
  nmax = nx /2 ;      lmax=nmax
  nmaxp1 = nmax +1 ;  lmaxp1=nmaxp1
  
  band_bmean = matrix(0, nrow = nx, ncol = nband)
  
  #-----------------------------------------------------------------------  
  # Calc c_j

  cj = apply(tranfu2, 2, sum)
  
  #-----------------------------------------------------------------------  
  # Calc p_j[l]
  # 
  # p_j[0] = w_j[0] / c_j
  # p_j[l] = (w_j[l] + w_j[-l]) / c_j    for l \in (1, lmax-1)
  # p_j[lmax] = w_j[lmax] / c_j
  
  pjl = matrix(0, nrow = J, ncol = lmaxp1)
  i_ll = c(2:lmax)
  
  for (j in 1:J){
    pjl[j,1] = tranfu2[1,j] /cj[j]
    pjl[j,lmaxp1] = tranfu2[lmaxp1,j] /cj[j]
    
    # i_n = n+1
    # i_n_symm = 2*nmaxp1 - i_n
    
    pjl[j, i_ll] = (tranfu2[i_ll,j] + tranfu2[2*lmaxp1 - i_ll, j]) / cj[j]
  }
  
  # apply(pjl,1,sum)
  # j=1
  # plot(tranfu2[,j])
  # plot(pjl[j,])
  #-----------------------------------------------------------------------  
  # from v[j]  to  band_bmean[j]

  band_bmean = t( apply(band_V, 1, function(t) t / cj) )
  
  # image2D(band_V, main="band_V")
  # image2D(band_bmean, main="band_bmean")
  # plot(apply(band_bmean, 2, mean), main="band_bmean_Ms")
  #-----------------------------------------------------------------------  
  # band-mean wvns
  # band_lmean[j] = sum_{l=0}^lmax  p_j[l] * l
  
  i_ll_full = c(1:lmaxp1)
  lp1_mean = apply(pjl, 1, function(t) sum(t*i_ll_full))
  
  band_lmean = lp1_mean -1
  # band_lmean
  #-----------------------------------------------------------------------  

  return(list("band_bmean"=band_bmean,  # [1:nx, 1:nband] (band-mean spectra)
              "band_lmean"=band_lmean)) # [1:nband] (band-mean wvns)
}