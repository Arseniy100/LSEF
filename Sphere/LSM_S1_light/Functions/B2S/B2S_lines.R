

B2S_lines = function(tranfu2, band_V, ne, V2b_bbc, 
                     niter_bbc, correc_bbc, lplot){
  #-------------------------------------------------------------------------------------------
  # LINES: Bands To Spectrum
  # Estm b_n(x) from the Band Variances band_V[x, 1:nband] 
  #  (for each x independently)
  # knowing the bands' spectral transfer functions squared,  tranfu2.
  # 
  #    Method
  # 
  # Compute b_mean[j], l_mean[j] foir each band j, and then draw Lines between
  # those points on the (l, b_l) plane.
  # 
  # Estm the loc. spectrum as follows.
  # Perform for each x independently.
  # 
  # With 
  #    w_j[n] = |tranfu_j[n]|^2,
  # whr n runs the whole circle,
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
  #    b_mean[j] := sum_{l=0}^lmax  p_j[l]*b[l]  == v_j / c_j
  #
  # We compute b_mean[j] and assign it to the band-mean wvns
  # ...........................................
  #    l_mean[j] = sum_{l=0}^lmax  p_j[l] * l, 
  # ...........................................
  # thus getting J values of b at l_mean[j=1:J].
  # Finally, we draw straight lines through those J points  (l_mean[j], b_mean[j])
  #
  #    Args
  # 
  # tranfu2[1:nx, 1:nband] - |tranfu|^2 for [i_n, band], i_n=1,...,nx,  i_n=n+1
  # band_V[1:nx, 1:nband] - [ix, band] BAND variances 
  #               (normally, estimated from the ensemble or may be the true band varinces) 
  #                           at the grid point x
  # ne - ensm size with which the Band Vars have been computed
  # V2b_bbc - apply bootstrap bias correction of the initial estimate? (T/F)
  # niter_bbc - number of correction iterations
  # correc_bbc - method of correction: "add" or "mult"
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_lines[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_lines)
  #         band_V_esd      (theoretical sampling-noise-std in ensm band variances)
  # 
  # M Tsy 2020 June, 2021 Apr
  #-------------------------------------------------------------------------------------------
  
  nx    = dim(tranfu2)[1]
  nband = dim(tranfu2)[2] ;  J=nband
  
  nmax = nx /2 ;      lmax=nmax
  nmaxp1 = nmax +1 ;  lmaxp1=nmaxp1
  
  b_mean = matrix(0, nrow = nx, ncol = nband)
  
  #----------------------------------
  # Calc c_j

  cj = apply(tranfu2, 2, sum)
  
  #----------------------------------
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
  #----------------------------------
  # from v[j]  to  b_mean[j]

  for(ix in 1:nx){
    b_mean[ix, 1:nband] = band_V[ix, 1:nband] / cj[1:nband]
  }
  
  # image2D(band_V, main="band_V")
  # image2D(b_mean, main="b_mean")
  # plot(apply(b_mean, 2, mean), main="b_mean_Ms")
  #----------------------------------
  # band-mean wvns
  # l_mean[j] = sum_{l=0}^lmax  p_j[l] * l
  
  i_ll_full = c(1:lmaxp1)
  lp1_mean = apply(pjl, 1, function(t) sum(t*i_ll_full))
  
  l_mean = lp1_mean -1
  l_mean
  #----------------------------------
  # Intpl b_mean over n.
  # draw nsegment = nband -1 lines through pairs of points on the (n-b)
  # or (x-y) plane.
  
  b_lines1 = bmeanBand_intpl(l_mean, b_mean)
  
  b_lines1[b_lines1 < 0] =0
  
  # sum(abs(b_lines1[b_lines1 < 0])) / sum(abs(b_lines1))
  # mx= max(b_lines1, b_true)
  # image2D(b_lines1, main="b_lines1", zlim=c(0,mx))
  # image2D(b_true, main="b_true", zlim=c(0,mx))
  # 
  # inm=nx/10
  # ix=sample(c(1:nx), 1, replace=F)
  # mx=max(b_lines1[ix,1:inm],b_true[ix,1:inm] )
  # plot(b_lines1[ix,1:inm], main="b_lines1[ix,] (blue), b_true (black) \n b_mean_j (circles)", ylim=c(0,mx),
  #      type="l", col="blue", xlab="n+1")
  # lines(b_true[ix,1:inm])
  # ind = which(l_mean < inm +2, arr.ind = TRUE)
  # points(x=l_mean[ind]+1, y=b_mean[ix,ind])
  # 
  # # b_lines1_Ms = apply(b_lines1, 2, mean)
  # inm=nx/6
  # mx=max(b_lines1_Ms[1:inm], b_true_Ms[1:inm] )
  # plot(b_true_Ms[1:inm], main="b_true_Ms, b_lines1_Ms (red)", ylim=c(0,mx),type="l")
  # lines(b_lines1_Ms[1:inm], col="red")
  # 
  # norm(b_lines1 - b_true, "F") / norm(b_true, "F")
  # mean(b_lines1 - b_true) / mean(b_true)
  
  #-----------------------------------------------------------------------
  # Correct systematic errors in b_lines1 as follows.
  # Take b_lines1 as a truth, find the error, and subtract it.
  # 1) Set b_true2 = b_lines1
  # 2) Compute band_V2 from b_true2 (multiplying by tranfu2)
  # 3) compute b_mean2 from b_true2
  # 4) Draw b_lines2 from b_mean2 as above
  # 5) Compute the correction
  #        err2 := b_lines2 - b_true2
  # 6) Subtract err2 from the initial b_lines1  
  #    Try also a multiplic correction
  
  b_lines = b_lines1
  b_lines_prev = b_lines1
  
  b_mean2 = b_mean # init
  
  if(V2b_bbc){
    for(iter in 1:niter_bbc){
      b_true2 = b_lines_prev  # from the previous iteration
      band_V2 = band_V   # init
      
      for (band in 1:nband){
        for (ix in 1:nx){
          band_V2[ix,band] = sum( tranfu2[,band] * b_true2[ix,] )
        }
      }
      for(ix in 1:nx){
        b_mean2[ix, 1:nband] = band_V2[ix, 1:nband] / cj[1:nband]
      }
      
      b_lines2 = bmeanBand_intpl(l_mean, b_mean2)
      
      # Now, if the true spectrum is b_true2 =b_lines at the previous iteration,
      # the restored from the true band variances spectrum is b_lines2.
      # The add  err is then b_lines2 - b_true2 (to be subtracted)
      # The mult err is then b_lines2 / b_true2 (to be divided by)
      
      if(correc_bbc == "add"){
        err2 = b_lines2 - b_true2
        b_lines = b_lines1 - err2
      
      }else if(correc_bbc == "mult"){
        err2 = matrix(1, nrow = nx, ncol = nx)
        err2[b_true2 > 0] = b_lines2[b_true2 > 0] / b_true2[b_true2 > 0]
        b_lines[b_true2 > 0] = b_lines1[b_true2 > 0] / err2[b_true2 > 0]
      }
      
      b_lines[b_lines < 0] = 0
      
      b_lines_prev = b_lines # for the next iteration
      
      if(lplot){
        inm=nx/20
        if(inm <20) inm=20
        if(inm >40) inm=40
        ix=sample(c(1:nx), 1, replace = TRUE)
        mx=max(b_lines1[ix,1:inm], b_lines2[ix,1:inm], b_true[ix,1:inm] )
        plot(b_lines1[ix,1:inm], main="b_lines1[ix,] (blue), b_true (black) \n b_lines (red), b_mean_j (circles)", ylim=c(0,mx),
             type="l", col="blue", xlab="n+1")
        lines(b_true[ix,1:inm])
        lines(b_lines[ix,1:inm], col="red")
        ind = which(l_mean < inm +2, arr.ind = TRUE)
        points(x=l_mean[ind]+1, y=b_mean[ix,ind])
      }
    }
  }
  
  norm(b_lines1 - b_true, "F") / norm(b_true, "F")
  mean(b_lines1 - b_true) / mean(b_true)
  norm(b_lines - b_true, "F") / norm(b_true, "F")
  mean(b_lines - b_true) / mean(b_true)
  
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by the estmted b_LSM
  # For any ix, and any band j
  # v_j_restored = sum_i_n( tranfu2[,j] * b_lines[ix,] )
  
  band_V_restored = matrix(nrow = nx, ncol = nband) # [ix, band]

  for(ix in 1:nx){
    band_V_restored[ix,] = apply( tranfu2, 2, function(t) sum(t * b_lines[ix,]) )
  }
  
  # mx=max(band_V, band_V_restored)
  # image2D(band_V, main="band_V", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")

  norm(band_V_restored - band_V, "F") / norm(band_V, "F")
  
  #-----------------------------------------------------------------------  

  return(list("b_lines"=b_lines,
              "band_V_restored"=band_V_restored)) # band variances restored from b_lines
}