
B2S_param = function(band_V, Omega_nmax, 
                     b_shape, a_search_times, na,
                     eps_V_sd, band_Ve_TD, 
                     eps_smoo, w_a_rglrz, 
                     BOOTENS, band_Ve_B, lplot){
  #-------------------------------------------------------------------------------------------
  # Bands To Spectrum.
  # Fit the following prm mdl to b_tofit[n] 
  # 
  #    b_tofit[n] \approx A* b_shape(n/a)              (*)
  #
  #    Methodology
  #    
  # The prms A,a are required to minimize the misfit to DATA, d (ensm band variances).
  # The obs eqn is
  #  
  #    d = Omega * y + err
  #    d_mod = Omega * y
  # 
  # Here d is the band variances vector (of length J), 
  #      y is the spectrum b we seek at each ix independently.
  #      Omega is defined as follows.
  #      Omega is a J*N mx.
  # At ix,     d[1:J] = sum tranfu2[1:nx] * b_true[ix,1:nx]
  # As length(y) = nmaxp1=N  and  b_true[ix,1:nx] is an Even fu on S1(nx),
  #          Omega[,1] = tranfu2[1,]
  #   1<n<N: Omega[,n] = tranfu2[i_n,] + tranfu2[i_n_symm,] 
  #          Omega[,N] = tranfu2[N,]
  # 
  # The data constraint is then
  # 
  #    ||Omega * y||_L2  -->  min
  #    
  # The min is found for each ix independently,
  # by exhaustive searching over  a  and exact solution for  A. 
  # 
  # 
  #    Args
  # 
  # band_V[1:nx, 1:nband] - [ix, band] BAND variances 
  #               (normally, estimated from the ensemble or may be the true band varinces) 
  #                           at the grid point x
  # Omega_nmax[1:J, 1:nmaxp1] -- d=Omega_nmax * b[1:nmaxp1]
  # b_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*b_shape(n/a)
  # a_search_times = max deviation of  a  for exhaustive search (in times)
  # na - nu of grid points for exhaustive search over  a  (an odd number)
  # eps_smoo - rel strength of penalty on Smoothness of errors in V_j
  # w_a_rglrz
  # eps_V_sd - portion of mean TSD(band_Ve) used as an additive rglrzr 
  # band_Ve_TD - theor SD of sampling errors in band_Ve
  # BOOTENS - TRUE if the bootstrap band variances  band_Ve_B  are available
  # band_Ve_B - bootstrap sample of  band_V, [ix,j=1:J,iB] (O means
  #               all bands arouund the spectrral circle)
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  #         
  # 
  # M Tsy 2020 Sep, 2021 Feb
  #-------------------------------------------------------------------------------------------

  nx = dim(band_V)[1] 
  J  = dim(band_V)[2] ; nband=J
  if(BOOTENS) nB = dim(band_Ve_B)[3]

  nmax=nx/2
  nmaxp1=nmax+1
  nmaxp2=nmax+2
  N = nmaxp1 # dim-ty of y
  
  ii_n_half = c(1:N)
  
  Omega = Omega_nmax
  
  #-----------------------------------------------------------------
  # SD(Ve_uncertainty)
  
  V_td_min = eps_V_sd * apply(band_Ve_TD, 1, mean)  # mean over bands, [ix] 
  V_sd = band_Ve_TD + V_td_min     # [ix,j]
  V_weights = 1/V_sd^2

  #-----------------------------------------------------------------
  # band_V comparis w band_Vt
  
  if(lplot){
    ix=sample(c(1:nx),1)
    mx=max(band_V[ix,], band_Vt[ix,])
    plot(band_V[ix,], main=paste0("Band variances at ix=", ix), pch=19, lwd=4,
         col="magenta", ylim=c(0,mx), xlab = "Band", ylab = "V")
    lines(band_Vt[ix,], lty=3, lwd=3)
    lines(V_sd[ix,], type="p", pch=20, lwd=3, col="green")
    
    leg.txt<-c('Ensemble', 'Truth', 'SD')
    leg.col<-c("magenta", "black", "green")
    legend("topright", inset=0, leg.txt, col=leg.col, 
           lwd=c(4,3,0.5, 3), lty=c(NA,3,3,NA), pch=c(19,NA,NA, 20),
           pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
    
    
    ix=sample(c(1:nx),1)
    lV = log(band_V[ix,])
    lVt = log(band_Vt[ix,])
    mn=min(lV,lVt)
    mx=max(lV,lVt)
    plot(lV, main=paste0("Band log-variances at ix=", ix), pch=19, 
         col="magenta", ylim=c(mn,mx), xlab = "Band", ylab = "log(V)")
    lines(lVt, lty=3, lwd=3)
    
    leg.txt<-c('Ensemble', 'Truth', 'SD')
    leg.col<-c("magenta", "black", "green")
    legend("topright", inset=0, leg.txt, col=leg.col, 
           lwd=c(NA,3,0.5), lty=c(1,3,3), pch=c(19,NA,NA),
           pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
    
  }
  
  # jm = apply(band_V, 1 , function(t) sum(t)/t[1]) # j-macro-scale
  # plot(jm, type="l")
  #-----------------------------------------------------------------------
  # Specify grid for  a
  # a_i is to grow on the log scale
  # a_i /a_{i-1} = mu
  # a[2] = a[1] *mu
  # ...
  # a[na] = a[1]*mu^{na-1}   ==>
  # log(a[na] / a[1]) = (na-1) * log(mu)
  # log(mu) = log(a[na] / a[1]) / (na-1)
  
  a_search = c(1:na)
  a_search[1] = 1/a_search_times
  a_search[na] = a_search_times

  logmu=log(a_search[na] / a_search[1]) / (na-1)
  mu=exp(logmu)
  a_search = a_search[1] * mu^{c(0:(na-1))}
  # plot(a_search)
  
  #-----------------------------------------------------------------------
  # MAIN PART. Optimize.
  # Calc the resulting b_fit
  # f(n) = A*g(n/a) 
  # For a fixed a, the optimum A is
  # 
  #      sum_1^J  c_j e_j V_j  +  eps_smoo * sum_2^J  c_j d_e_j d_V_j
  #  A = ------------------------------------------------------------
  #      sum_1^J  c_j e_j^2    +  eps_smoo * sum_2^J  c_j d_e_j^2
  #  
  # Here
  # c_j = V_weights[ix,]
  # V_j = band_V[ix,j]
  # e_j = Omega * g(./a)
  # d_e_j = e_j - e_{j-1}
  # d_V_j = V_j - V_{j-1}
  # 

  AA_prm = c(1:nx) # init result
  aa_prm = c(1:nx) # init result
  Aa = c(1:na)     # aux
  
  b_fit = matrix(0, nrow=nx, ncol=nx) # [ix, i_n]
  
  b_shape_n_scaled_search = matrix(0, nrow=N, ncol=na) # aux
  e_search  =matrix(0, nrow=J,   ncol=na) # aux
  d_e_search=matrix(0, nrow=J-1, ncol=na) # aux
  
  #lower_bounds = c(0, 1)
  #upper_bounds = c(AA_max, aa_max)
  
  # Prepare aux arrays
  
  for(ia in 1:na){
    a = a_search[ia]
    b_shape_n_scaled = evalFuScaledArg(ii_n_half, b_shape, a) # g(n/a)
    b_shape_n_scaled_search[,ia] = b_shape_n_scaled
    e_search[,ia] = drop( Omega %*% b_shape_n_scaled )
    d_e_search[,ia] = e_search[2:J, ia] - e_search[1:(J-1), ia]
  }
  misfits = c(1:na) ; misfits[] = Inf
  
  # Main loop
  
  for (ix in 1:nx){
    # Search  a
    for(ia in 1:na){
      a = a_search[ia]
      b_shape_n_scaled = b_shape_n_scaled_search[,ia]
      e = e_search[,ia]
      d_e = d_e_search[,ia]
      
      V = band_V[ix,]
      d_V = V[2:J] - V[1:(J-1)]
      c = V_weights[ix,]
      cd = c[2:J]
      
      numerator =  drop( crossprod(c, e*V) ) +
        eps_smoo * drop( crossprod(cd, d_e*d_V) )
      denominator = drop( crossprod(c, e^2) ) +
         eps_smoo * drop( crossprod(cd, d_e^2) )
      
      A = numerator/denominator
      Aa[ia] = A
      misfits[ia] = V_prm_misfit(V, e, d_e, d_V, A, a, c, cd, eps_smoo, w_a_rglrz)
    }
    # plot(misfits)
    ia_opt = which(misfits == min(misfits), arr.ind = T)
    
    aa_prm[ix] = a_search[ia_opt]
    AA_prm[ix] = Aa[ia_opt]
    
    # The resulting fit

    b_fit[ix,1:N] = Aa[ia_opt] * b_shape_n_scaled_search[,ia_opt]
    b_fit[ix, (N+1):nx] = b_fit[ix, rev(2:nmax)]
  }
  
  # plot(AA_prm)
  # plot(aa_prm)
  
  # sum(b_fit < 0) / sum(b_fit >= 0)
  # b_fit[b_fit < 0] = 0
  
  #-----------------------------------------------------------------------
  #-----------------------------------------------------------------------
  # Diags
  
  if(lplot){
    ix=sample(c(1:nx), 1)
    inm=nx/6
    plot(b_true[ix,1:inm]/b_true[ix,1], 
         main=paste0("b_true/b_true[1] (circ), b_fit/b_fit[1] \n ix=", ix),
         ylim=c(0,1), xlab="n+1")
    lines(b_fit[ix,1:inm]/b_fit[ix,1], col="red")
    
    ix=sample(c(1:nx), 1)
    inm=nx/6
    mn=min(b_true[ix,1:inm], b_fit[ix,1:inm])
    mx=max(b_true[ix,1:inm], b_fit[ix,1:inm])
    plot(b_true[ix,1:inm], 
         main=paste0("b_true (circ), b_fit \n ix=", ix),
         ylim=c(mn,mx), xlab="n+1")
    lines(b_fit[ix,1:inm], col="red")
    abline(h=0)
    
    # d=20
    # j=4
    # mx = max(band_Vt[(ix-d): (ix+d), j], band_V[(ix-d): (ix+d), j])
    # plot(band_Vt[(ix-d):(ix+d), j], ylim=c(0,mx))
    # lines(band_V[(ix-d):(ix+d), j])
    # 
    # mx = max(b_true[(ix-d): (ix+d), j], b_fit[(ix-d): (ix+d), j])
    # plot(b_true[(ix-d):(ix+d), j], ylim=c(0,mx))
    # lines(b_fit[(ix-d):(ix+d), j])
    
    
    b_fit_Ms = apply(b_fit, 2, mean)
    
    inm=nx/6
    mx=max(b_fit_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_fit_Ms[1:inm], main="b_fit_Ms (red), b_true_Ms", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
    lines(b_true_Ms[1:inm], lwd=2)
    
  }
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by the estmted b_fit
  # For any ix, and any band j=1:J
  # V_j_restored = sum_i_n( tranfu2[,j] * b_fit[ix,] )

  band_V_restored = t( apply( b_fit[,1:nmaxp1], 1, function(t) drop(Omega %*% t) ) )
  
  # mx=max(band_V, band_V_restored)
  # image2D(band_V, main="band_V", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")
  
  if(lplot){
    ix=sample(c(1:nx), 1)
    mx=max(band_V[ix,], band_V_restored[ix,], band_Vt[ix,])
    plot(band_V[ix,], main="V_e (circ), V_param_fit(red), V_tru(blu)", ylim=c(0,mx),
         xlab = "band", ylab = "Band variances",
         sub=paste0("ix=", ix))
    lines(band_V_restored[ix,], col="red")
    lines(band_Vt[ix,], col="blue")
    
    
    ix=sample(c(1:nx),1)
    plot((band_V[ix,] - band_Vt[ix,])/V_sd[ix,])
    abline(h=0)
  }
  
  norm(band_V_restored - band_V, "F") / norm(band_V, "F")
  norm(b_fit - b_true, type="F") / norm(b_true, type="F")
  
  #-----------------------------------------------------------------------  

  return(list("b_fit"=b_fit,
              "AA_prm"=AA_prm, 
              "aa_prm"=aa_prm, 
              "band_V_restored"=band_V_restored)) # band variances restored from b_fit
}
