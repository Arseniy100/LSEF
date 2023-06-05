
B2S_prmFitAnlsErr = function(b_tofit, b_shape, a_search_times, na, lplot){
  #-------------------------------------------------------------------------------------------
  # Fit a parametric model  A* b_shape(n/a)  
  # to the spectrum         b_tofit[n]  using 
  # (1) exhaustive search and
  # (2) the misfit defined as the deviance
  # 
  # misfit := sum (b[n]-b_tofit[n])^2 / [ (b_tofit[n] + r) (b[n] + r)^2  ]
  # 
  # defined as the excess of the variance of a hypothetical suboptimal anls performed 
  # (for each x independently)
  # with the NON-LOCAL spectrum  b[n] (i.e. the true field is Stationary)
  # over the variance of the Optimal analysis performed with the spectrum  b_tofit[n].
  # Both analyses are idealized in the sense that all grid points are observbed
  # (which implies that all spectral components resolved by the grid are
  # observed as well.)
  # 
  # r is the obs-err variance.
  # 
  #    b[n] =: A* b_shape(n/a)              (*)
  #
  #
  #    Methodology
  #        
  # The min misfit is found for each ix independently,
  # by exhaustive searching over  a  and exact solution for  A. 
  # 
  # 
  #    Args
  # 
  # b_tofit[ix=1:nx, i_n=1:nmaxp1] - the spectrum to be fitted by the prm mdl
  # b_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*b_shape(n/a)
  # a_search_times = max deviation of  a  for exhaustive search (in times)
  # na - nu of grid points for exhaustive search over  a  (an odd number)
  # lplot - plotting: draw plots only if TRUE
  # 
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points)
  #         band_V_restored (band variances restored from b_fit)
  #         
  # 
  # M Tsy 2022 Aug
  #-------------------------------------------------------------------------------------------

  nx     = dim(b_tofit)[1] 
  nmaxp1 = dim(b_tofit)[2]

  nmax=nmaxp1 -1
  N = nmaxp1 
  
  ii_n_half = c(1:N)

  #-----------------------------------------------------------------------
  # Calc the (background-error) variances  from b_tofit  for all  x
  
  var_tofit = apply(b_tofit, 1, function(t) t[1] + 2*sum(t[2:nmax]) + t[nmaxp1] )
  # plot(var_tofit)
  
  var_mean = mean(var_tofit)
  
  #-----------------------------------------------------------------------  
  # Specify  r (obs err variance per spectral mode)
  
  r = var_mean / nx

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

  AA_prm = c(1:nx) # init 
  aa_prm = c(1:nx) # init 
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
