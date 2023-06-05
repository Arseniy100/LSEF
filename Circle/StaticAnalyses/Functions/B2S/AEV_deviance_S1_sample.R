
AEV_deviance_S1_sample = function(ff, gg, obs_err_variance_supplied, obs_err_variance = -1){
  #--------------------------------------------------------------------------------
  # Calc anls-err variance based deviance between two spectra -- 
  # for a number of instances in the sample.
  # 
  # 
  # deviance_AEV = deviance_AEV_per_wvn[1] + deviance_AEV_per_wvn[nmaxp1] +
  #            2* sum (  deviance_AEV_per_wvn[2:nmax] )
  # 
  #    Args
  #            
  # ff[1:n_sample, 1:nmaxp1] - true (reference) spectra
  # gg[1:n_sample, 1:nmaxp1] - spectra in question
  # obs_err_variance_supplied - logical: is the physical-space obs_err_variance is
  #      supplied as the argument?
  #      if TRUE, r  is computed from the next argument, the physical-space obs_err_variance,
  #      if FALSE, r is computed from the true (f) spectrum
  # obs_err_variance = r (obs_err_variance = -1 just indicates that -1 is never used)
  # 
  #    Return
  #    
  # loss-aev_sampleMean
  # 
  # M Tsy 2023 Jan
  #--------------------------------------------------------------------------------
  
  n_sample = dim(ff)[1]
  nmaxp1 =   dim(ff)[2]
  nmax = nmaxp1 -1
  nx = 2*nmax
  
  # Anls-err variance loss
  
  lloss_aev = c(1:n_sample)
  for(i in 1:n_sample){
    f = ff[i,]
    g = gg[i,]
    lloss_aev[i] = AEV_deviance_S1(f, g, obs_err_variance_supplied, obs_err_variance)$deviance
  }
  # plot(lloss_aev)
  aev_sampleMean = mean(lloss_aev)
  
  return(aev_sampleMean)    
}
  