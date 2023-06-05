
AEV_deviance_S1 = function(f, g, obs_err_variance_supplied, obs_err_variance = -1){
  #--------------------------------------------------------------------------------
  # Calc anls-err variance based deviance between two spectra.
  # 
  # 1) Select the spectral analysis err variance  r  from the assumption
  #    the it corresponds to the physical-space idealized-anls-obs-err variance
  #    Equal to the true Background error variance.
  #    
  #     v_bck_true = f[1] + f[nmaxp1] + 2 * sum( f[2:nmax] )
  #
  #   so, set
  #   
  #   r = v_bck_true
  # 
  #  2) Loss per wvn:
  #  
  #  y_true     =: f
  #  y_examined =: g
  #  
  # deviances_per_wvn = r^2 * (f-g)^2 /  ( (f+r)*(g+r)^2 )
  # 
  # deviance = deviances_per_wvn[1] + deviances_per_wvn[nmaxp1] +
  #            2* sum (  deviances_per_wvn[2:nmax] )
  # 
  #    Args
  #            
  # f - true (reference) spectrum on S1
  # g - spectrum in question
  # obs_err_variance_supplied - logical: is the physical-space obs_err_variance is
  #      supplied as the argument?
  #      if TRUE, r  is computed from the next argument, the physical-space obs_err_variance,
  #      if FALSE, r is computed from the true (f) spectrum
  # obs_err_variance = r (obs_err_variance = -1 just indicates that -1 is never used)
  # 
  #    Return
  #    
  # loss-aev
  # 
  # M Tsy 2022 Dec
  #--------------------------------------------------------------------------------
  
  nmaxp1 =length(f)
  nmax = nmaxp1 -1
  nx = 2*nmax
  
  # Select r from  phys-space  obs_err_variance
  # As the obs errs are uncrlted and have the same variance, so do they 
  # in spe space as well.
  # There are  nx  spectral components in spe space, so per spectral
  # component (per wvn)  we have  obs_err_variance / nx
  
  if(obs_err_variance_supplied){
    r = obs_err_variance / nx
  }else{
    v_bck_true = f[1] + f[nmaxp1] + 2 * sum( f[2:nmax] )
    r = v_bck_true / nx
  }

  coeffs = c(1:nmaxp1) ;  coeffs[] = 2
  coeffs[1] = 1 
  coeffs[nmaxp1] = 1
  
  deviances_per_wvn = r^2 * (f-g)^2 /  ( (f+r) * (g+r)^2 )
  deviance = deviances_per_wvn[1] + deviances_per_wvn[nmaxp1] + 2*sum( deviances_per_wvn[2:nmax] )
  
  return(list("deviances_per_wvn"=deviances_per_wvn,
              "deviance"=deviance))    
}
  