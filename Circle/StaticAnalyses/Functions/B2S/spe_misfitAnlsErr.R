spe_misfitAnlsErr = function(spe, spe_tru, r){
  
  #-----------------------------------------------------------------------
  # Given two spectra on S1:  spe  and  spe_tru,
  # calc their ``difference'' using the metric that is related to the RMSE of
  # an idealized analysis.
  # 
  #   
  #  L = {(spe - spe_tru)^2}  / {(spe_tru  + r) (spe  + r)^2}
  #  
  #  where .
  #  NB: The obs err is const in spe space so that the 
  #      phys-space obs err is nx*r.
  #      We take  r  in such a way that nx*r equals the halfsum of 
  #      spe1-phys-space variance and spe2-phys-space variance
  #  
  #  
  #   Arguments
  # 
  # spe, spe_tru[1:nmaxp1] - non-neg wvn part of the two spectra 
  # r - the obs err variance per spectral mode
  #  
  # M Tsy 2022 Aug
  #-----------------------------------------------------------------------
  
  
  nmaxp1 = length(spe1)
  nmax = nmaxp1 -1
  nx = 2 * nmax
  
  #-----------------------------------------------------------------------  
  # misfit
  # L = {(spe - spe_tru)^2}  / {(spe_tru  + r) (spe  + r)^2}
  
  arr = (spe - spe_tru)^2 / ( ( spe_tru +r ) * ( spe +r )^2 )
  
  misfit = varfu(arr)
  # misfit
  
  #-----------------------------------------------------------------------  
  
  return(misfit)
}
  
  