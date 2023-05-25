
gen_obs = function(x_true, n_obs, sd_obs, repeated_obs_location = TRUE){
  #-------------------------------------------------------------------------------------------
  # Generate randomly located observations for the ANLS.
  # 
  # That is, given
  # 
  # (1) the truth, x_true,
  # (2) the number of obs, n_obs,
  # (3) the obs-err st.dev., sd_obs,
  # 
  # and assuming that
  # 
  # (i) the obs are collocated with some randomly chosen 
  #       (with or without replacement) grid points,
  # (ii) the obs errors are uncorrelated 
  # 
  # specify:
  # 
  # a) the obs values, the x_obs vector,
  # b) the obs oprt, H,
  # c) the obs error cvm, R.
  # 
  #    Args:
  # 
  # x_true[1:nx] - truth
  # n_obs - nu of OBS
  # sd_obs - obs-err SD
  # repeated_obs_location - allow several obs to be located at the same grid point? (T or F)
  # 
  # Return: H, R, x_obs
  # 
  # M Tsy 2020 Sep, 2022 Apr
  #-------------------------------------------------------------------------------------------
  
  nx = length(x_true)
  
  # Specify random locations of the obs
  
  if(repeated_obs_location){
    replacement = TRUE
  }else{
    replacement = FALSE
  }
  
  ind_obs = sample(c(1:nx), size = n_obs, replace = replacement) 
  ind_obs = sort(ind_obs)
  
  # Specify H
  
  H = matrix(0, nrow = n_obs, ncol = nx)
  for (i_obs in 1:n_obs){
    ind = ind_obs[i_obs]
    H[i_obs, ind] = 1
  }
  
  # R
  
  R = sd_obs^2 * diag(n_obs)
  
  # Generate obs errors
  
  eta = sd_obs * rnorm(n_obs, mean=0, sd=1)
  
  # Generate obs values
  
  x_obs = as.vector(H %*% x_true) + eta
  
 
  return(list("H"=H, "R"=R, "x_obs"=x_obs))
}
 

