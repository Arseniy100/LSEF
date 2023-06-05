
dsadm_generator <- function(X_mdl_start, n, ne, ntime, dt, U, rho, nu, sigma, Rem, forcing){
  
  # Integrate DSADM FOR MULTIPLE TIME STEPs.
  # An ensemble of ne members is generated.
  # Can be used as the FCST model (ie unforced) or as 
  # the generating model of TRUTH (forced), or as 
  # the generating model for an ensemble member (possible truth, forced).
  # In the two latter (forced) cases, the forcing is generated within this routine
  #  as the white noise multiplied by sigma(t,s).
  # By default, forcing is ON.
  # 
  # The starting field  X_mdl_start  is the starting field for the output fcst field at step 1
  # (i.e. in fact,  X_mdl_start  pertains to t=0).
  #-------------------------------------------
  # Arguments:
  #
  # X_mdl_start[1:n, 1:ne] - ensm of initial fields
  # n - dim-ty of the state vector x
  # ne - ensm size
  # ntime - number of model time steps to be generated including the starting time instant
  #   (so that ntime should normally be >=2)
  # dt - time step, s
  # U[1:n], rho[1:n], nu[1:n], sigma[1:n] - coefficient fields of the DSADM
  # Rem - Earth radius, m
  # forcing - logical switch: if TRUE, then forcing is computed here and added to the solution,
  #                           if FALSE, the pure FCST is computed.
  # 
  # Return: the generated space-time field: field[space, time].
  #
  # M Tsyrulnikov
  # June 2018, 
  # May 2021: step 1 --> step 0
  #***************************************************************
  
  field = matrix(0, nrow=n, ncol=ntime)
  field3D = array(0, dim=c(n, ne, ntime))
  
  start = X_mdl_start # the fcst valid at t=1  starts from X_mdl_start (at t=0)
  
  if(ntime > 1){
    for (t in 1:ntime){
      fcst = dsadm_step(start, n, ne, dt, U[,t], rho[,t], nu[,t], sigma[,t], Rem, forcing)
      field3D[,,t] = fcst
      start = fcst # for the next time step
    }
  }
  
  # If ne=1, return 2D array, otherwise 3D array
  
  if(ne == 1){
    field = field3D[,1,]
  }else{
    field = field3D
  }
  return(field)
}
