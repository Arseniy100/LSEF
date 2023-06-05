
LinIntpl_genGrids = function(xx_in, ff_in, xx_ou){
  
  #*************************************************************************
  # Intpl the values of the input function  ff_in  defined on a GENERAL
  # grid  xx_in  to the GENERAL output grid  xx_ou.
  # GENERAL means not necessarily regular.
  # If some of  xx_ou[]  are  beyond the grid,  ff_in  is extpltd by const.
  #
  # Args
  # 
  # xx_in[1:nx_in] - vector of abscissae where ff_in is defined (increasing!)
  # ff_in[1:nx_in] - vector of function values at grid points 1,2,...,nx_in.
  # xx_ou[1:nx_ou] - vector of points where f is to be evaluated (increasing!)
  # 
  # Return: ff_ou
  #
  # M Tsy 2022 Feb
  #*************************************************************************
  
  # deb
  # xx_in = c(1,2,4)
  # ff_in = c(1,2,3)
  # xx_ou = c(0:15)/3
  # end deb
  
  nx_in = length(ff_in)
  nx_ou = length(xx_ou)
  n_int_in = nx_in +1 # nu of intervals of the input grid, inclu -Inf, +Inf

  ff_ou = xx_ou ;  ff_ou[] = 0 #init
  
  
  #-----------------------------------------------------------------
  # Checks: the grids should be monotonically increasing.
  
  eps = abs(xx_in[1] - xx_in[nx_in]) * 1e-10
  xx_in.nonIncreasing = diff(xx_in) < eps
  xx_ou.nonIncreasing = diff(xx_ou) < eps

  if(any(xx_in.nonIncreasing) | any(xx_in.nonIncreasing)){
    message("Either input or output gris in non-increasing")
    message("xx_in")
    print(xx_in) 
    message("")
    message("xx_in")
    print(xx_ou)  
    stop("Fix the grids", call.=TRUE)
    return()
  }
  
  # for each output grid point  x_ou,
  # find the mesh (interval) of the input grid where  x_ou  lies
  # Loop over the input grid
  
  for (int_in in 1:n_int_in){
    
    if(int_in == 1) {
    
      x_in_low = -Inf
      x_in_upp = xx_in[1]
      
    }else if(int_in == n_int_in){
      
      x_in_low = xx_in[nx_in]
      x_in_upp = Inf
      
    }else{
      
      ind_in_low = int_in -1
      ind_in_upp = int_in
      x_in_low = xx_in[ind_in_low]
      x_in_upp = xx_in[ind_in_upp]
    }
    
    ind_ou = which(xx_ou > x_in_low  &  xx_ou <= x_in_upp, arr.ind = T) 
    
    # Intpl:
    # f(x) = f_low + (f_upp - f_low)/(x_upp - x_low) * (x - x_low)
    #      = f_low + c * (x - x_low)
    
    if(int_in == 1) {                # extrap
      ff_ou[ind_ou] = ff_in[1]
      
    }else if(int_in == n_int_in){    # extrap
      ff_ou[ind_ou] = ff_in[nx_in]
      
    }else{
      c = (ff_in[ind_in_upp] - ff_in[ind_in_low]) / (x_in_upp - x_in_low)
      ff_ou[ind_ou] = ff_in[ind_in_low] + c * (xx_ou[ind_ou] - x_in_low)
    }
  }
  
  # plot(x=xx_in, y=ff_in)
  # plot(x=xx_ou, y=ff_ou)
  
  return(ff_ou)
}