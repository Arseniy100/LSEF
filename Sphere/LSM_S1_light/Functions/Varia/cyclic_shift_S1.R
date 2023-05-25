
cyclic_shift_S1 = function(f, shift_clockwise=1) {
  
  #---------------------------------------------------------------------------------
  # For vector f[1:n] viewed as a function on a regular grid on S1,
  # perform its rotation in the clockwise direction by shift_clockwise grid points.
  #
  # return: f_shifted
  # 
  # Example: f=c(f1,f2,f3,f4) ==
  # 
  #         f2
  # 
  #    f3       f1
  #    
  #        f4
  # 
  # with shift_clockwise=1, we should obtain
  # 
  #         f3
  # 
  #    f4       f2
  #    
  #        f1
  # 
  # == shifted=c(f2,f3,f4,f1)
  # 
  # Return: f_shifted
  # 
  # M Tsy May 2020
  #---------------------------------------------------------------------------------
  
  # debug
  #n=4
  #f=c(1:4)
  #shift_clockwise=2
  # end debug
  
  n = length(f)
  shifted=c(1:n) # init
  
  for (is in 1:n){ # loop over shifted points 
    i = is + shift_clockwise
    if(i > n) i=i-n
    if(i < 1) i=i+n
    shifted[is] = f[i]
  }

  return(shifted)
}
  
  
  
  