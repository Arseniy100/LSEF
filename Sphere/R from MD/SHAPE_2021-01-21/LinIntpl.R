
LinIntpl = function(xx_f_vec, f_vec, xx_intpl){
  
  #*************************************************************************
  # Intpl the values at a Regular grid that starts from x=1 and 
  # whose mesh size is equal to 1: x=1,2,..,nf
  # stored in vector f_vec to the vector of arguments xx_intpl.
  # If some of xx_intpl[] are  beyond the grid, f_vec is extpltd by const.
  #
  # Args
  # 
  # xx_f_vec[1:nf] - vector of ascissae where f_vec is defined 
  # f_vec[1:nf] - vector of function values at grid points 1,2,...,nf.
  # xx_intpl[1:nxx_intpl] - vector of points where f is to be evaluated
  # 
  # Return: ff   = f(xx_intpl[])
  #
  # M Tsy 2020 Sep
  #*************************************************************************
  
  nf = length(f_vec)
  dx = xx_f_vec[2] - xx_f_vec[1]
  x0 = xx_f_vec[1]
  
  nxx_intpl = length(xx_intpl)
  ff_intpl = xx_intpl #init
  
  # find the mesh where x lies
  
  for (ixx_intpl in 1:nxx_intpl){
    x=xx_intpl[ixx_intpl]
    
    i_low = floor((x-x0)/dx) +1
    i_upp = i_low +1
    
    if(i_low < xx_f_vec[1]){        # beyond the grid on the left
      ff_intpl[ixx_intpl] = f_vec[1]
      
    }else if(i_upp > xx_f_vec[nf]){ # beyond the grid on the right
      ff_intpl[ixx_intpl] = f_vec[nf]
      
    }else{                 # within the grid 
      
      x1=xx_f_vec[i_low]
      x2=xx_f_vec[i_upp]
      
      y1=f_vec[i_low]
      y2=f_vec[i_upp]
      
      coef=(y2-y1)/(x2-x1)
      
      ff_intpl[ixx_intpl] = y1 + coef*(x-x1)
    }
  }
  
  return(ff_intpl)
}