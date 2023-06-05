
ThreePointSmoo_S1 = function(x, nsweep){
  
  #----------------------------------------------------------------------------
  # On a grid with n points on S1,
  # apply the simplest symmetric smoo flt
  # y[j] = a*x[j-1] + b*x[j] + a*x[j+1]
  # Its spectral transfer function is easily seen to be 
  # 
  # H(m) = 2*a* cos(m*(2*pi)/n) + b
  # 
  # and ranges from -2a+b at the maximal frequencies m=+-n/2
  # to 2a+b at m=0.
  # 
  # If we require that the highest grid-resolved frequency n/2 be 
  # completely damped, then -2a+b = 0 ==>
  # b=2a ==> 
  # 
  # y[j] = a*x[j-1] + 2a*x[j] + a*x[j+1]
  # 
  # H[m] = 2*a* (1+ cos(m*(2*pi)/n)) >=0
  # 
  # We have, in particular,
  # 
  # H[0] = 4a
  # 
  # If we, in addition, require that the filter does NOT change the MEAN value 
  # of its input x, then H[0]=1 ==>
  # 
  # a = 1/4 ;  b=1/2
  # 
  # Finally,
  #-----------------------------------------------
  # y[j] = 0.25*x[j-1] + 0.5*x[j] + 0.25*x[j+1]
  # 
  # H[m] = 0.5 * (1+ cos(m*(2*pi)/n))
  #----------------------------------------------- 
  #
  #   Args
  #
  # x - input vector on the regular grid on S1
  # nsweep >=0 - number of applications of the filter to x
  #
  # return(y)
  # 
  # M Tsy Feb 2020
  #----------------------------------------------------------------------------
  
  n=length(x) 
  y=x # init AND is returned if nsweep=0
  
  if(nsweep != 0) {
    for (isweep in 1:nsweep){
      y[1] = 0.25*x[n]   + 0.5*x[1] + 0.25*x[2]
      y[n] = 0.25*x[n-1] + 0.5*x[n] + 0.25*x[1]
      
      for (j in 2:(n-1)) {
        y[j] = 0.25*x[j-1] + 0.5*x[j] + 0.25*x[j+1]   
      }
      x=y
    }
  }

  return(y)
}