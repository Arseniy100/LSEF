
ThreePointSmoo_segment = function(x, nsweep, maintainMx = FALSE){
  
  #----------------------------------------------------------------------------
  # On a grid with n points on a segment [x[1], x[nx]],
  # apply the simplest symmetric smoo flt
  # y[j] = a*x[j-1] + b*x[j] + a*x[j+1]
  # (with the end points unchanged).
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
  # x - input vector on the regular grid on the segment
  # nsweep >=0 - number of applications of the filter to x
  # maintainMx - if TRUE, a constant is added to y such that
  #             mean(x)=mean(y)
  #
  # return(y)
  # 
  # M Tsy Apr 2020
  #----------------------------------------------------------------------------
  
  Mx=mean(x)
  n=length(x) 
  y=x # init AND is returned if nsweep=0
  
  if(nsweep != 0) {
    for (isweep in 1:nsweep){
      
      for (j in 2:(n-1)) {
        y[j] = 0.25*x[j-1] + 0.5*x[j] + 0.25*x[j+1]   
      }
      x=y
      
    }
  }
  
  if(maintainMx){
    My=mean(y)
    y = y - My + Mx
  }

  return(y)
}