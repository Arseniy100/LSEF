BshiftMx_S1 = function(nx){
  
  # Create the "backshift" mx, which, when applied to an n-vector,
  # circularly (on S1) shifts its entries:
  # y = B * x  ==>  y_i = x_{i-1}
  # NB: FshiftMx_S1 = t(BshiftMx_S1)
  
  
  BshiftMx = matrix(nrow=nx, ncol=nx, data=0)
  
  BshiftMx[1,nx] = 1
  for (i in 2:nx) BshiftMx[i,i-1] = 1
  
  return(BshiftMx)
}
