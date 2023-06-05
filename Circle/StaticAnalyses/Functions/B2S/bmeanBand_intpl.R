bmeanBand_intpl = function(l_mean, b_mean){
  
  #*************************************************************************
  # Intpl b_mean (attributed to  l_mean[]) over n.
  # draw nsegment = nband -1 lines through pairs of points on the (n-b)
  # or (x-y) plane.
  # The line through the 2 points 
  # P1=(x1, y1) and
  # P2=(x2, y2) is
  # ..................................................
  #  y=y1 + (x-x1) * (y2-y1)/(x2-x1) == y1 + coef*(x-x1)
  # ..................................................
  # 
  #    Args
  # 
  # l_mean[1:nband] - mean values of n for all bands bands (non-integer)
  # b_mean[ix, j] - mean b over band j at the grid point x
  # 
  # Return: b_intpl
  #
  # M Tsy 2020 Oct
  #*************************************************************************
  
  nx    = dim(b_mean)[1]
  nband = dim(b_mean)[2]
  
  nmax = nx/2
  nmaxp1 = nmax +1

  b_intpl = matrix(0, nrow = nx, ncol = nx) # [ix, i_n]
  
  # First, fill in b_intpl[ix,] for n in 0:nmax, i.e. i_n=1:(nmax+1)
  # (fill in the rest just below)
  
  nsegment = nband -1
  for(segment in 1:nsegment){ 
    
    band1=segment
    band2=segment +1
    
    x1=l_mean[band1]
    x2=l_mean[band2]
    
    y1=b_mean[,band1] # [ix]
    y2=b_mean[,band2] # [ix]
    
    # Treat here n that lie within [x1, x2]: 
    # n = n_left : n_right

    n_left  = ceiling(x1)
    if(segment == 1) n_left = 0
    
    n_right = floor(x2)
    if(segment == nsegment) n_right = nmax
    
    if(n_left <= n_right){ # non-empty segment 
      
      coef = (y2 - y1) / (x2 - x1)  # [ix]
      
      for(n in n_left : n_right){
        b_intpl[,(n+1)] = y1 + coef*(n-x1)
      }  
    }
  }
  
  # fill in b_intpl[ix,] for i_n in (nmax+2):nx -- from symmetry
  
  for(i_n in (nmax+2) : nx){
    i_n_mainpart = nx - i_n +2   # i_n already filled in
    b_intpl[,i_n] = b_intpl[,i_n_mainpart] 
  }
  
  return(b_intpl)
}