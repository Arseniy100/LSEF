
lcz_matrix = function(n ,c, Rem){
  
  #---------------------------------------
  # Create a lcz mx using Gaspari-Cohn lcz function, see
  # Gaspari and Cohn (1999, Eq.(4.10))
  # 
  # Args
  # 
  # n - number of grid points on the circular domain
  # c - lcz length scale (NB: the lcz functions vanishes at distances > 2*c), m
  # Rem - Earth radius, m
  #
  # Return: C (the lcz mx)
  #---------------------------------------
  
  C = matrix(0, ncol = n, nrow = n)

  hrad = 2*pi/n  # mesh size, rad.
  
  for(i in 1:n){
    for(j in 1:n){
      
      if(abs(i-j)<n/2){
        z = abs(i-j)             # greate-circle distance, mesh sizes
      }else{
        z = n - abs(i-j)         # greate-circle distance, mesh sizes
      }
      
      zrad = z * hrad            # greate-circle distance, rad
      
      z = 2*sin(zrad/2)*Rem # chordal distance, m
      
      zdc=z/c
      
      if(z>=0 & z<=c)   C[i,j] = -1/4*zdc^5 + 1/2*zdc^4 + 5/8*zdc^3 - 5/3*zdc^2 +1
      if(z>=c & z<=2*c) C[i,j] = 1/12*zdc^5 - 1/2*zdc^4 + 5/8*zdc^3 + 5/3*zdc^2 - 5*zdc + 4 -2/3*c/z
      
      # C[i,j] = exp(-0.5*zdc^2)
    }
  }
  return(C)
}
