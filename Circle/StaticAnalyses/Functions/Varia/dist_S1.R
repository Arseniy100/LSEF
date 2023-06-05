 
dist_S1 = function(x,y){
  
  # distance between two points on S1
  # x and y can be vectors, then each component is treated independently
  # NB: The components of both x & y should lie in the same segment:
  #     [0,2pi] or [-pi,pi]
  # 
  # M Tsy 2020 Sep
  
  dist_S1 = abs(x-y)
  dist_S1[dist_S1 > pi] = 2*pi - dist_S1[dist_S1 > pi]
  
  return(dist_S1)
}