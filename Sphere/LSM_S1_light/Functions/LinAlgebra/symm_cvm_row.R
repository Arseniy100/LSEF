
symm_cvm_row = function(row, n, i, direction = "f") {
  
  # From the i-th row of a cvm (of size n),
  # Compute the vector 
  # srow[1:n] such that :
  # 1) it is the cyclic rotation of the original mx row
  # 2) the i-th position of the original row (ie at the diagonal of the cvm)
  #    corresponds to the n/2 (center) position in srow
  # In other words, we just rotate the original row by the angle 2pi/n * (n/2 -i)
  # 
  # Args
  # 
  # row[1:n] - input vector
  # n = dim(CVM)
  # i - number of the row in the CVM
  # direction - if "f", then compute symm_row from row,
  #             if "b", do the inverse transform
  # 
  # return the transformed row, trow
  # 
  # M Tsy
  # 2018
  
  trow=row # init
  
  if(direction == "f"){
    shift = floor(n/2) -i
    
  }else if(direction == "b"){
    shift = i - floor(n/2)
  }else{
    stop("symm_cvm_row. Wrong direction")
  }
  
  for (js in 1:n){ # js is the position in the shifted ("symmetric") row
    
    j=(js - shift -1) %% n  +1  # j is the position in the original mx row
    
    trow[js] = row[j]
  }
  return(trow)
}