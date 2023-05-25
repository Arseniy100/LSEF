
SpatAveCrf_S1 = function(cvm){
  
  #----------------------------------------------------------------------
  # Given the input cvm, calc the spatially-averaged crf --
  # by averaging over all the diagonals.
  #
  # return: crf & var_mean
  #
  # Required functions:
  # extract_cyclic_superdiagonals
  #
  # M Tsy 2021 Jan
  #----------------------------------------------------------------------
  
  n = dim(cvm)[1]
  p=n-1
  
  var_mean = mean(diag(cvm))
    
  crm  = Cov2VarCor(cvm)$C
  
  sdiagonals = extract_cyclic_superdiagonals(crm,p)$sdiagonals # n*(p+1) mx
  
  # Ave over the diagonals (columns in sdiagonals). 
  # NB: sdiagonals[ix, shift], shift=0,1,..., n-1
  # Reconstruct the mean crf (an n-vector) 
  
  # crf = colMeans(sdiagonals)
  crf = .colMeans(sdiagonals, n, n)
  

  return(list("var_mean"=var_mean, "crf"=crf))
  
} 