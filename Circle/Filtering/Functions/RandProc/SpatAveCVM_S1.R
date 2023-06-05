
SpatAveCVM_S1 = function(cvm){
  
  #----------------------------------------------------------------------
  # Given the input cvm, calc the spatially-averaged (stationary) CVM --
  # by averaging over all the diagonals.
  #
  # return: crf & var_mean
  #
  # M Tsy 2021 Jun
  #----------------------------------------------------------------------
  
  n = dim(cvm)[1]
  p=n-1
  
  cvm_statio = matrix(0, nrow = n, ncol = n)
  
  sdiagonals = extract_cyclic_superdiagonals(cvm,p)$sdiagonals # n*(p+1) mx
  
  # Ave over the diagonals (columns in sdiagonals). 
  # NB: sdiagonals[ix, shift], shift=0,1,..., n-1
  
  cvf = .colMeans(sdiagonals, n, n)
  #sdiagonals_statio = t(matrix(cvf, nrow = n, ncol = p))
  
  # Reconstruct the the stationary-cvf CVM
  
  BackShiftMx = BshiftMx_S1(n)
  cvm_statio[1,] = cvf
  cvf_shifted = cvf
  for (i in 2:n){
    cvf_shifted = drop( BackShiftMx %*% cvf_shifted ) # move the row 1 element to the right
    cvm_statio[i,] = cvf_shifted
  }

  # image2D(cvm, main="cvm")
  # image2D(cvm_statio, main="cvm_statio")
  # plot(cvf)
  # plot(cvm_statio[1,])
  # plot(cvm_statio[11,])
  # plot(cvm_statio[31,])
  
  
  return(cvm_statio)
  
} 