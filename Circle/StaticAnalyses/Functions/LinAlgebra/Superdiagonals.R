# Contents:
# extract_cyclic_superdiagonals 
# construct_symm_mx_from_cyclic_superdiagonals
# M Tsy 2017 


extract_cyclic_superdiagonals <- function(A,p){
  
  # extract the main diag & p super-diagonals from mx A.
  # Do this CYCLICALLY, i.e. each super-diagonal has the same length n
  # and continues as on the circle
  # p is the nu of super-diags
  # returns:
  #  1) superdiagonals: superdiagonals only (an n*p mx).
  #  2) sdiagonals: superdiagonals AND the main diagonal (an n*(p+1) mx).
  
  n <- dim(A)[1]  # dim-ty
  sdiagonals      <- matrix(nrow=n, ncol=p+1)
  superdiagonals  <- matrix(nrow=n, ncol=p)
  
  # Extract the main diagonal
  
  #diagonals[,1] <- diag(A)
  maindiag <- diag(A)
  
  # Extract the super-diagonals
  
  for (i in 1:p){
    superdiagonals[1:(n-i),   i] <- 
      diag(matrix(as.vector(A[-(n-i+1):-n, -1:-i]), nrow=n-i, ncol=n-i))  # diagonals
    superdiagonals[(n-i+1):n, i] <- 
      diag(matrix(as.vector(A[-1:-(n-i), -(i+1):-n]), nrow=i, ncol=i))  # corners
  }
  
  sdiagonals <- cbind(maindiag, superdiagonals)
  
  return(list("sdiagonals"=sdiagonals, "superdiagonals"=superdiagonals))
}



construct_symm_mx_from_cyclic_superdiagonals <- function(sdiagonals){
  
  # From the main diag & p super-diagonals (all put in 'sdiagonals')
  #  of a symmetric mx A, build the whole A.
  # NB: each super-diagonal has the same length n and continues as on the circle.
  # returns A.
  
  n <- dim(sdiagonals)[1]       # dim-ty
  p <- dim(sdiagonals)[2] -1    # nu of superdiags
  
  A <- matrix(0, nrow=n, ncol=n) # initialize by 0's

  # Fill in the super-diagonals
  
  for (i in 1:p){
    a <- matrix(0, nrow=n-i, ncol=n-i) # submx with the MAIN part of the
                                       # i-th A-superdiag on its main diag
    diag(a) <- sdiagonals[1:(n-i),   i+1]  
    A[-(n-i+1):-n, -1:-i] <- A[-(n-i+1):-n, -1:-i] + a # add the MAIN part of the diagonal
    
    b <- matrix(0, nrow=i, ncol=i)     # submx with the CORNER part of the
                                       # i-th A-superdiag on its main diag
    diag(b) <- sdiagonals[(n-i+1):n, i+1]
    A[-1:-(n-i), -(i+1):-n] <- A[-1:-(n-i), -(i+1):-n] + b       # corners
  }
  
  # Fill in the subdiagonals by symmetry
  
  A <- A + t(A)
  
  # Finally, fill in the main diagonal
  
  diag(A) <- sdiagonals[,1]
  
  return(A)
}



