
mult_squareMx_ADAT = function(A, d){
  
  #--------------------------------------------------------------
  # For a square n*n  mx  A  and an n-vector  d,  multiply
  # 
  #    A*diag(d)*A^T 
  # efficiently.
  # 
  # return: B=A*diag(d)*A^T
  # 
  # M Tsy 2021 May
  #--------------------------------------------------------------
  
  # # begin debug
  # A=matrix(c(1:9), nrow=3)
  # d=c(5:7)
  # BB = A %*% diag(d) %*% t(A)
  # # end debug
  
  n = length(d)
  
  D_cols = matrix(d, nrow=n, ncol=n) 
  D_rows = t(D_cols)  
  
  AD = A * D_rows   # Compute A \circ D_rows: Schur product 
  B = tcrossprod(AD, A)
  
  # max(abs(BB - B))
  
}