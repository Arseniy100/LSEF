
Cov2VarCor = function(B) {
  
  # B=CVM  --> v=diag(B), C=CRM
  # B can be complex (in contrast to cov2cor).
  # diag(B) cannot contain zeros.
  
  v=diag(B)
  std=sqrt(v)
  C=diag(1/std) %*% B %*% diag(1/std) 
  
  return(list("v"=v, "C"=C))
}
