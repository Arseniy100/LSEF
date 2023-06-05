symm_pd_mx_rglrz = function(A, cond_max){
  
  #------------------------------------------------------------
  # Regularize symmetric pos.definite  mx A
  # by trimming its ei-values from below, coercing them 
  # all to be = ei_val_max / cond_max.
  # 
  # Return:  A_rgl
  # 
  # M Tsy 2021 May
  #------------------------------------------------------------
    
  ei=eigen(A)
  eval=ei$values
  
  if(is.complex(eval) == TRUE){
    message("symm_pd_mx_sqrt")
    print(eval)
    stop("Non-symmetric input mx")
  }
  
  eval_max=max(eval)
  eval_min = eval_max /cond_max
  
  eval_rgl = eval
  eval_rgl[eval < eval_min] = eval_min
  
  evec=ei$vectors
  
  A_rgl = evec %*% diag(eval_rgl) %*% t(evec)
  A_rgl = mult_squareMx_ADAT(evec, eval_rgl)
}

  

