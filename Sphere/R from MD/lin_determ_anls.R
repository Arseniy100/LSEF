
lin_determ_anls = function(X_f, X_obs, H, BB, R, B_true){
  #-------------------------------------------------------------------------------------------
  # Perform multiple linear deterministic analyses with different
  # set of FG& OBS but with the same oobs oprt H and
  # the same error covariance matrices B & R.
  # 
  # That is, given
  # (1) Na versions of FG & OBS, 
  # (2) the obs oprt H 
  # (3) the available approximate FG-err CVM B, and
  # (4) the exactly known obs-err CVM R,
  # compute:
  # (i) the Na anlss X_a, and
  # (ii) the anls-err CVM (for which we need to utilize B_true)
  # 
  #    Methodology
  # 
  # Gain mx: 
  #    K = B H^T (H B H^T + R)^{-1}
  # Anls:
  #    X_a[1:nx, 1:Na] = X_f + K (X_obs - H X_f)
  # Anls-err CVM:
  # With the imprecise B, the anls-err CVM is
  # A = (I-KH) B_true (I-KH)^T + K R K^T
  # 
  #    Args:
  # 
  # X_f[1:nx, Na] - FG mx
  # X_obs[1:nx, Na] - OBS mx
  # H[1:n_obs, 1:nx] - obs oprt
  # BB - available (used in the anls) FG-err CVM
  # R[1:n_obs, 1:n_obs] - obs-err CVM
  # B_true - true FG-err CVM
  # 
  # Return: X_a, A
  # 
  # M Tsy 2020 June
  #-------------------------------------------------------------------------------------------
  
  Na    = dim(X_f)[2]  # nu of analyses to be computed
  nx    = dim(X_f)[1]
  n_obs = dim(X_obs)[1]
  
  # K
  
  K = BB %*% t(H) %*% solve(H%*% BB %*% t(H) + R)
  
  # x_a
  
  Y_obs = X_obs - H %*% X_f # obs increment (innovation)
  X_a = X_f + K %*% Y_obs
  
  # A = (I-KH) B_true (I-KH)^T + K R K^T
  
  I = diag(nx)
  ImKH = I - K %*% H
  
  KRKT = K  %*% R %*% t(K)
  
  A  = ImKH  %*% B_true %*% t(ImKH)  + KRKT
 
  return(list("X_a"=X_a, "A"=A))
}
 

