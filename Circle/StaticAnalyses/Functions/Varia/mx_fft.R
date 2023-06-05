mx_fft = function(A, direction){
  
  #-------------------------------------------------------------------------
  # Perform matrix FFT of A (n by n):
  # F_A= F A F*
  # Forward  DFFT if direction="f"
  # Backward DFFT if direction="b"
  #
  # NB: If F(m,j)=exp( -i (2pi/n) m j), 
  # where m is the wvn and j labels the spatial grid point,
  # then the fwd DFT is 
  # xi=F \cdot x / n
  # and the bckw DFT is 
  # x=F* \cdot xi
  #
  # Forw:
  #-----------------------------
  #     F A F* = F (F A*)*
  #-----------------------------
  # (1) Apply DFFT to the rows of conj(A) and place the results as rows of the cmplx mx FAaT:
  #     FAaT = (FA*)^T
  # (2) Cmplx conjugate FAaT
  #     FAaa = (FA*)* == A F*
  # (3) Apply DFFT to the cols of FAaa
  #
  # Backw:
  #-----------------------------
  # A=Finv F_A Finv* = Finv (Finv F_A*)*
  #-----------------------------
  # Do the same but with the F* (backw FFT) instead of F (forw DFFT)
  #
  # Return FA (NB: cplx-valued!)
  #
  # M Tsyr 5 Jul 2017
  # Tested.
  #-------------------------------------------------------------------------
  
  n=dim(A)[1]
  
  if(direction == "f") {
    inv=FALSE
    denom=n
  }else{
    inv=TRUE
    denom=1
  }
  
  FAaT = matrix(0+0i, nrow=n, ncol=n) # (F * A_adjoint)_transpose
  FAaa = matrix(0+0i, nrow=n, ncol=n) # (F * A_adjoint)_adjoint
  FA   = matrix(0+0i, nrow=n, ncol=n) # the final result F_A

  for (j in 1:n){
    FAaT[j,]=fft(Conj(A[j,]), inverse=inv) /denom
  }
  FAaa=Conj(FAaT)
  
  for (j in 1:n){
    FA[,j]=fft(FAaa[,j], inverse=inv) /denom
  }
 
  return(FA)
} 