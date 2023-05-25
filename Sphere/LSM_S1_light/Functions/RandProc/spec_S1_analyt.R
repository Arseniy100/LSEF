
spec_S1_analyt = function(crftype, nmax, L){
  # Compute spectrum of the convolution of the white noise 
  # with the kernel f(|x|)=crf(|x|/L) 
  # 
  # cos Fourier transf: Ditkin, Prudnikov 1961, 7.47
  # 
  # crftype = "AR2" or "exp" allowed at the moment
  # Fourier transform of crf
  # L - len scale, rad.
  # 
  # return: b_nrm_n (normalized spectrum - up to a const)
  # 
  # M Tsy 2021
  
  if(crftype != "exp" & crftype != "AR2"){
    message("spec_S1_analyt. Wrong crftype")
    stop
  }
  
  nmaxp1=nmax+1
  nnp1=1:nmaxp1
  nn = nnp1 -1
  bb=c(1:nmaxp1)
  
  zz2=(L*nn)^2
  
  if(crftype == "AR2"){
    bb[nnp1] = (1- zz2)^2 / (1+ zz2)^4
  }else if(crftype == "exp"){
    bb[nnp1] = 1/ (1+ zz2)^2
  }
  nrm=bb[1]
  bb=bb/nrm
}

