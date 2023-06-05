
gen_homProc_fromSpectrum_S1 = function(bb, nrealiz) {
  #=========================================================================
  # Generate  nrealiz  realizations of a 
  # real-valued homogeneous (stationary) random process  xi[1:nx]  on S1
  # using the pre-specified spectrum,  bb[1:nx].
  # 
  # ssigma = sqrt(bb)
  # 
  # xi(x) = \sum_{n=1}^{n=nx} ssigma(n) alpha(n) exp(inx) 
  # 
  # whr    alpha(-n) = Conj(alpha(n))
  # 
  # Use cplx arithmetic w the standardd  fft.
  # 
  # return: xxi[1:nx, nrealiz]
  #
  # NB: nx sh.be even
  #     bb sh.be symmetric wrt the origin
  # M Tsy Feb 2022
  #=========================================================================

  nx = length(bb)
  nmax = nx /2
  nmaxp1 = nmax +1
  
  nn_half = c(0:nmax)
  nn = c(0:(nx-1))

  ssigma = sqrt(bb)
  
  xxi = matrix(0, nrow=nx, ncol=nrealiz)
  
  #-----------------------------------------------------------------
  # Checks. 
  # (1) bb sh.be >= 0
  
  eps = 1e-15
  if(min(bb) < eps){
    message("gen_homProc_fromSpectrum_S1 error. bb <0")
    print(bb)  
    stop("bb<0", call.=TRUE)
    return()
  }
  
  # (2) bb should be ``symmetric'' wrt the origin, ie
  # bb[2] = bb[nx]
  # n>1
  # bb[n] = [nx - n +2]
  
  nmaxp2 = nmax +2
  b_neg = c(bb[1], rev(bb[nmaxp2:nx]), bb[nmaxp1])
  
  if(max(abs(b_neg - bb[1:nmaxp1])) > eps){
    message("gen_homProc_fromSpectrum_S1 error. bb not symmetric wrt origin")
    print(bb)  
    stop("bb not symmetric wrt origin", call.=TRUE)
    return()
  }
  
  #-----------------------------------------------------------------
  # xxi
  
  for(irealiz in 1:nrealiz){
    #-------------------------------
    # 1. Generate the CN(0,1)  spec coeffs  alpha[n].
    # 
    # a) alpha[1], alpha[nmaxp1] ~ N(0,1) (real)
    # 
    # b) all the others are cplx:
    # alpha[np1] = alpha_re[np1] + 1i*alpha_im[np1],
    # whr
    # Var(alpha_re[np1]) = Var(alpha_im[np1]) =1/2
    
    alpha_re = rnorm(nmaxp1, mean=0, sd=1/sqrt(2))
    alpha_im = rnorm(nmaxp1, mean=0, sd=1/sqrt(2))
    
    alpha_re[1] = alpha_re[1] * sqrt(2)
    alpha_re[nmaxp1] = alpha_re[nmaxp1] * sqrt(2)  
    
    alpha_im[1] = 0
    alpha_im[nmaxp1] = 0
    
    alpha_half = alpha_re + 1i * alpha_im
    alpha = c(alpha_half, rev(Conj(alpha_half[2:nmax])))
    
    #-------------------------------
    # 2. xi
    
    xxi[,irealiz] = fft(alpha*ssigma, inverse = TRUE)
  }
  # max(abs(Im(xxi))) # OK
  xxi = Re(xxi)
  
  # plot(xxi[,3])
  #-----------------------------------------------------------------
  
  return(xxi)
}