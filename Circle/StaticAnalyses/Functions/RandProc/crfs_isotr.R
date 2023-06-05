crfs_isotr <- function(crftype, nu=1, x){

  # Specify an isotropic crf as a function of the unitless distance from 
  # the list of crfs defined by crftype:
  # crftype - character variable. Can attain values:
  #  "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern".
  # nu ~ optional ~ used for "Bessel" & "Matern" & "Cauchy"
  #      For "Bessel", nu >0.5
  #      For "Bessel", nu > (d-2)/2, where the crf is defined in R^d.
  #      For Matern, nu defines Differentiability of the crf & xi: xi is nu-1 times ms differentiable:
  #                  nu=1/2 crftype == exp
  #                  nu=1   crftype == x*K_1(x)
  #                  nu=3/2 crftype == AR2
  #                  nu=5/2 crftype == AR3
  #                so, use nu=1 by default to get smth different from the other crf types.
  #
  # x - unitless distance, x \ge 0.
  #
  # M Tsy Mar 2017
  
  #=====
  # Debug
  #crftype="Matern"
  #=====
  
  omega=0.9 # for ecos: sh.be <1 on R2
  eps=.Machine$double.eps
  
  if(crftype == "exp"){
    crf <- exp(-x)
    
  }else if(crftype == "AR2"){
    crf <- (1 + x)*exp(-x)
    
  }else if(crftype == "AR3"){
    crf <- (1 + x + (x^2)/3)*exp(-x)
    
  }else if(crftype == "Gau"){
    crf <- exp(-(x^2)/2)
    
  }else if(crftype == "Cauchy"){
    crf <- 1/ ( (1+x^2)^nu )
    
  }else if(crftype == "ecos"){
    crf <- exp(-x)* cos(omega*x)
    
  }else if(crftype == "Matern"){
    
    if(x < eps){  # use an asymptotic formula for x \to 0 [Wiki]
      # For nu>0, K(x,nu) ~ Gamma(nu)/2 * (2/x)^nu
      # x cancels =>
      crf <- gamma(nu)/2 * 2^nu
    }else{
      crf <- x^(nu) * besselK(x, nu)
    }
    
  }else if(crftype == "Bessel"){
    
    if(x < eps){  # use an asymptotic formula for x \to 0 [Wiki]
      # J(x,nu) ~ 1/Gamma(1+nu) * (x/2)^nu
      # x cancels =>
      crf <- 1/(gamma(1+ nu) * 2^nu)
    }else{
      crf <- x^(-nu) * besselJ(x, nu)
    }
    
  }else{
    message("invalid crftype")
    print(crftype)  
    stop("invalid crftype")
  }
 
  return(crf)
}