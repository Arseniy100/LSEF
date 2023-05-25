
gen_LogitGauStatioProc_S1 = function(n, add_xi, mult_xi, kappa, len_scale_eta, 
                                     crftype, nu, sqrtCRM = NULL) {
  #=========================================================================
  # Generate a realization xi[1:n] of a homogeneous logit-Gaussian random process on S1(R_km).
  # A logit-Gaussian random process xi[x] is, by definition,  such that logit(xi[x]) ~ Gau.
  #
  # ==> xi[x] = add_xi + mult_xi * logistic( eta[x] ),
  #
  # whr eta[x] is a Stationary (homogeneous) N(mean=0, SD = log(kappa)) Gau proc 
  #     w len scale len_scale_eta and 
  #     logit(x) is the inverse logistic function,
  #     logistic(y) is defined as follows:
  #
  # g(x,b) = logistic(x,b) = (1+exp(b))/(1+exp(b-x)).
  #
  # NB: 
  # x \to  -\infty  ===>  g(x,b) \to  0
  # x \to  +\infty  ===>  g(x,b) \to  1 + exp(b)
  # 
  #  (by default, b=1)
  #     
  # SD(eta[x]) = log(kappa)
  #
  # So, the spatial crls come from eta, but the pointwise distrib comes from 
  # the transform fu g().
  #
  #   Args 
  #
  # mult_xi - location parameter (median) of the pointwise distribution of xi
  # add_xi - additive (to prevent xi to be too close to zero, if needed)
  # kappa = exp(SD(eta)) - the scale parameter of xi: kappa is, roughly, how many times
  #                        xi, typically, differs from median(xi).
  #      (kappa=1...4;  kappa=1: SD(xi)=0; kappa=2: nrm)
  # len_scale_eta - len scale of eta[x] (rad.)
  # crftype - character variable. Can attain values:
  #      "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern".
  # nu - optional ~ used for "Bessel" & "Matern" & "Cauchy"
  #      For "Bessel", nu >0.5
  #      For "Bessel", nu > (d-2)/2, where the crf is defined in R^d.
  # sqrtCRM - sqrt(CRM): if =NULL, it's computed within this function,
  #                      if not, ist's taken from this argument and 
  #                      passed to  gen_homProc_S1,  where it's used
  # 
  # NB: If len_scale_eta>rekm, then we Return a properly distributed constant.
  # NB: If L=0, then we Return a white noise with the same pointwise distrib!
  #
  # M Tsy Jan 2020, Feb 2021 (sqrtCRM argument added)
  #=========================================================================

    
  # Debug
  # n=360
  # add_xi=0
  # mult_xi=2
  # kappa=4
  # len_scale_eta = 1000
  # crftype="AR2"
  # nu=1
  #=====

  # Generate eta
  
  mean_eta = 0
  sd_eta   = log(kappa)
 
  homProc = gen_homProc_S1(n, mean_eta, sd_eta, len_scale_eta, crftype, nu, sqrtCRM)
  eta = homProc$xi
  sqrtCRM = homProc$sqrtCRM
  
  # plot(eta)
  #  sd(eta)
  #  sd_eta
  # image2D(sqrtCRM)
   
  xi  = add_xi + mult_xi * logistic(eta)  
    
  #median(xi)
  #median(abs(xi - median(xi)))
  #plot(xi)
  
  return(list("xi"=xi, "sqrtCRM"=sqrtCRM))
}