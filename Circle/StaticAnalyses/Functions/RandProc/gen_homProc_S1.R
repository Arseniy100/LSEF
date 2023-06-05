
gen_homProc_S1 = function(n, mean_xi, sd_xi, len_scale, crftype, nu,
                          sqrtCRM = NULL) {
  #=========================================================================
  # Generate a realization xi[1:n] of a homogeneous random process on S1
  # using the pre-specified covariance function, mean, and sd.
  #
  # mean_xi=E[xi]
  # sd_xi=s.d.[xi] 
  # len_scale - len scale (rad.)
  # crftype - character variable. Can attain values:
  #  "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern".
  # nu - optional ~ used for "Bessel" & "Matern" & "Cauchy"
  #      For "Bessel", nu >0.5
  #      For "Bessel", nu > (d-2)/2, where the crf is defined in R^d.
  # sqrtCRM - sqrt(CRM): if =NULL, it's computed within this function,
  #                     if not, ist's taken from this argument
  #
  # NB: If len_scale>2*pi, then we return an N(mean_xi, sd_xi) constant.
  # NB: If L=0, then we return an N(mean_xi, sd_xi) white noise!
  # 
  # 
  # return: xi, sqrtCRM
  #
  # M Tsy Jan 2020, Feb 2021 (sqrtCRM argument added)
  #=========================================================================

  
  # Debug
  # n=512
  # mean_xi=2
  # sd_xi=2
  # len_scale = 0.2
  # crftype="AR2"
  # nu=1
  #=====

  xi <- rep(0,n) # result
  
  #-----------------------------------------------------------------
  # Checks

  if(crftype != "exp" &
     crftype != "AR2" &
     crftype != "AR3" & 
     crftype != "Gau" & 
     crftype != "Cauchy" & 
     crftype != "ecos" & 
     crftype != "Bessel" & 
     crftype != "Matern"){
    message("gen_homProc_S1. Invalid crftype")
    print(crftype)  
    stop("invalid crftype", call.=TRUE)
    return()
  }

  #-----------------------------------------------------------------
  # Field structure
  
  if(len_scale > 2*pi){      # constant field
    field_stru <- "constant" 
  }else if(len_scale == 0){  # w/noise
    field_stru <- "wnoise" 
  }else{                # "normal" case
    field_stru <- "field" 
  }
 
  #-----------------------------------------------------------------
  # Generate the N(0,1) pre-xi process, xi_N01
  
  wnoise <- rnorm(n, mean=0, sd=1)
  
  if(field_stru == "constant"){     # then at the log scale, we specify an N(0,1) const field here
    xi_N01  <- rep(wnoise[1], times=n)
    
  }else if(field_stru == "wnoise"){ # then the N(0,1) w/noise field
    xi_N01  <- wnoise
    
  }else if(field_stru == "field"){  # then the correlated N(0,1) field
     
  # Generate the CRM of xi
  
    if(is.null(sqrtCRM)){
      CRM <- create_crm_S1(n, len_scale, crftype, nu)
      sqrtCRM <- symm_pd_mx_sqrt(CRM)$sq
    }

    xi_N01  <- c(sqrtCRM %*% wnoise)
  }
  
  #-----------------------------------------------------------------
  # From xi_N01 - to xi:
  #           E[xi]    = log(mode_xi)
  #           s.d.(xi) = log(mode_xi)
  
  xi[] <- mean_xi + sd_xi * xi_N01[]
  
  #sd(xi)
  #mean(xi)
  #plot(xi)
  #-----------------------------------------------------------------
  
  return(list("xi"=xi, "sqrtCRM"=sqrtCRM))
}