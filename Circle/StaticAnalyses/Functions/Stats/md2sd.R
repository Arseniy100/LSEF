
md2sd = function(distrib="Gau", nu=1){
  
  #------------------------------------------------------------------
  # Calc the ratio of MD to SD:
  # 
  # MD=Mean Abs Deviation crom the mean
  # MD(X) := E|X-EX|
  # 
  # Calc md2sd for the specified distributions from the list:
  # "Gau"
  # "chi2"
  # ...
  # 
  # Args
  # 
  # For chi-square distr with nu degrees of freedom,
  # according to Wolfram, 
  # https://mathworld.wolfram.com/MeanDeviation.html or to
  # Johnson Kotz Balakrishnan Continuous Univar Distrib v1
  # 
  #   md = (2e)^(-nu/2) nu^(nu/2 +1) (nu+2) / gamma(2+ nu/2)
  #
  # whereas sd(chi2_nu) = sqrt(2*nu)
  #   
  # M Tsy 2021 Jan
  #------------------------------------------------------------------
  
  if(distrib == "Gau"){
    
    md2sd = sqrt(2/pi)
    
  }else if(distrib == "chi2"){
    
    if(nu < 100){
      nud2 = nu/2
      e = exp(1)
      sd=sqrt(2*nu)
      md2sd = (2*e)^(-nud2) * nu^(nud2 +1) * (nu+2) / gamma(2+ nud2) /sd
      
    }else{ # Gau apx
      
      md2sd = sqrt(2/pi)
    }
  }
  return(md2sd)
}