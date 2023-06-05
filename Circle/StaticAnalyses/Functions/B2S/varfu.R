varfu = function(spe){
  
  # Find VARIANCE from the non-neg part of the spectrum
  # 
  #   Args
  #   
  # spe[1:nmaxp1] - non-neg part of the spectrum
  # 
  # M Tsy 2022 Aug
  
    N = length(spe)
    varfu = spe[1] + 2*sum(spe[2:(N-1)]) + spe[N]
  }