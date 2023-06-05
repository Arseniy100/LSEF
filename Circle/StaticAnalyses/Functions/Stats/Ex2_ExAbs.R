Ex2 = function(x){
  
  # Calc Variance without subtracting the mean:
  # Return M(|x|^2), whr M is the mean.
  # Works for cplx x as well.
  # M Tsy 2020 Apr
  
  Ex2 = mean(abs(x)^2)
  return(Ex2)
  
}

ExAbs = function(x){
  
  # Calc Mean Abs Value of the vector  x:
  # Return M(|x[]|), whr M is the mean.
  # Works for cplx x as well.
  # M Tsy 2020 May
  
  ExAbs = mean(abs(x))
  return(ExAbs)
  
}

