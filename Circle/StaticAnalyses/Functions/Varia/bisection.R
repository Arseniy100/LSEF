
bisection = function(a, b, eps, fun, ...){
  
  #-----------------------------------------------------------------
  # Solve the eqn 
  #    fun(t) = 0           (*)
  # on the segment [a,b] by bisection.
  # eps is the allowable error
  #
  #   Method
  #   
  # 1) Find the number of iterations ro reach the eps error upper bound.
  # eps=(b-a)/2^n   ==>
  # 2^n = (b-a)/eps
  # n*log(2) = log( (b-a)/eps )
  # n = log( (b-a)/eps ) / log(2)
  # 
  # 2) Perform bisection.
  # 
  # return: solution of Eq(*)
  # 
  # M Tsy 2020 Jan
  #-----------------------------------------------------------------
  # debug:
  # fun(t)=t^2-0.5
  #b=1;  a=0; eps=0.00000001
  #fun=function(t) t^2-0.5
  ## end debug
  
  #browser()
  
  n = ceiling( log( (b-a)/eps ) / log(2) )
  
  x=a      ;  y=(a+b)/2 ;  z=b
  u=fun(x, ...) ;  v=fun(y, ...)  ;  w=fun(z, ...)
  
  for (i in 1:n){
  
    if(sign(u) != sign(v)){       # select the left  sub-intvl
      
      z=y ;  w=v
      
    }else if(sign(v) != sign(w)){ # select the right sub-intvl
      
      x=y ; u=v
    
    }else if(u == 0){
      y=x
      break
    }else if(v == 0){
      break
    }else if(w == 0){
      y=z
      break
    }else{
      message("bisection.  fun  does not change sign")
      print("u,w=")
      print(u)
      print(v)
      print("iter")
      print(i)
      stop("stop")
    }
    
    y=(x+z)/2 ;  v=fun(y, ...)
  }
  return(y)
}