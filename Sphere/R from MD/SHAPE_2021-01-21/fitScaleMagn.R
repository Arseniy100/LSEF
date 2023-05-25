
fitScaleMagn = function(f_tofit, f_shape, xx, moments, lplot){
  #-------------------------------------------------------------------------------------------
  # Fit the following two-parameter Magnitude-Scale mdl to the function 
  # f_tofit(x) defined on a regular grid: 
  # 
  #    f_tofit[n] \approx A* f_shape(x/a)            (*)
  #
  # More precisely, f_tofit is a collection of functions, column-by column.
  # Similarly, so is the output f_fit.
  # 
  # 
  #    Methodology
  #    
  # The method of moments is used to retrieve A,a from (*):
  # We have the 0th, 1st, and 2nd moment eqs:
  # 
  #  I= \int f_tofit(x) dx = A \int f_shape(x/a) dx = Aa \int f_shape(t) dt = Aa*G
  # 
  #  I1= \int x f_tofit(x) dx = A \int x g(x/a) dx = Aa^2 \int t f_shape(t) dt = Aa^2*G1
  # 
  # I2= \int x^2 f_tofit(x) dx = A \int x^2 g(x/a) dx = Aa^3 \int t^2 f_shape(t) dt = Aa^3*G2
  # 
  # or
  # 
  #    AaG = I                           (1)
  #    Aa^2 G1 = I1                      (2)
  #    Aa^3 G2 = I2                      (3)
  #   
  # From (1),(2), we have 
  #
  #        G*I1
  # a01 = ------
  #        G1*I
  #      
  #    
  #        G1*I^2
  # A01 = --------
  #        G^2*I1    
  #  
  # From (2),(3), we have
  #
  #        G1*I2
  # a12 = ------
  #        G2*I1
  #      
  #    
  #        I1^3 * G2^2
  # A12 = -------------
  #        G1^3 * I2^2    
  # 
  # To compute G,G1,G2,I,I1,I2 we replace the integrals by the sums.
  #      
  #    Args
  # 
  # f_tofit[1:n, 1:nf] - nf functions to be fitted defined on a regular n-grid
  #  (n is the function's support length and nf i sth enu of functions to be treated)
  # f_shape[1:n] - shape of the spectrum:
  #           f_fit[1:n] = A*f_shape(n/a)
  #                defined of the SAME grid as f_tofit.
  # xx[1:n] - the common grid to all functions here
  # moments - which moments to equate: "01" or "12" 
  # lplot - plotting: draw plots only if TRUE
  # 
  # NB: Normally, all f-functions here need to decay to 0 well before x goes
  #     to the end of the grid.
  # 
  # return: f_fit[1:n,1:nf] (the resulting fit), AA, aa.
  # 
  # M Tsy 2020 Oct
  #-------------------------------------------------------------------------------------------
   
  n = dim(f_tofit)[1]
  nf = dim(f_tofit)[2]
  
  xx2=xx^2
  
  f_fit = f_tofit # init
  
  #-----------------------------------------------------------------------
  # Calc I, I1, I2 -- x-dependent 
  # 
  # I= \int f_tofit(x) dx    = sum_1^n f_tofit[i]  ...
  # I1= \int x f_tofit(x) dx 
  # I2= \int x^2 f_tofit(x) dx 
  
  II  = apply( f_tofit[,], 2, sum )
  II1 = apply( f_tofit[,], 2, function(z) sum(z*xx) ) 
  II2 = apply( f_tofit[,], 2, function(z) sum(z*xx2) )  
  
  #-----------------------------------------------------------------------
  # Calc  G, G1, G2 -- const
  # 
  # G = \int g(t) dt = sum g[n]
  # G1= \int t g(t) dt = sum g[n] *n
  # G2= \int t^2 g(t) dt = sum g[n] *n^2
  
  G = sum(f_shape)
  G1 = sum(f_shape * xx)
  G2 = sum(f_shape * xx2)
  
  #-----------------------------------------------------------------------
  # Calc A,a
  #
  #        G*I1
  # a01 = ------
  #        G1*I
  #      
  #    
  #        G1*I^2
  # A01 = --------
  #        G^2*I1    
  #  
  # From (2),(3), we have
  #
  #        G1*I2
  # a12 = ------
  #        G2*I1
  #      
  #    
  #        I1^3 * G2^2
  # A12 = -------------
  #        G1^3 * I2^2    
  # 
  
  if(moments == "01"){
    II1_d_II = II1 / II
    aa = (G/G1) * II1_d_II
    AA = (G1/G^2) * II / II1_d_II
    
  }else if(moments == "12"){
    II2_d_II1 = II2 / II1
    aa = (G1/G2) * II2_d_II1
    AA = (G2^2 / G1^3) * II1 / II2_d_II1^2
  }
  
  #-----------------------------------------------------------------------
  # Calc the resulting f_fit
  # f(n) = A*f_shape(n/a) 
  
  for (i in 1:nf){
    tt = xx / aa[i] # args of f_shape, the scaled grid, i-th column
    
    # intpl f_shape from the grid xx to the grid  yy
    
    f_fit[,i] = LinIntpl(xx, AA[i]*f_shape, tt)
  }
  
  #-----------------------------------------------------------------------
  # Diags
  
  if(lplot){
    inm=n
    i=sample(c(1:n), 1)
    mx=max(f_tofit[1:inm,i], f_fit[1:inm,i] )
    plot(f_tofit[1:inm,i], main="f_tofit (blue), f_fit (red)", ylim=c(0,mx),
         type="l", col="blue", xlab="x")
    lines(f_fit[1:inm,i], col="red")
  }

  #-----------------------------------------------------------------------  

  return(list("f_fit"=f_fit, 
              "Magnitudes"=AA, 
              "Scales"=aa 
              ))
}
