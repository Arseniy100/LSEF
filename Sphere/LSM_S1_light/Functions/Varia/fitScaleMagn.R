
fitScaleMagn = function(f_tofit, f_shape, xx, 
                        moments, a_max_times, w_a_fg, lplot){
  #-------------------------------------------------------------------------------------------
  # Fit the following two-parameter Magnitude-Scale mdl to the function 
  # f_tofit(x) defined on a regular grid: 
  # 
  #    f_tofit[nx] \approx A* f_shape(x/a)            (*)
  #
  # More precisely,  f_tofit  is a collection of functions, column-by column.
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
  # ------------------------------------------------------------
  # Having computed a01 and a12, we use them as "observations" along
  # with a_fg=1 and minimize
  # 
  # (a-a01)^2 + (a-a01)^2 + (a-a_fg)^2 --> min
  # 
  # The solution is 
  # 
  #   a = (a01 + a12 + w_a_fg * a_fg) / (2 + w_a_fg)      (*)
  # 
  # Then we check that  a  lies within  (a_fg/a_max_times , a_fg*a_max_times).
  # And fiunally, calc  A  from, e.g. (1):
  # 
  #   A = I/(aG)                                          (**)
  #      
  # ------------------------------------------------------------     
  #    Args
  # 
  # f_tofit[1:nx, 1:nf] - nf functions to be fitted defined on a regular nx-grid
  #  (nx is the function's support length and nf i sth enu of functions to be treated)
  # f_shape[1:nx] - shape of the spectrum:
  #           f_fit[1:nx] = A*f_shape(nx/a)
  #                defined of the SAME grid as f_tofit.
  # xx[1:nx] - the common grid to all functions here
  # moments - which moments to equate: "01" or "12" or "012"
  # a_max_times = max deviation of  a  in SHAPE in times 
  # w_a_fg - weight of the ||a-1||^2 weak constraint
  # lplot - plotting: draw plots only if TRUE
  # 
  # NB: Normally, all f-functions here need to decay to 0 well before x goes
  #     to the end of the grid.
  # 
  # return: f_fit[1:nx,1:nf] (the resulting fit), AA[1:nf], aa[1:nf].
  # 
  # Required functions:
  # evalFuScaledArg
  # 
  # M Tsy 2020 Oct
  #-------------------------------------------------------------------------------------------
  
  nx = dim(f_tofit)[1]
  nf = dim(f_tofit)[2]
  
  xx2=xx^2
  a_fg=1
  
  a_low = 1/a_max_times
  a_high= a_max_times
  
  f_fit_01 = f_tofit # init
  f_fit_12 = f_tofit # init
  f_fit = f_tofit # init
  
  #-----------------------------------------------------------------------
  # Calc I, I1, I2 -- x-dependent 
  # 
  # I= \int f_tofit(x) dx    = sum_1^nx f_tofit[i]  ...
  # I1= \int x f_tofit(x) dx 
  # I2= \int x^2 f_tofit(x) dx 
  
  II  = apply( f_tofit[,], 2, sum )
  II1 = apply( f_tofit[,], 2, function(z) sum(z*xx) ) 
  II2 = apply( f_tofit[,], 2, function(z) sum(z*xx2) )  
  
  #-----------------------------------------------------------------------
  # Calc  G, G1, G2 -- const
  # 
  # G = \int g(t) dt = sum g[nx]
  # G1= \int t g(t) dt = sum g[nx] *nx
  # G2= \int t^2 g(t) dt = sum g[nx] *nx^2
  
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
  
  II1_d_II = II1 / II
  aa_01 = (G/G1) * II1_d_II
  
  II2_d_II1 = II2 / II1
  aa_12 = (G1/G2) * II2_d_II1
  # AA_12 = (G2^2 / G1^3) * II1 / II2_d_II1^2
  
  # Calc the regularized solution:
  # a = (a01 + a12 + w_a_fg * a_fg) / (2 + w_a_fg)
  
  if(moments == "01"){
    aa = (aa_01 +       + w_a_fg * a_fg) / (1 + w_a_fg)
  }else if(moments == "12"){
    aa = (      + aa_12 + w_a_fg * a_fg) / (1 + w_a_fg)
  }else if(moments == "012"){
    aa = (aa_01 + aa_12 + w_a_fg * a_fg) / (2 + w_a_fg)
  }
  
  # plot(aa)
  # lines(aa_01)
  # lines(aa_12, col="blue")
  
  # Check that  a  lies within  (a_fg/a_max_times , a_fg*a_max_times).
  
  outliers = aa < a_low
  aa[outliers] = a_low  

  outliers = aa > a_high
  aa[outliers] = a_high 

  # Calc  A  from, e.g.   A = I/(aG) 
  
  AA = II / (aa * G)
  
  #-----------------------------------------------------------------------
  # Calc the resulting f_fit
  # f(nx) = A*f_shape(nx/a) 
  # f_fit[1:nx,1:nf]
  
  for (i in 1:nf){
    f_fit[,i] = AA[i] * evalFuScaledArg(xx, f_shape, aa[i])
  }

  #-----------------------------------------------------------------------
  # Diags
  
  if(lplot){
    inm=nx
    i_f=sample(c(1:nf), 1)
    mn=min(f_tofit[1:inm,i_f], f_fit[1:inm,i_f] )
    mx=max(f_tofit[1:inm,i_f], f_fit[1:inm,i_f] )
    plot(f_tofit[1:inm,i_f], main="f_tofit (blue), f_fit (red)", ylim=c(mn,mx),
         type="l", col="blue", xlab="x")
    lines(f_fit[1:inm,i_f], col="red")
  }

  #-----------------------------------------------------------------------  

  return(list("f_fit"=f_fit, "AA"=AA, "aa"=aa))
}
