

V2E_shape_S2 = function(E_tofit, E_shape, moments, a_max_times, w_a_fg){
  #-------------------------------------------------------------------------------------------
  # Fit the following prm mdl to E_tofit[n] 
  # 
  #    E_tofit[n] \approx A* E_shape(n/a)\equiv A*g(n/a)               (*)
  #
  #    Methodology
  #    
  # The method of moments is used to retrieve A,a from (*):
  # Assuming, for simplicity of derivation, that n is a continuous variable z,
  # we have the 0th and 1st moment eqs:
  # 
  #  I= \int E_tofit(z) dz   = A \int g(z/a) dz = Aa \int g(t) dt = Aa*G
  # 
  #  I1= \int z E_tofit(z) dz = A \int z g(z/a) dz = Aa^2 \int t g(t) dt = Aa^2*G1
  # 
  # we can also take the 2nd moment:
  # 
  # I2= \int z^2 E_tofit(z) dz = A \int z^2 g(z/a) dz = Aa^3 \int t^2 g(t) dt = Aa^3*G2
  # 
  # See function fitScaleMagnFu  for details.
  # 
  # To compute G,G1,G2,I,I1,I2 we replace the integrals by the sums.
  #     
  # NB: n >= 0 everwhere in the FITTING process, z \in [0, nmax].
  # 
  #    Args
  # 
  # E_tofit[ix=1:nx, i_n=1:nmaxp1], i_n=1,...,nmaxp1,  i_n := n+1 - spectrum to be fitted
  # E_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*E_shape(n/a) 
  # moments - which moments to equate: "01" or "12" or "012"
  # a_max_times = max deviation of  a  in SHAPE in times (5--10 nrm)
  # w_a_fg - weight of the ||a-1||^2 weak constraint (0.2 nrm)
  # 
  # return: E_fit[ix, i_n] (the resulting spectra at all grid points) - shaped as E_tofit
  #         The fitted values of A,a (AA, aa) at all grid points
  # 
  # M Tsy 2020 Sep 
  #       2021 Jan Mar
  #-------------------------------------------------------------------------------------------
  
  nx     = dim(E_tofit)[1]
  nmaxp1 = length(E_shape)
   
  nmax = nmaxp1 -1
  nn=c(0:nmax)

  E_fit = E_tofit # init

  #-----------------------------------------------------------------------
  # Calc the resulting E_fit
  # f(n) = A*g(n/a) 
  
  lplot=F
  FIT = fitScaleMagn(t(E_tofit[,1:nmaxp1]), E_shape, nn, 
                     moments, a_max_times, w_a_fg, lplot)
  E_fit[,1:nmaxp1] = t(as.matrix(FIT$f_fit))
  AA = FIT$AA
  aa = FIT$aa
  
  #-----------------------------------------------------------------------  
  # Test
  # plot(E_tofit[1,1:nmaxp1], type="l", main="E_tofit (black), E_fit (red)")
  # lines(E_fit[1,1:nmaxp1], col="red")
  #-----------------------------------------------------------------------  

  return(list("E_fit"=E_fit, "AA"=AA, "aa"=aa))
}
