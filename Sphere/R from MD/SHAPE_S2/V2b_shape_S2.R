V2b_shape_S2 = function(b_tofit, b_shape, moments){
  #-------------------------------------------------------------------------------------------
  # Fit the following prm mdl to b_tofit[n]
  #
  #    b_tofit[n] \approx A* b_shape(n/a)\equiv A*g(n/a)               (*)
  #
  #    Methodology
  #
  # The method of moments is used to retrieve A,a from (*):
  # Assuming, for simplicity of derivation, that n is a continuous variable z,
  # we have the 0th and 1st moment eqs:
  #
  #  I= \int b_tofit(z) dz   = A \int g(z/a) dz = Aa \int g(t) dt = Aa*G
  #
  #  I1= \int z b_tofit(z) dz = A \int z g(z/a) dz = Aa^2 \int t g(t) dt = Aa^2*G1
  #
  # we can also take the 2nd moment:
  #
  # I2= \int z^2 b_tofit(z) dz = A \int z^2 g(z/a) dz = Aa^3 \int t^2 g(t) dt = Aa^3*G2
  #
  # See function fitScaleMagnFu  for details.
  #
  # To compute G,G1,G2,I,I1,I2 we replace the integrals by the sums.
  #
  # NB: n >= 0 everwhere in the FITTING process, z \in [0, nmax].
  #
  #    Args
  #
  # b_tofit[ix=1:nx, i_n=1:nmaxp1], i_n=1,...,nmaxp1,  i_n := n+1 - spectrum to be fitted
  # b_shape[i_n=1:nmaxp1] - shape of the spectrum:
  #           b(n) = A*b_shape(n/a)
  # moments - which moments to equate: "01" or "12"
  #
  # return: b_fit[ix, i_n] (the resulting spectra at all grid points) - shaped as b_tofit
  #         The fitted values of A,a (AA, aa) at all grid points
  #
  # M Tsy 2020 Sep
  #       2021 Jan
  #-------------------------------------------------------------------------------------------

  nx     = dim(b_tofit)[1]
  nmaxp1 = length(b_shape)

  nmax = nmaxp1 -1
  nn=c(0:nmax)

  b_fit = b_tofit # init

  #-----------------------------------------------------------------------
  # Calc the resulting b_fit
  # f(n) = A*g(n/a)

  lplot=FALSE
  FIT = fitScaleMagn(t(b_tofit[,1:nmaxp1]), b_shape, nn, moments, lplot)
  b_fit[,1:nmaxp1] = t(as.matrix(FIT$f_fit))
  AA = FIT$AA
  aa = FIT$aa

  #-----------------------------------------------------------------------

  return(list("b_fit"=b_fit, "AA"=AA, "aa"=aa))
}
