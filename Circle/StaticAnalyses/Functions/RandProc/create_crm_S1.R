
create_crm_S1 <- function(n, L, crftype, nu=1){

# Gen CRM on S^1
#
# n - nu of point of the grid
# L - len scale (rad.)
# crftype - character variable. Can attain values:
#  "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern".
# nu ~ optional ~ used for "Bessel" & "Matern" & "Cauchy"
#      For "Bessel", nu >0.5
#      For "Bessel", nu > (d-2)/2, where the crf is defined in R^d.
# Return: crm
#
# M Tsy Mar 2017, Jan 2020 (from km to rad.)
 
  
  #=====
  # Debug
  #n=64
  #L = 2000/6370
  #crftype="Matern"
  #=====

  crm  <- matrix(ncol = n, nrow = n)
  ds=2*pi/n
  
  omega=0.9 # for ecos: sh.be <1 on R2

  for(i in (1:n)) {
    for(j in (1:n)) {
      d=abs(i-j)
      if ((d> floor(n/2))) {d <- n-d}
      rho <- ds * d   # rad.
      r=2*sin(rho/2) # chordal dist, unitless
      x=r/L
      
      crm[i,j] = crfs_isotr(crftype, nu, x)
    }
  }
  V=crm[1,1]
  crm=crm/V
  
  return(crm)
}