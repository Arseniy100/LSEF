# Call the main program  truth_worlds_and_filters
# several times (replicates), collect filters' RMSE, calc their mean-square RMSEs
# and bootstrap tolerance stripes.
# 
# M Tsy 2023 Mar

library(plot3D)

source('./truth_worlds_and_filters.R')

setup <- read.table('./setup.txt', sep = ';')

n_realiz = setup[setup$V1 == "n_realiz", 2]
seed     = setup[setup$V1 == "seed",     2]

n_realiz
seed

TRUTH_RMSE  = c(1:n_realiz)
KF_fRMSE    = c(1:n_realiz)
Var_fRMSE   = c(1:n_realiz)
EnKF_fRMSE  = c(1:n_realiz)
EnVar_fRMSE = c(1:n_realiz)
LSEF_fRMSE  = c(1:n_realiz)

for(i_realiz in 1:n_realiz){
  message("     irealiz = ", i_realiz)
  
  #......................................
  RMSEs = truth_worlds_and_filters(seed)
  #......................................
  
  TRUTH_RMSE [i_realiz] = RMSEs$TRUTH_RMSE
  KF_fRMSE   [i_realiz] = RMSEs$KF_fRMSE
  Var_fRMSE  [i_realiz] = RMSEs$HHBEF_fRMSE[1]
  EnKF_fRMSE [i_realiz] = RMSEs$HHBEF_fRMSE[2]
  EnVar_fRMSE[i_realiz] = RMSEs$HHBEF_fRMSE[3]
  LSEF_fRMSE [i_realiz] = RMSEs$LSEF_fRMSE
  
  seed = seed + 12341
}

if(max(KF_fRMSE)   == 0) KF_fRMSE[] = 0
if(max(Var_fRMSE)  == 0) Var_fRMSE[] = 0
if(max(EnKF_fRMSE) == 0) EnKF_fRMSE[] = 0
if(max(EnVar_fRMSE)== 0) EnVar_fRMSE[] = 0
if(max(LSEF_fRMSE) == 0) LSEF_fRMSE[] = 0

RMStable = cbind(KF_fRMSE, Var_fRMSE, EnKF_fRMSE, EnVar_fRMSE, LSEF_fRMSE) # [i_realiz, i_flt]
print(RMStable)

RMSEs_realizMeanSquare = apply(RMStable, 2, function(t) sqrt(mean(t^2)))
print(RMSEs_realizMeanSquare)

# REE: Relative Excess errors
# REE = (e - e_KF)/e_KF

n_flt = 5
n_flt_exam = n_flt -1 # Var, EnKF, EnVar, LSEF

if(max(KF_fRMSE) > 0){
  REE_mean = (RMSEs_realizMeanSquare[2:n_flt] - RMSEs_realizMeanSquare[1]) / RMSEs_realizMeanSquare[1]
  message("REE_mean")
  print(REE_mean)
}

#----------------------------------------
# Bootstrap REE

if(n_realiz > 1 & max(KF_fRMSE) > 0){

  n_boo = 100        # bootstrap sample size
  REE_boo = matrix(0, nrow = n_realiz, ncol = n_flt_exam)
  
  REE_boo = matrix(0, nrow = n_boo, ncol = n_flt_exam)
  for (i_boo in 1:n_boo){
    ii_boo = sample(c(1:n_realiz), n_realiz, replace = TRUE)
    RMS_boo = RMStable[ii_boo,]
    RMS_realizMeanSquare = apply(RMS_boo, 2, function(t) sqrt(mean(t^2)))
    REE_boo[i_boo,] = (RMS_realizMeanSquare[2:n_flt] - RMS_realizMeanSquare[1]) / RMS_realizMeanSquare[1]
  }
  
  apply(REE_boo, 2, mean)
  apply(REE_boo, 2, sd)
  
  # 90% confidence intervals
  
  p=0.1
  omp=1-p
  
  bounds = apply(REE_boo, 2, function(t) quantile(t, probs = c(p, omp)))
  low_bounds =bounds[1,]
  upp_bounds =bounds[2,]
  
  for(i_flt in 1:n_flt_exam){
    message("i_flt=", i_flt, " low_bound = ", signif(low_bounds[i_flt], 4), 
            " upp_bound = ", signif(upp_bounds[i_flt],4))
  }
  
  message("Bounds")
  print(low_bounds)
  print(upp_bounds)
  
  #----------------------------------------------
  # Plotting REE
  
  LSEF  = REE_mean[4]
  enkf  = REE_mean[2]
  var   = REE_mean[1]
  envar = REE_mean[3]
  
  mx = max(upp_bounds, REE_mean)
  
  filters = c("Var", "EnKF", "EnVar", "LSEF")
  
  plot(REE_mean, xaxt = "n", type="p", pch=8, ylim = c(0,mx),
       xlab = "Filters", ylab = "Relative Excess RMS Error", cex.lab=1.3)
  lines(low_bounds, type="p", pch=2)
  lines(upp_bounds, type="p", pch=6)
  axis(1, at=seq_along(filters), labels = filters, cex.axis=1.3, xlab = "Filters")
  abline(h=0)
  
  #----------------------------------------------
  # Write to a file
  
  results = list(REE_mean=REE_mean, low_bounds=low_bounds, upp_bounds=upp_bounds)
  
  REE_file = paste0("./REE.txt")
  unlink(REE_file)
  sink(REE_file) # , append=TRUE)
  
  cat("REE_mean")
  print(REE_mean)
  
  cat("\n")
  
  cat("low_bounds\n")
  cat(low_bounds)
  
  cat("\n")
  
  cat("upp_bounds\n")
  cat(upp_bounds)
  
  sink()
}


