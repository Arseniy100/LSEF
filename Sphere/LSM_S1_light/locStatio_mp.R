# Explore non-stationary parametric spatial convolutions (moving-average) models 
# and thier estimators on S1.
# 
# =Generate the true local spectra.
# =Generate the truth and the ensm.
# =Create the spectral bands of the multi-scale bandpass filter.
# =E2B: Apply the multi-scale bandpass filter to the ensm members.
# =B2S: From band variances, restore thhe whole local spectrum (at all grid points indep-ly)
# =Compare the estimated spectra w the true (specified) ones. The same with cvfs.
# =From the local spectra, compute W and an estm the true B: B_LSM=W*W^T.
# =Analysis: compare B=W_SLM* (W_SLM)^T   with   S_lcz  in computing the 
#           deterministic anls x_a. Assess the accuracy of x_a as compared w truth.
#
#--------------
# Notation:
# 
# Averaging: suffixes
# _Ms -- spatial sample mean
# _Me -- ensm sample mean
# _Mb -- sample mean over the bands
# _Mw -- mean over the wavenumbers
# _Mt -- mean over the True model
# _Mr -- mean over the replicates of the whole scheme
# _MB - mean over the Bootstrap sample
# 
# _A* - mean abs
# _AE - MAE
# _AD - Mean abs deviation
# _ME - mean error (bias)
# _V* - variances (understood here always as Ex(x^2), i.e. w/o subtract.the mean)
# _S* - sums
# _D* - st.devs
# _R* - RMSs
# _RE - RMSE
# _Q - mean squared
# _QE - mean squared Error
# _DE - st.dev. Error
# 
# _T - theoretical moment (eg _TD - theoretical SD)
# 
# _O* - smoothing
# 
# _MA* - mean-abs value
# ---
# band - pass band
# 
# ----------------
# With  B2S_SVshape, first, run with  CALIBRATE = T  and n_repl=50  
# to calc the shape of the mean spectrum.
# Then, switch to  CALIBRATE = F
# 
# With  B2S_NN:
#  if LEARN = T, run the code with n_repl=50 to estm weights of the neural network.
#  if LEARN = F, read NN from a file
#
# M Tsy 2020, 2021, 2022
#*****************************************************
library(attempt)
library(plot3D)
#library(Matrix)
library(pracma)
library(nnls)
library(stats)
library(mixmeta)
library(moments)

source('./Functions/Anls/gen_obs.R')
source('./Functions/Anls/lin_determ_anls.R')

source('./Functions/B2S/B2S_lines.R')
source('./Functions/B2S/B2S_monot.R')
source('./Functions/B2S/B2S_param.R')
source('./Functions/B2S/B2S_shape.R')
source('./Functions/B2S/B2S_SVshape.R')
source('./Functions/B2S/bmeanBand_intpl.R')
source('./Functions/B2S/V_prm_misfit.R')

source('./Functions/CreateBands/CreateExpqBands.R')
source('./Functions/CreateBands/Omega_SVD.R')
source('./Functions/CreateBands/tranfuBandpassExpq.R')

source('./Functions/E2B/E2B.R')

source('./Functions/LinAlgebra/BshiftMx_S1.R')
source('./Functions/LinAlgebra/Cov2VarCor.R')
source('./Functions/LinAlgebra/mult_squareMx_ADAT.R')
source('./Functions/LinAlgebra/symm_cvm_row.R')
source('./Functions/LinAlgebra/symm_pd_mx_sqrt.R')
source('./Functions/LinAlgebra/Superdiagonals.R')

source('./Functions/LSM/Sigma2WB.R')
source('./Functions/LSM/createNstatioWSigma_S1.R')

source('./Functions/RandProc/create_crm_S1.R')
source('./Functions/RandProc/crfs_isotr.R')
source('./Functions/RandProc/CvfSpeTransfPair_d1_3.R')
source('./Functions/RandProc/gen_homProc_fromSpectrum_S1.R')
source('./Functions/RandProc/SpatAveCrf_S1.R')
source('./Functions/RandProc/spec_S1_analyt.R')
source("./Functions/RandProc/ThreePointSmoo_S1.R")
source('./Functions/RandProc/ThreePointSmoo_segment.R')
source("./Functions/RandProc/tranfuSmoo_lowpass_dscrWvns.R")

source('./Functions/Stats/Ex2_ExAbs.R')
source('./Functions/Stats/md2sd.R')

source("./Functions/Varia/bisection.R")
source('./Functions/Varia/cyclic_shift_S1.R')
source('./Functions/Varia/evalFuScaledArg.R')
source('./Functions/Varia/fitScaleMagn.R')
source('./Functions/Varia/LinIntpl.R')
source("./Functions/Varia/logistic.R")
source('./Functions/Varia/monotSmooFit.R')

#--------------------------------------------------------------
# external prms

nx = 360; nmax=floor(nx/2); dx=2*pi/nx # nx sh.be a highly composite number (for fft)
ne = 10  # ensm size
kappa = 2 # nstatio strength (NSS); =1...4;  kappa=1: statio; kappa=2 nrm
NSL = 3 # Non-stationarity Length (relative to L_xi). NSL=3 nrm

J=6 ; nband=J # nu of bands
B2S_method = "NN" # SVshape NN LINSHAPE LINES PARAM MONOT
bandWidth_mult = 1 # multiplier for all bands' halfwidths
nc2_mult = 1 # multiplier for nc2 (2nd band's tranfu-maximum wvn-location)

# Non-stationarity: preTransform flds prms

SDxi_med = 1       # median of the statio fld S(x)  
SDxi_add = SDxi_med /10 # minimal SDxi
lambda_add = dx/3    # minimal lambda(x): accounts for the grid resolution
lambda_med = dx*3  # median lambda(x): the desired median length scale 
gamma_add = 1.0      # minimal gamma(x): avoid too low gamma ==> weird crf
gamma_med = 4.0    # S1: 3.0 yields almost AR2 cvf (S2: gamma=3.1 yields AR2)
n0 = 0             # prm spec mdl b[n]=V/(1+(lambda*(n+n0))^gamma)

# Fit a time-mean spectral shape to the local spectrum -- prms

moments = "012" # : use "01" , "12" , "012"
a_max_times = 5e3 # max deviation of the scale multiplier  a  from 1 in times 
w_a_fg = 0.0  #  weight of the ||a-1||^2 weak constraint in fitScaleMagn

# BANDS' prms

q_tranfu=2 # tranfu=exp(-|(n-nc)/halfwidth|^q_tranfu)) 2, 3 nrm
halfwidth_min=bandWidth_mult * nx/50 # 360:/50
nc2=                nc2_mult * nx/40           # 360:/40                 
halfwidth_max=bandWidth_mult * nx/5  # 360:/5 

if(nx == 60){
  q_tranfu=3 # tranfu=exp(-|(n-nc)/halfwidth|^q_tranfu)) 2, 3 nrm
  halfwidth_min=bandWidth_mult * 2.5 # 360:/50
  nc2=                nc2_mult * 4.5           # 360:/40                 
  halfwidth_max=bandWidth_mult * 20  # 360:/5 
  
  moments = "01"
}

# B2S prms

if(B2S_method == "SVshape" | B2S_method == "MONOT"){
   nSV_discard = 0 # discard trailing SV to filter sampling noise
}
if(B2S_method == "MONOT"){
  c_z_fg_rel = 3e-3 # weight of FG penalty (rel z_fg_mean)
  c_smoo_monot=1e-2   # weight of smoothing penalty (not sens)
  LS_solver ="nnls" # "ls" "nnls"
  k0_log=1 # prm in log-transform of  k=l+1: t=log(k+k0_log), >=0 (not sens)
  eps_V_sd = 0.01 # portion of mean TSD(band_Ve) used as an additive rglrzr 
}
if(B2S_method == "LINES" | B2S_method == "LINSHAPE" | B2S_method == "PARAM" 
   | B2S_method == "MONOT"){
  B2S_bbc = TRUE # FALSE TRUE apply bootstrap bias correc (bbc) B2S_lines (T/F)
  niter_bbc = 3 # number of bbc iterations
  correc_bbc = "mult" # "add" or "mult" - method of bbc-correction
} 
if(B2S_method == "PARAM"){
  na=21
  a_search_times = 1.5
  eps_V_sd = 0.2 # portion of mean TSD(band_Ve) used as an additive rglrzr 
  eps_smoo=0.25 # rel strength of penalty on Smoothness of errors in V_j
  w_a_rglrz = 10 # weight, penalize deviations of  a  from  a_fg=1
}

rel_W_threshold = 1e-2 # threshold W
glob_loc_thresholding = 1

w_ensm_envar = 1 #0.5 weight in hybr S_lcz <- w_ensm_envar*S_lcz + (1-w_ensm_envar)*B_median (opt 0.5)
w_LSM_envar = 1    # weight in hybr B_LSM <- w_LSM_envar*B_LSM + (1-w_LSM_envar)*B_median (opt 1)

BOOTENS = F # T F bootstrap the ensm?
nB = 21 # Bootstrap-sample size. ODD if median is to be within the sample

LEARN = F                    
if(LEARN){                 # train the NN here
  message("NN learning mode")
  message("(Skip B2S & ANLS)")
  
}else{
  if(B2S_method == "NN"){  #  a pre-trained NN is to be used
    
    # Read a pre-trained NN from a file
  
    
    
    
  }
}

CALIBRATE = F # if TRUE, calc only the n_repl-averaged b_shape_estm[n]
if(CALIBRATE){
  message("CALIBRATION mode: estm  b_mean")
  message("(Skip ENSM, E2B, B2S, & ANLS)")
}

obs_per_grid_cell = 1
n_obs=round(nx*obs_per_grid_cell) # 1 obs per mesh_obs_mean grid meshes (on average) (1 or 2)
sd_obs_rel_FG=1 # 1 0.5
repeated_obs_location = T # F or T allow several obs to be located at the same grid point?

seed = 1341573 # 25093714
# seed=seed+10
set.seed(seed)  # fix start seed

n_repl = 1

# Plotting

l_CRL_COV_plots = 1 # 1 - spat crl, 2 - covar

#--------------------------------------------------------------
# derived and minor prms, synonyms

truth_available = TRUE

# constants

Rem=6.37*10^6 # m
Rekm=Rem / 10^3 # km

# Grids

dx_km = dx* Rekm
xx=c(0:(nx-1))* dx
xx_km = xx * Rekm
ii_x = c(1:nx)
ii_n = c(1:nx)
nn_half = c(0:nmax)
nn_half_pn0 = nn_half + n0
nn = c(0:(nx-1))

# Lcz fu type

crftype = "AR2" # "exp", "AR2", "AR3", "Gau", "Cauchy", "ecos", "Bessel", "Matern"

# Bands' tranfu shape: rectangular?

rectang = FALSE # FALSE TRUE rectangular shape of passbands' transfer functions

# True spectra: derived prms

SDxi_mult  = SDxi_med - SDxi_add
lambda_mult = lambda_med - lambda_add 
gamma_mult =  gamma_med - gamma_add

Lambda_preTransform = lambda_med * NSL      # pre-transform (statio) len scale
Gamma_preTransform = gamma_med   # pre-transform (statio) spectral shape prm

n1=1
lognp1 = log(c(0:nmax) + n1) 

# ratio MD (or MAD - mean abs dev from mean - not median!) to SD for chi^2 distr
md2sd_chi2 = md2sd(distrib = "chi2", nu=ne-1)

# scipen: favors sci notation (1e5) over fixed notation (100000)

options(scipen = -1)   

# R plotting margins

par(mar=c(5.1, 4.1, 4.1, 4.1))

# General

n=nx
nmaxp1 = nmax +1
nmaxp2 = nmax +2
ns = nmaxp1

Jm1 = J-1
Jm2 = J-2

# Checks

if(nB %% 2 == 0){
  message("nB=", nB, " -- should be ODD")
  stop("Change nB")
}

#--------------------------------------------------------------
# Modify L_u_mult according to crftype

if(crftype=="exp") mult=1
if(crftype=="Gau") mult=1
if(crftype=="AR2") mult=1/sqrt(2)
if(crftype=="AR3") mult=1/sqrt(3)
if(crftype=="ecos") mult=1  # d=1,2 only

nu=1

if(crftype=="Cauchy") {
  mult=1
  # nu >0. Range: 1 2 3
  if(nu==1) mult=1
}

if(crftype=="Bessel") {
  if(nu==0.2) mult=1/5
  if(nu==0.5) mult=1/5
  if(nu==1) mult=1/4
  if(nu==2) mult=1/3
  
  # NB: nu >= (d-2)/2  !
  # range:
  # d=1: 0.2,0.5,1,2 
  # d=3: 0.6, 1, 2
}

if(crftype=="Matern") {
  mult=1/3
  # nu >0. Range: 1 2 3
  if(nu==1) mult=1.5
  if(nu==2) mult=1.5
}

# Specify L_lcz

if(ne < 10) L_lcz = 1.5* lambda_med/2  # lcz length scale, rad.
if(ne < 20 & ne >= 10) L_lcz = 2.0* lambda_med/2  # lcz length scale, rad.
if(ne >= 20 & ne < 40) L_lcz = 3 * lambda_med/2  # lcz length scale, rad.
if(ne >= 40 & ne < 80) L_lcz = 3.5 * lambda_med/2  # lcz length scale, rad.
if(ne >= 80) L_lcz = 4.5 * lambda_med/2  # lcz length scale, rad.

L_xi_median = lambda_med/2 

# End of the SETUP section
#--------------------------------------------------------------
#--------------------------------------------------------------
# Median and pre-transform W, B, Sigma, spectra

#-------------------------
# Pre-transform (i.e. chi fields') spectrum.
#    b(n) = c/( 1 + (Lambda_median * (n+n0))^gamma_median ) 
# c  is calculated to ensure that 
#    Var(chi) = \sum{all n on the circle} b[n] =1

b_preTransform = c(1:nx) # init
b_preTransform[1:nmaxp1] = 1/( 1 + (Lambda_preTransform * nn_half_pn0[])^Gamma_preTransform )
b_preTransform[nmaxp2:nx] = rev(b_preTransform[2:nmax])
v = sum(b_preTransform)
b_preTransform = b_preTransform / v 

# sum(b_preTransform)
# plot(b_preTransform)

# implied preTransform fld cvf 

ccvf_preTransform = fft(b_preTransform, inverse = TRUE)
# max(abs(Im(ccvf_preTransform)))
ccvf_preTransform=Re(ccvf_preTransform)

# mn=min(ccvf_preTransform)
# mx=max(ccvf_preTransform)
# plot(ccvf_preTransform, type="p", ylim=c(0,mx))

# nx_plot=nx/2
# mx=max(ccvf_preTransform[1:nx_plot])
# mn=min(ccvf_preTransform[1:nx_plot])
# plot(x=xx_km[1:nx_plot], y=ccvf_preTransform[1:nx_plot], type="p", ylim=c(mn,mx),
#      main="Cvf of preTransform fields chi_*", xaxs="i", yaxs="i")
# abline(h=0)
#--------------------------
# Median spectrum
#    b(n) = c/( 1 + (lambda_med * (n+n0))^gamma_med ) 
# c  is calculated to ensure that 
#    Var(chi) = \sum{all n on the circle} b[n] = SDxi_med^2

b_median_1D = c(1:nx) # init
b_median_1D[1:nmaxp1] = 1/( 1 + (lambda_med * nn_half_pn0[])^gamma_med )
b_median_1D[nmaxp2:nx] = rev(b_median_1D[2:nmax])
v = sum(b_median_1D)
b_median_1D_nrm = b_median_1D / v
b_median_1D = b_median_1D_nrm * SDxi_med^2

# sum(b_median_1D)
# plot(b_median_1D)

plot(x=log(nn_half+n1), y=log(b_preTransform[1:nmaxp1]/b_preTransform[1]),
     main="Shape of \n log(preTransform spectrum) (circ) \n log(median spectrum) (line) ")#, xaxs="i", yaxs="i")
lines(x=log(nn_half+n1), y=log(b_median_1D[1:nmaxp1]/b_median_1D[1]))

# implied median fld cvf 

ccvf_med = fft(b_median_1D, inverse = TRUE)
# max(abs(Im(ccvf_med)))
ccvf_med=Re(ccvf_med)

# mn=min(ccvf_med)
# mx=max(ccvf_med)
# plot(ccvf_med, type="p", ylim=c(0,mx))
plot(ccvf_med[1:(nmax/4)])
abline(h=0)

nx_plot=nx/2
mx=max(ccvf_med[1:nx_plot], ccvf_preTransform[1:nx_plot])
mn=min(ccvf_med[1:nx_plot], ccvf_preTransform[1:nx_plot])

plot(x=xx_km[1:nx_plot], y=ccvf_med[1:nx_plot], type="p", ylim=c(mn,mx),
     main="Cvf of median fields (circ), \n preTransform (chi) flds (line)", 
     xaxs="i", yaxs="i")
lines(x=xx_km[1:nx_plot], y=ccvf_preTransform[1:nx_plot])
abline(h=0)

#------------
# Sigma_median, B_median
# NB: Sigma[ix,n+1] etc.

b_median_T = matrix(b_median_1D, nrow = nx, ncol = nx)
b_median = t(b_median_T)
Sigma_median = sqrt(b_median)

WB_median = Sigma2WB(Sigma_median)

W_median = WB_median$W
B_median = WB_median$B

# plot(B_median[1,])
# lines(ccvf_med, type="l", ylim=c(0,mx))
# max(abs(B_median[1,] - ccvf_med)) # OK

#--------------------------------------------------------------
#--------------------------------------------------------------
# Create the BANDS

BANDS = CreateExpqBands(nmax, nband, halfwidth_min, nc2, halfwidth_max, 
                        q_tranfu=3, rectang = FALSE)

tranfu = BANDS$tranfu
band_centers_n  = BANDS$band_centers_n
hhwidth = BANDS$hhwidth
round(band_centers_n,1)
round(hhwidth,1)

tranfu2 = (abs(tranfu))^2 # [i_n, j=1:J]

H_flt = t(tranfu) # NB H_flt is real valued

# "tall" SVD

H_flt_SVD = svd(H_flt)
UH_flt = H_flt_SVD$u
dH_flt = H_flt_SVD$d
VH_flt = H_flt_SVD$v


# "full" SVD

# H_flt_fullSVD = svd(H_flt, nu=J, nv=nx)
# UH_flt_full = H_flt_fullSVD$u
# dH_flt_full = H_flt_fullSVD$d
# VH_flt_full = H_flt_fullSVD$v

#---------------------------------------------
# Omega and its SVD

lplot=F

Omega_SV = Omega_SVD(tranfu2, lplot)

Omega_nmax = Omega_SV$Omega_nmax
Omega_S1   = Omega_SV$Omega_S1
SVD_Omega_nmax = Omega_SV$SVD_Omega_nmax
SVD_Omega_S1   = Omega_SV$SVD_Omega_S1

#--------------------------------------------------------------
#--------------------------------------------------------------
# Preparations for the BIG EXTERNAL (replicates) LOOP

if(CALIBRATE) crf_SrMs = rep(0, nx)

if(LEARN){
  thinningFactor = 10
  spatInd = seq(from = 1, by = thinningFactor, to = nx)
  nx_thinned = length(spatInd)
  TrueSpectra  = array(0, dim=c(nmaxp1, nx_thinned, n_repl))
  EnsmBandVars = array(0, dim=c(nband,  nx_thinned, n_repl))
}

xi_Vt_Ms = c(1:n_repl)
xi_Ve_Ms = c(1:n_repl)
xi_Ve_AEs = c(1:n_repl)
xi_Ve_TAD_Ms = c(1:n_repl)

# Band space

bband_Ve_MEs    = matrix(0, nrow=n_repl, ncol=nband)
bband_Ve_AEs    = matrix(0, nrow=n_repl, ncol=nband)
bband_Ve_TD_Ms = matrix(0, nrow=n_repl, ncol=nband)  # theor stand dev
bband_Ve_TAD_Ms = matrix(0, nrow=n_repl, ncol=nband) # theor abs dev

bband_Vt_Ms     = matrix(0, nrow=n_repl, ncol=nband)

bband_V_restored_AEs   = matrix(0, nrow=n_repl, ncol=nband)
bband_V_restored_MEs   = matrix(0, nrow=n_repl, ncol=nband)
bband_V_restored_misfit_As   = matrix(0, nrow=n_repl, ncol=nband)
bband_V_restored_misfit_Ms   = matrix(0, nrow=n_repl, ncol=nband)

bband_V_esd_Ms  = matrix(0, nrow=n_repl, ncol=nband)

GGamma_true_Ms   = array(0, dim=c(J,J, n_repl))
ccvm_phi_estm_Ms = array(0, dim=c(J,J, n_repl))


# Spe space

bb_true_Ms      = matrix(0, nrow=n_repl, ncol=nx) 
b_LSM_MEs       = matrix(0, nrow=n_repl, ncol=nx)
b_LSM_AEs       = matrix(0, nrow=n_repl, ncol=nx) # mean-abs err in spectrum b

# Phys space: correlations

aaligned_C_Ms        = matrix(0, nrow=n_repl, ncol=nx) # truth
aaligned_C_LSM_MEs   = matrix(0, nrow=n_repl, ncol=nx) # LSM bias
aaligned_CS_lcz_MEs  = matrix(0, nrow=n_repl, ncol=nx) # S_lcz bias
aaligned_C_LSM_AEs   = matrix(0, nrow=n_repl, ncol=nx) # LSM MAE
aaligned_CS_lcz_AEs  = matrix(0, nrow=n_repl, ncol=nx) # S_lcz MAE
nn_nmon = c(1:n_repl)

LLL_u_local = matrix(0, nrow=nx, ncol=n_repl) # loc len scales
ssd_xi_local= matrix(0, nrow=nx, ncol=n_repl) # local SD(xi)

ms_anls_err_Banls_isB_true = c(1:n_repl)
ms_anls_err_Banls_isB_LSM = c(1:n_repl)
ms_anls_err_Banls_isS_lcz = c(1:n_repl)
ms_anls_err_Banls_isB_med = c(1:n_repl)

Amx_diag_Banls_isB_true = c(1:n_repl)
Amx_diag_Banls_isB_LSM = c(1:n_repl)
Amx_diag_Banls_isS_lcz = c(1:n_repl)
Amx_diag_Banls_isB_med = c(1:n_repl)

err_Frob_LSM = c(1:n_repl)
err_Frob_S_lcz = c(1:n_repl)

RelBias_Diag_LSM = c(1:n_repl)
RelBias_Diag_S_lcz = c(1:n_repl)

RelMSE_Diag_LSM = c(1:n_repl)
RelMSE_Diag_S_lcz = c(1:n_repl)

#----------------------------------
# Start of n_repl BIG EXTERNAL LOOP
                                                i_repl=1
                                                for(i_repl in 1:n_repl){
                                                cat("\r",paste0(round(i_repl/n_repl*100,0),'%'))
                                                  t0=proc.time()
                                                lplot=FALSE
                                                #if(n_repl == 1) lplot=TRUE
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#                      W section
# Generate the 3 preTransform flds: chi_S, chi_lambda, chi_gamma
# (all statio and w the same spectrum,  b_preTransform )                                         

nchi =3 # c(chi_S(x), chi_lambda(x), chi_gamma(x))
                                                
cchi = gen_homProc_fromSpectrum_S1(b_preTransform, nchi)

cchi = cchi * log(kappa)                                                
chi_S = cchi[,1]
chi_lambda = cchi[,2]
chi_gamma = cchi[,3]

# plot(chi_S, main="chi_S", type="l")
# plot(exp(chi_S), main="exp(chi_S)", type="l")

g_cchi = apply(cchi, c(1,2), logistic)

# plot(g_cchi[,1], main="g_cchi[,1]", type="l")
# plot(g_cchi[,2], main="g_cchi[,2]", type="l")
# plot(g_cchi[,3], main="g_cchi[,3]", type="l")

# The 3 parameter fields

SDxi_field = SDxi_add   + SDxi_mult   * g_cchi[,1]
Vxi_field = SDxi_field^2
lambda = lambda_add + lambda_mult * g_cchi[,2]
gamma =  gamma_add  + gamma_mult  * g_cchi[,3]

# The local spectra
#    b(x,n) = c(x)/( 1 + (lambda(x) * (n+n0))^gamma(x) ) 
# c  is calculated to ensure that 
#    Var(xi(x)) = \sum{all n on the circle} b[x,n] = S^2(x)

b_true = matrix(0, nrow = nx, ncol = nx) # [ix,i_n]

for (ix in 1:nx){
  b_true[ix, 1:nmaxp1] = 1 /( 1 + (lambda[ix] * nn_half_pn0[])^gamma[ix] )
  b_true[ix, nmaxp2:nx] = rev(b_true[ix, 2:nmax]) # symm
  
  v = sum(b_true[ix,])
  b_true[ix,] = b_true[ix,] / v * Vxi_field[ix]  # nrmlz
}

# W, B, Sigma (true)
# Sigma[ix,n+1] = sigma_n(ix) -- the local spectrum

Sigma = sqrt(b_true)

WB = Sigma2WB(Sigma)

W = WB$W
B = WB$B

#-----------------------
# Plot spectra 

# ix=sample(c(1:nx),1)
# dn=nx/2
# mx=max(b_true[ix, 1:dn])
# plot(b_true[ix, 1:dn], main=paste0("b_true at ix=",ix), type="l", ylim=c(0,mx), xlab="n+1")

b_true_Ms = apply(b_true, 2, mean)  # x-mean spectrum
bb_true_Ms[i_repl,] = b_true_Ms

xi_Vt = apply(b_true, 1, sum) # true variance at all x
sd_xi = sqrt( mean(xi_Vt) )
xi_Vt_Ms[i_repl] = mean(xi_Vt)

ssd_xi_local[,i_repl] = sd_xi

# ix=sample(c(1:nx), 1, replace = FALSE)
# plot(b_true[ix,1:(nx/4)], main=paste0("b_true[n] at x=", ix, "b_true_Ms[n](blu)"), type="l", xlab="wvn+1")
# lines(rep(b_true_Ms), col="blue")

AA_true = b_true[,1]

#--------------------------------------------------------------
# Calc CVM & the symm-pos-def sqrt of CVM

B=W %*% t(W)
C = Cov2VarCor(B)$C

sqrt(mean(diag(B)))

sqB = symm_pd_mx_sqrt(B)$sq

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#                      # CREATE the ENSM section

gau_mx_N01=matrix(nrow=nx, ncol=ne, data=rnorm(nx*ne))

ENSM=sqB %*% gau_mx_N01

#image2D(x=ii_x, y=(1:ne), z=ENSM, main="ENSM", xlab = "Grid point", ylab = "Ensm member")

# LL = EstmMicroScale_S1(ENSM)
# LL_km = LL * Rekm
# L_xi_diagnosed_km = mean(LL_km)
# L_xi_diagnosed_km
# 
# j = sample(1:J, 1)
# plot(ENSM[,j], type="l")

#--------------------------------------------------------------
# Ensm variance. Always subtract ensm mean

xi_Ve = apply(ENSM, 1, var) # ENSM[ix,ie]

xi_Ve_Ms[i_repl]  = mean(xi_Ve)
xi_VeE = xi_Ve - xi_Vt  # error in xi_Ve
xi_Ve_AEs[i_repl] = ExAbs(xi_VeE)

# theoretical estm of SD(S).
# NB: It's SD(S), not Var(S) that is Unbiased. So, we average SD's here.

xi_Ve_TD = xi_Ve * sqrt(2/(ne-1)) # theor SD(xi_Ve)
xi_Ve_TAD = xi_Ve_TD * md2sd_chi2
xi_Ve_TAD_Ms[i_repl] = mean(xi_Ve_TAD)  

if(i_repl == 1){
  xi_Ve_upp = xi_Ve + xi_Ve_TD
  xi_Ve_low = xi_Ve - xi_Ve_TD
  xi_Ve_low[xi_Ve_low<0] = 0
  
  mx=max(max(xi_Ve, xi_Vt, xi_Ve_upp, xi_Ve_low))
  mn=0
  namefile=paste0("./Out/TrueEnsmVars_ne",ne, "ne", ne, "kap", kappa, "NSL", NSL,".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(xi_Vt, type="l", main=paste0("True variance Vt, ensm variance Ve (red), Ve +- SD (grn)"), 
       ylim=c(mn,mx), xlab="Grid point", ylab = "Variance",
       sub=paste0("xi_Vt_Ms=", signif(xi_Vt_Ms[i_repl],4), "   xi_Ve_Ms=", signif(xi_Ve_Ms[i_repl],4)))
  lines(xi_Ve, col="red")
  lines(xi_Ve_low, type="l", col="green", lty=3)
  lines(xi_Ve_upp, type="l", col="green", lty=3)
  abline(h=0, lty=3)
  dev.off()
}

#--------------------------------------------------------------
# Ensm sample CVM
# Calc ensm perturbations.
# Always subtract the mean.

# ensm mean

ENSM_Me = rowMeans(ENSM) # ENSM[ix,ie], ave over ie: ENSM_Me[ix]

# S

d_ENSM = ENSM - matrix(ENSM_Me, nrow = nx, ncol = ne)
S = 1/(ne-1) * d_ENSM %*% t(d_ENSM)

#----------------------------------------------------------------------
# Ave S to get the mean crf
# Calibration: estm and accumulate spat-ave crf

if(CALIBRATE){
  crf_Ms = SpatAveCrf_S1(S)$crf
  crf_SrMs = crf_SrMs + crf_Ms
  next # cycle the BIG external   i_repl  loop
}
#--------------------------------------------------------------
# Localized S

C_lcz = create_crm_S1(nx, L_lcz, crftype, nu=1)
S_lcz = S * C_lcz

# The HYBRID: replace S_lcz

S_lcz = w_ensm_envar*S_lcz + (1-w_ensm_envar)*B_median

# namefile=paste0("S_NSL=",signif(NSL,2), "_ne=",ne, ".png")
# png(namefile, width=7.48, height=5.47, units = "in", res=300)
# par(mgp=c(2.5, 1, 0))
# image2D(x=ii_x, y=ii_x, z=S,
#         main=paste0("S.  NSL=",signif(NSL,2), "_ne=",ne),
#         xlab="Grid point", ylab="Grid point")
# dev.off()

# i=sample(1:nx, 1, replace=TRUE)
# mx=max( max(B[i,]), max(S[i,]) )
# plot(B[i,], type="p", main="B[i,], S[i,] (grn), S_lcz (grn-dash)", ylim=c(0,mx), xlab="x" )
# lines(S[i,], col="green")
# lines(S_lcz[i,], col="blue", lwd=2)
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#                      BAND-SPACE section
#--------------------------------------------------------------
# Calc TRUE band variances from Sigma[ix, i_n], 

band_Vt = t( apply(b_true, 1, function(t) t(tranfu2) %*% t) ) # tranfu2[i_n,j]

band_Vt_Ms   = apply(band_Vt, 2, mean)   # [band]
bband_Vt_Ms[i_repl,] = band_Vt_Ms

#--------------------------------------------------------------
# Calc band spectra for the Median true spectrum

band_VtMed = matrix(0, nrow=nx, ncol=nband)

for (band in (1:nband)){
  for (ix in 1:nx){
    band_VtMed[ix,band]   = sum( (tranfu2[,band] * Sigma_median[ix,])^2 )
  }
}
#image2D(band_VtMed, main="band_VtMed")
#--------------------------------------------------------------
# Calc ensm band vars (and, optionally, their bootstrap versions)

bandData = E2B(ENSM, tranfu, BOOTENS, nB, truth_available)

band_Ve = bandData$band_Ve
if(BOOTENS) band_Ve_B = bandData$band_Ve_B
H1 = bandData$H1 # for Re
H2 = bandData$H2 # For Im
H_big1 = bandData$H_big1
H_big2 = bandData$H_big2
pphi1 = bandData$pphi1  # Re
pphi2 = bandData$pphi2  # Im
Gamma1_true = bandData$Gamma1_true # [1:J,     1:J,     ix=1:nx]
Gamma2_true = bandData$Gamma2_true # [1:(J-2), 1:(J-2), ix=E2B1:nx] (w/o bands 1 & J)
cvm_phi1_estm = bandData$cvm_phi1_estm # cvm of Re(phi)
cvm_phi2_estm = bandData$cvm_phi2_estm # cvm of Im(phi) [1:J,     1:J,     ix=1:nx]

band_Ve_Ms = apply(band_Ve, 2, mean)
Gamma1_true_Ms = apply(Gamma1_true, c(1,2), mean)
Gamma2_true_Ms = apply(Gamma2_true, c(1,2), mean)
cvm_phi1_estm_Ms = apply(cvm_phi1_estm, c(1,2), mean)
cvm_phi2_estm_Ms = apply(cvm_phi2_estm, c(1,2), mean)

GGamma_true_Ms[,,i_repl] = Gamma1_true_Ms
GGamma_true_Ms[2:Jm1,2:Jm1,i_repl] = GGamma_true_Ms[2:Jm1,2:Jm1,i_repl] + Gamma2_true_Ms

ccvm_phi_estm_Ms[,,i_repl] = cvm_phi1_estm_Ms
ccvm_phi_estm_Ms[2:Jm1,2:Jm1,i_repl] = ccvm_phi_estm_Ms[2:Jm1,2:Jm1,i_repl] + cvm_phi2_estm_Ms

#--------------------------------------------------------------
# Mean & mean-abs & RMS err in band_Ve (spat ave)

band_Ve_MEs = apply(band_Ve - band_Vt, 2, mean) 
bband_Ve_MEs[i_repl,]=band_Ve_MEs

band_Ve_AEs = apply(band_Ve - band_Vt, 2, ExAbs) 
bband_Ve_AEs[i_repl,]=band_Ve_AEs

# band_V theoretical SD - estm assuming normality of the ensm:  SD(band_Ve)
# NB: It is  SSD(Ve)=sqrt(2/(ne-1))*Ve  (SSD stands for Sample SD) that, 
#     with the unknown true variance Vt, is an unbiased estm of of the true SD(Ve).

band_Ve_TD = sqrt(2/(ne-1)) * band_Ve # "theoretical" SD of sample var
bband_Ve_TD_Ms[i_repl,] = apply(band_Ve_TD, 2, mean)

band_Ve_TAD = band_Ve_TD * md2sd_chi2
bband_Ve_TAD_Ms[i_repl,] = apply(band_Ve_TAD, 2, mean)

#-----------------------------------------------------------------
# Plot band vars fields

if(n_repl == 1){
  j=round(nband/2)
  mx=max(band_Ve[,1:j], band_Vt[,1:j])
  image2D(x=ii_x, y=c(1:j), z=band_Vt[,1:j], main="band_Vt", zlim=c(0,mx), 
          xlab = "Grid point", ylab = "Band")
  image2D(x=ii_x, y=c(1:j), z=band_Ve[,1:j], main=paste0("band_Ve"), 
          zlim=c(0,mx), xlab = "Grid point", ylab = "Band")
  
  nb=nband
  mx=max(band_Ve_Ms[1:nb], band_Vt_Ms[1:nb])
  plot(band_Vt_Ms[1:nb], main="band_V_Ms: true & ensm (red)\n TMD_Ms (grn)", xlab="band", ylim=c(0,mx))
  lines(band_Ve_Ms[1:nb], col="red")
  lines(bband_Ve_TAD_Ms[i_repl,1:nb], col="green")
}

if(n_repl == 1){
  ix=sample(c(1:nx), 1, replace = FALSE)
  mx=max(band_Ve[ix,], band_Vt[ix,])
  plot(band_Ve[ix,], main=paste0("band_Ve (red), band_Vt (blck)",
                                 "\n TMD (grn).  @ix=",ix), 
       ylim=c(0,mx), type="l", col="red")
  lines(band_Vt[ix,], col="black")
  lines(band_Ve_TAD[ix,], col="green")
}
#--------------------------------------------------------------
# Stats in BAND-SPACE. 

# Plot over x

ixm = nx
if(i_repl == 1){
  for( band in c(1,2,3,seq(from=4, to=nband, by=2)) ){
    if(!BOOTENS){
      band_Ve_upp = band_Ve[1:ixm, band] + band_Ve_TD[,band]
      band_Ve_low = band_Ve[1:ixm, band] - band_Ve_TD[,band]
      band_Ve_low[band_Ve_low<0] = 0
      mx=max(band_Vt[1:ixm, band], band_Ve[1:ixm, band], band_Ve_upp[1:ixm], band_Ve_low[1:ixm])
    }else{
      mx=max(band_Vt[1:ixm, band], band_Ve[1:ixm, band], band_Ve_B[1:ixm,band,])
    }
    namefile=paste0("./Out/band_VeFields_j", round(band_centers_n[band],1), 
                    "hw", round(hhwidth[band],1), 
                    "ne", ne, "kap", kappa, 
                    "NSL", NSL, ".png")
    png(namefile, width=7.48, height=5.47, units = "in", res=300)
    par(mgp=c(2.5, 1, 0))
    plot(band_Vt[1:ixm, band], type="l", xlab="Grid point", ylab="Band variance",
         main=paste0("Band ",band, ":\n V_true (black), V_ensm (red), V_ensm +- SD (grn)"), 
         ylim=c(0,mx))
    lines(band_Ve[1:ixm, band], col="red", lty=1, lwd=2)
    if(!BOOTENS){
      lines(band_Ve_low[1:ixm], col="green", lty=3, lwd=2)
      lines(band_Ve_upp[1:ixm], col="green", lty=3, lwd=2)
    }else{
      for(iB in 1:nB){
        lines(band_Ve_B[1:ixm,band,iB], col=rgb(red=0, green=1, blue=0, alpha=0.5), lwd=0.5, lty=3)
      }
    }
    abline(h=0, lty=3)
    dev.off()
  }
}

# Space averaged stats: plot over bands

j5=min(5,J)

if(n_repl == 1){
  mn=min(0, band_Ve_AEs, band_Ve_MEs, band_Vt_Ms, bband_Ve_TAD_Ms[i_repl,])
  mx=max(band_Ve_AEs, band_Ve_MEs, band_Vt_Ms, bband_Ve_TAD_Ms[i_repl,])
  namefile=paste0("./Out/BandspaceErr_", 
                  "ne", ne, "kap", kappa, 
                  "NSL", NSL, ".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  plot(band_Ve_AEs, lwd=1.5,
       main=paste0("Ensemble band variances: sample & theor errors. Truth",
                   "\n ne=", ne, 
                   " NSL=",NSL, " kappa=", kappa),  
       type="l", xlab="band", ylab="Truth and errors", ylim=c(mn,mx), col="red",
       sub=paste0( "Rel RMS band 1: ",signif(band_Ve_AEs[1]/band_Vt_Ms[1],2),
                   "  mean over bands 1-5: ",signif( mean(band_Ve_AEs[1:j5]/band_Vt_Ms[1:j5]),2 ) ) )
  lines(band_Ve_MEs, col="blue")
  lines(band_Vt_Ms, col="black", lwd=2)
  lines(bband_Ve_TAD_Ms[i_repl,], col="green", lty=3, lwd=2)
  
  leg.txt<-c('MAE', 'Bias', 'Truth', "MD")
  leg.col<-c("red", "blue", "black", "green")
  legend("topright", inset=0, leg.txt, col=leg.col, 
         lwd=c(1.5,2,2), lty=c(1,1,1,3), pch=c(NA,NA,NA,NA),
         pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
  
  abline(h=0, lty=3)
  dev.off()
}

# Band variances plots point by point

if(n_repl == 1){
  ix=sample(c(1:nx),1)
  if(!BOOTENS){
    band_Ve_upp = band_Ve[ix,] + band_Ve_TD[ix,]
    band_Ve_low = band_Ve[ix,] - band_Ve_TD[ix,]
    band_Ve_low[band_Ve_low < 0] = 0
    mx=max(band_Ve_upp, band_Vt[ix,])
  }else{
    mx=max(band_Vt[ix,], band_Ve[ix,], band_Ve_B[ix,,])
  }
  
  plot(band_Ve[ix,], main=paste0("Band variances at ix=", ix), pch=19, 
       col="magenta", ylim=c(0,mx), xlab = "Band", ylab = "V")
  lines(band_Vt[ix,], type = "p", pch=16, lty=3, lwd=3)
  if(!BOOTENS){
    lines(band_Ve_upp, type = "l", lty=3, lwd=2, col="green")
    lines(band_Ve_low, type = "l",  lty=3, lwd=2, col="green")
    leg.txt<-c('Ve', 'Vt', 'Ve +- SD')
  }else{
    for(iB in 1:nB){
      lines(band_Ve_B[ix,,iB], col=rgb(red=0, green=1, blue=0, alpha=0.5), lwd=0.5, lty=3)
    }
    leg.txt<-c('Ve', 'Vt', 'Bootstrap')
  }
  leg.col<-c("magenta", "black", "green")
  legend("topright", inset=0, leg.txt, col=leg.col, 
         lwd=c(NA,NA,2,2), lty=c(NA,NA,3,3), pch=c(19,16,NA,NA),
         pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
}
# End of the BAND-SPACE section.
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Store the data to learn the neural network after the big loop is over

if(LEARN){
  # TrueSpectra  = array(0, dim=c(nmaxp1, nx_thinned, n_repl))
  # EnsmBandVars = array(0, dim=c(nband,  nx_thinned, n_repl))
  
  TrueSpectra[,,i_repl] = t(b_true[spatInd, 1:nmaxp1])
  EnsmBandVars[,,i_repl]= t(band_Ve[spatInd,1:nband])
  
  next # go to the next BIG-loop iteration (skip B2S and anls)
}

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#           B2S:  FROM band_Ve[x, 1:nband] to b_n(x) section

#----------------------------------------------------------------------
# Extract spe "shape"  g(n)  so that the real spectrum c/b apxted as  A*g(n/a).
# If the CALIBRATE run has been in effect, then the mean spectrum  b_shape_estm  is used
# Otherwise, the median spectrum is utilized

if( exists("b_shape_estm") ){   # Select the spe shape: true or estmtd
  
  b_shape = b_shape_estm
  # b_shape = b_median_1D_nrm[1:nmaxp1]   # tmp!
  
}else{ # fall back to the true spectrum at the point x where it is the widest.
  # Extract the true-shape spectrum mdl A*g(n/a),   g = b_shape_true
  # (take the point with the LARGEST L_u_local. If not,
  # the spectrum decays too slowly & sometimes doesn't reach 0 even)
  # In this case, a <= 1.
  # b_shape_true[] is the narrowest true spectrum for the current realization
  #  of L_u_local[].
  
  b_shape = b_median_1D_nrm[1:nmaxp1]
  #b_shape = b_median_analyt
}

# Find AA_true, aa_true at all grid points  via  fitScaleMagn

test_fitScaleMagn=T
if(test_fitScaleMagn){
  Aa_true = fitScaleMagn(t(b_true[1:nx, 1:nmaxp1]), b_shape, xx[1:nmaxp1],
                         moments, a_max_times, w_a_fg, lplot)
  AA_true = Aa_true$AA
  aa_true = Aa_true$aa
  
  # plot(AA_true)   # wrt b_shape (c/b dfr fr  b_median_1D_nrm  after CALIBRATION)
  # plot(aa_true)
}

#----------------------------------------------------------------------
#    "LINES"
# Draw segments of straight lines through (nc[j, bmean[j]])

if(B2S_method == "LINES" | B2S_method == "LINSHAPE"){

  band_V=band_Ve # band_Ve, band_Vt - for debug
  t1=proc.time()
  
  SPECTRUM_e_lines = B2S_lines(tranfu2, band_V, ne,
                               B2S_bbc, niter_bbc, correc_bbc, lplot)
  
  b_lines               = SPECTRUM_e_lines$b_lines
  band_V_lines_restored = SPECTRUM_e_lines$band_V_restored
  
  norm(b_lines - b_true, "F") / norm(b_true, "F")
  mean(b_lines - b_true) / mean(b_true)
  proc.time()-t1
}
#----------------------------------------------------------------------
#   LINSHAPE
# Use the true spectrum's SHAPE to fit b_lines and get a smooth and
# better-behaving-near-the-origin estimate of b_n 

if(B2S_method == "LINSHAPE"){
 
  if(moments == "skip"){  # use "lines"
    
    b_linshape               = b_lines
    band_V_shape_restored = band_V_lines_restored
    
  }else{                  # calc and use "shape"
    
    b_tofit = b_lines
    band_V  = band_Ve
    t1=proc.time()
    
    SPECTRUM_e_shape = B2S_shape(b_tofit, b_shape, tranfu2, band_V, 
                                 moments, a_max_times,w_a_fg, lplot)
    proc.time()-t1
    
    b_linshape            = SPECTRUM_e_shape$b_fit
    band_V_shape_restored = SPECTRUM_e_shape$band_V_restored
    AA_linshape = SPECTRUM_e_shape$AA  # wrt b_shape
    aa_linshape = SPECTRUM_e_shape$aa  # wrt b_shape
  }
  
  norm(b_linshape - b_true, "F") / norm(b_true, "F")
  mean(b_linshape - b_true) / mean(b_true)
  
  norm(b_lines - b_true, "F") / norm(b_true, "F")
  mean(b_lines - b_true) / mean(b_true)
}

#----------------------------------------------------------------------
#    SVshape
# Estimate  b_n  using SVD of  Omega

if(B2S_method == "SVshape" | B2S_method == "MONOT"){
  band_V = band_Ve # band_Ve, band_Vt (debug)
  band_Vt_S1 = cbind(band_Vt, band_Vt[, (J-1):2])
  
  SPECTRUM_SVshape = B2S_SVshape(band_V, Omega_S1, SVD_Omega_S1,
                                 b_shape, moments,
                                 BOOTENS, band_Ve_B,
                                 nSV_discard, a_max_times, w_a_fg, 
                                 truth_available, lplot)
  
  b_SVshape = SPECTRUM_SVshape$b_fit
  band_V_SVshape_restored = SPECTRUM_SVshape$band_V_restored
  AA_SVshape = SPECTRUM_SVshape$AA  # wrt b_shape
  aa_SVshape = SPECTRUM_SVshape$aa  # wrt b_shape
  rel_err_discard_nullspace_b = SPECTRUM_SVshape$rel_err_discard_nullspace_b
}

#----------------------------------------------------------------------
#    NN
# Estimate  b_n  using NN

if(B2S_method == "NN"){
  band_V = band_Ve # band_Ve, band_Vt (debug)

  SPECTRUM_NN = B2S_NN(band_V, NN, Omega_nmax, truth_available, lplot)
  
  b_NN = SPECTRUM_NN$b_fit
  band_V_NN_restored = SPECTRUM_NN$band_V_restored
  
}
#----------------------------------------------------------------------
#    MONOT
# Estimate b_n using a variational formulation 
# with an optional Monotonic-b constraint

if(B2S_method == "MONOT"){
  t1=proc.time()
  band_V = band_Ve # band_Ve, band_Vt (debug)
  b_fg = b_SVshape

  SPECTRUM_monot = B2S_monot(band_V, Omega_nmax, 
                         ne, eps_V_sd, band_Ve_TD, 
                         b_fg, c_z_fg_rel, 
                         k0_log, c_smoo_monot,
                         LS_solver, truth_available, lplot)
  b_monot = SPECTRUM_monot$b_fit
  band_V_monot_restored = SPECTRUM_monot$band_V_restored

  
  ix = sample(1:nx, 1)
  plot(b_SVshape[ix,])
  lines(b_monot[ix,])
}
#----------------------------------------------------------------------
# Estimate b_n using a parametric formulation
 
if(B2S_method == "PARAM"){
  band_V = band_Ve # band_Ve, band_Vt (debug)
  
  SPECTRUM_param = B2S_param(band_V, Omega_nmax, 
                             b_shape, a_search_times, na, 
                             eps_V_sd, band_Ve_TD, 
                             eps_smoo, w_a_rglrz, 
                             BOOTENS, band_Ve_B, lplot)
  
  b_param = SPECTRUM_param$b_fit
  band_V_param_restored = SPECTRUM_param$band_V_restored
}

#----------------------------------------------------------------------
# (2) Final estm of b_n

if(B2S_method == "LINES"){
  
  b_LSM=b_lines
  band_V_restored = band_V_lines_restored
  
}else if(B2S_method == "LINSHAPE"){
  
  b_LSM=b_linshape
  band_V_restored = band_V_shape_restored
  
}else if(B2S_method == "SVshape"){
  
  b_LSM=b_SVshape
  band_V_restored = band_V_SVshape_restored  
  
}else if(B2S_method == "NN"){
  
  b_LSM=b_NN
  band_V_restored = band_V_NN_restored  
  
}else if(B2S_method == "MONOT"){
  
  b_LSM=b_monot
  band_V_restored = band_V_monot_restored
  
}else if(B2S_method == "PARAM"){
  
  b_LSM=b_param
  band_V_restored = band_V_param_restored
} 

# Store the spatially-averaged bias & MSE of the spectra

b_LSM_MEs[i_repl,]   = apply(b_LSM   - b_true, 2, mean)
b_LSM_AEs[i_repl,]   = apply(b_LSM   - b_true, 2, ExAbs)

# mx=max(b_LSM, b_true)
# image2D(b_true, main="b_true", zlim=c(0,mx))
# image2D(b_LSM, main="b_LSM", zlim=c(0,mx))
# plot(b_LSM[1,])

#----------------------------------------------------------------------
#            b_LSM: Restored-BAND-space stats 
# (restored from the estimated local spectra b_LSM)

# Restored-BANDSPACE errors

band_V_restored_err = band_V_restored - band_Vt  # [ix, band], error
bband_V_restored_AEs[i_repl,] = apply(band_V_restored_err, 2, ExAbs)
bband_V_restored_MEs[i_repl,] = apply(band_V_restored_err, 2, mean)

band_V_restored_misfit = band_V_restored - band_Ve # misfit wrt original Ve
bband_V_restored_misfit_As[i_repl,] = apply(band_V_restored_misfit, 2, ExAbs)
bband_V_restored_misfit_Ms[i_repl,] = apply(band_V_restored_misfit, 2, mean)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# LSM Deliverables: Sigma, W, B

Sigma_LSM = sqrt(b_LSM)

WB = Sigma2WB(Sigma_LSM)
B_LSM = WB$B
W_LSM = WB$W

#------------------------------------
# Thresholding W_LSM

# sparsity of W_LSM

sparsity_W_orig = sum(W_LSM == 0) / (nx^2)
sparsity_W_orig

if(glob_loc_thresholding == 1){
  W_max = max(W_LSM)
  thres = rel_W_threshold * W_max
  W_LSM[W_LSM < thres] =0
  
}else if(glob_loc_thresholding == 2){ # threshold along rows
  wW_max = apply(W_LSM, 1, max)
  tthres = rel_W_threshold * wW_max
  for (ix in 1:nx){
    vanish = which(W_LSM[ix,] < tthres, arr.ind = T)
    W_LSM[ix, vanish] = 0
  }
}

# ==> Little difference betw glob_loc_thresholding=1 and =2:
#     both in the sparsity and the anls RMSE.

sparsity_W_thresholded = sum(W_LSM == 0) / (nx^2)
message("sparsity_W_thresholded=", signif(sparsity_W_thresholded,2))

# Recompute  B_LSM  from the thresholded W_LSM

if(rel_W_threshold > 1e-5){
  B_LSM = W_LSM %*% t(W_LSM)
}

# Hybridize  B_LSM   w  B_median

B_LSM = w_LSM_envar*B_LSM + (1-w_LSM_envar)*B_median

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#                       STATS of spe & covs

# B_LSM, S_lcz, & B cvf-plots

# image2D(B_LSM, main="B_LSM")
# image2D(B, main="B")
# image2D(S_lcz, main="S_lcz")

# which(B_LSM == max(B_LSM), arr.ind = TRUE)

# B_LSM: PHYS-space stats

if(0==1){
  ix=sample(c(1:nx), 1)
  mn=min(B[ix,], B_LSM[ix,], S_lcz[ix,])
  mx=max(B[ix,], B_LSM[ix,], S_lcz[ix,])
  plot(B[ix,], main=paste0("B_tru, B_LSM (red), S_lcz (blu)\n at ix=",ix), 
       ylim=c(mn,mx), type="l")
  lines(B_LSM[ix,], col="red")
  lines(S_lcz[ix,], col="blue")
  
}

# b_LSM: SPE-space stats

if(n_repl == 1){
  ix=sample(c(1:nx), 1, replace = FALSE)
  nm=nmax/2
  plot(b_true[ix,1:nm]/b_true[ix,1], type="l", ylim=c(0,1), lwd=2, 
       main=paste0("b_n at ix=", ix, 
                   ":\n true (black, thick), LSM (red)"), xlab="n+1")
  lines(b_LSM[ix,1:nm]/b_LSM[ix,1], col="red")
  
}
if(n_repl == 1){
  ix=sample(c(1:nx), 1, replace = FALSE)
  # ix=(ix+2) %% nx
  nm=nmax/2
  mx = max(b_true[ix,1:nm], b_LSM[ix,1:nm]  )
  plot(b_true[ix,1:nm], type="l", ylim=c(0,mx), lwd=2, 
       main=paste0("b_n at ix=", ix, 
                   ":\n  true (black, thick), LSM (red)"), xlab="n+1")
  lines(b_LSM[ix,1:nm], col="red")
  
}


if(1==1){
  d=nx/8
  ix=sample(c((d+1):(nx-d)),1)
  # ix=(ix+2) %% nx
  mx=max( max(B[ix, (ix-d):(ix + d)]), max((S_lcz[ix, (ix-d):(ix + d)])), max((B_LSM[ix, (ix-d):(ix + d)])) )
  mn=min( min(B[ix, (ix-d):(ix + d)]), min((S_lcz[ix, (ix-d):(ix + d)])), min((B_LSM[ix, (ix-d):(ix + d)])) )
  if(mn > 0) mn=0
  plot(B[ix, (ix-d):(ix + d)], type="l", lwd=2, main="Row: B, B_LSM(red), S_lcz(blu)", 
       sub=paste0("i_repl=", i_repl, "  ix=",ix), xlab="Distance, meshes", ylim=c(mn,mx))
  lines(B_LSM[ix, (ix-d):(ix + d)], col="red")
  lines(S_lcz[ix, (ix-d):(ix + d)], col="blue")
  
}

if(1==2){
  d=nx/8
  ix=sample(c((d+1):(nx-d)),1)
  # ix=(ix+2) %% nx
  mx=max( max(B[ix, (ix-d):(ix + d)]) )
  mn=min( min(B[ix, (ix-d):(ix + d)]) )
  if(mn > 0) mn=0
  plot(B[ix, (ix-d):(ix + d)], type="l", lwd=1.5, main="Row: B_true", 
       sub=paste0("i_repl=", i_repl, "  ix=",ix), xlab="Distance, meshes", ylim=c(mn,mx))
  
}

norm(B_LSM-B, type = "2") / norm(B, type = "2") 
norm(S_lcz-B, type = "2") / norm(B, type = "2") 

#------------------------------
# Biases & MAEs in the CRMs/CVMs

C_LSM  = Cov2VarCor(B_LSM)$C
CS_lcz = Cov2VarCor(S_lcz)$C

# Align C, CS_lcz, C_LSM  so that the main diagonal 
# becomes vertical and stands in the middle on the plot (at ix=nmax=nx/2)

aligned_C = C  # init
aligned_C_LSM = C_LSM  # init
aligned_CS_lcz = S_lcz  # init

# Length scales: L0, L1, L2

dd = abs(xx_km - xx_km[nmax])
LL0_t = c(1:nx) ; LL0_s = LL0_t ; LL0_l = LL0_t
                                                
for (ix in 1:nx){
  row     = C[ix,]
  row_cov = B[ix,]
  
  if(l_CRL_COV_plots == 1){
    row4plot = row
  }else if(l_CRL_COV_plots == 2){
    row4plot = row_cov
  }
  alignedRow = symm_cvm_row(row4plot, nx, ix) # for plotting
  aligned_C[ix,] = alignedRow
  
  alignedRow = symm_cvm_row(row, nx, ix)      # for len scales evaluation
  
  LL0_t[ix] = sum( abs(alignedRow)*dx_km ) /2                # macro scale
  
  
  row     = CS_lcz[ix,]
  row_cov = S_lcz[ix,]
  
  if(l_CRL_COV_plots == 1){
    row4plot = row
  }else if(l_CRL_COV_plots == 2){
    row4plot = row_cov
  }
  alignedRow = symm_cvm_row(row4plot, nx, ix)
  
  aligned_CS_lcz[ix,] = alignedRow
  
  alignedRow = symm_cvm_row(row, nx, ix)
  
  LL0_s[ix] = sum( abs(alignedRow)*dx_km ) /2                # macro scale
  
  
  row     = C_LSM[ix,]
  row_cov = B_LSM[ix,]
  
  if(l_CRL_COV_plots == 1){
    row4plot = row
  }else if(l_CRL_COV_plots == 2){
    row4plot = row_cov
  }
  alignedRow = symm_cvm_row(row4plot, nx, ix)
  aligned_C_LSM[ix,] = alignedRow
  
  alignedRow = symm_cvm_row(row, nx, ix)
  
  LL0_l[ix] = sum( abs(alignedRow)*dx_km ) /2                # macro scale
}
  # Plot len scales

if(i_repl == n_repl){
  mx=max(LL0_t, LL0_s, LL0_l)
  plot(LL0_t, ylim=c(0,mx), main = "Macro scale: true (black), \n S_lcz (blue), LSM (red)", type="l")
  lines(LL0_s, col="blue")
  lines(LL0_l, col="red")
}

aaligned_C_Ms[i_repl,]       = apply(aligned_C,      2, mean) # truth

aaligned_C_LSM_MEs[i_repl,]  = apply(aligned_C_LSM - aligned_C,  2, mean)
aaligned_C_LSM_AEs[i_repl,]  = apply(aligned_C_LSM - aligned_C,  2, ExAbs)

aaligned_CS_lcz_MEs[i_repl,] = apply(aligned_CS_lcz - aligned_C, 2, mean)
aaligned_CS_lcz_AEs[i_repl,] = apply(aligned_CS_lcz - aligned_C, 2, ExAbs)

#-------------------------
# Monotonicity of crl C_LSM

ix_c = nmax
eps=3e-2
n_nmon = c(1:nx)

for (ix in 1:nx){
  row = C_LSM[ix,]
  alignedRow = symm_cvm_row(row, nx, ix)
  
  # left half
  ind = 2:ix_c
  growth = alignedRow[ind] > alignedRow[ind-1] - eps
  
  # right half
  ind = ix_c:(nx-1)
  decay = alignedRow[ind] > alignedRow[ind+1] - eps
  
  n_nmon[ix] = sum(!growth) + sum(!decay)
}
nn_nmon[i_repl] = sum(n_nmon)

#---------------------
# Matrix err norms

err_Frob_LSM[i_repl]   = norm(B_LSM-B, type = "F") / norm(B, type = "F") 
err_Frob_S_lcz[i_repl] = norm(S_lcz-B, type = "F") / norm(B, type = "F") 

RelBias_Diag_LSM[i_repl]   = mean(diag(B_LSM-B)) / mean(diag(B))
RelBias_Diag_S_lcz[i_repl] = mean(diag(S_lcz-B)) / mean(diag(B))

RelMSE_Diag_LSM[i_repl]   = mean(diag( (B_LSM-B)^2 )) / mean(diag(B^2))
RelMSE_Diag_S_lcz[i_repl] = mean(diag( (S_lcz-B)^2 )) / mean(diag(B^2))

#------------------------------------------------------------------
#------------------------------------------------------------------
# ANLS
# Perform the secondary KF's anls with:
# 
# B_specified = B_true
# B_specified = S_lcz
# B_specified = B_LSM
# 
# And compute the 3 respective anls-err cvms A
# A = (I-KH) B_true (I-KH)^T + K R K^T
# (B_true is B in this program)

#----------------------------------
# Preparations to the ANLS

# FG=0

x_f = c(1:nx) ;  x_f[] =0

# Generate truth using the true fcst-err mdl: xi=W*N(0,I):
# x_true = x_f - e_f = -xi (minus the FG error)
# Since the CVM of xi is B=W*W^T, we simulate xi as 
# xi=W*gau_N01, whr gau_N01 is the N(0,I) noise

gau_N01 = rnorm(nx, mean=0, sd=1) 
xi = W %*% gau_N01
x_true = -xi

# Specify obs-err SD

var_FG_err = median(diag(B))
sd_obs = sd_obs_rel_FG * sqrt(var_FG_err)

#--------------
# Generate OBS: x_obs, H, R

OBS = gen_obs(x_true, n_obs, sd_obs, repeated_obs_location)

H = OBS$H
R = OBS$R
x_obs = OBS$x_obs
 
#----------------------------------
# ANLS

ANLS_opt = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, B, R, B)
ANLS_LSM = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, B_LSM, R, B)
ANLS_S   = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, S, R, B)
ANLS_lcz = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, S_lcz, R, B)
ANLS_med = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, B_median, R, B)

# plot(diag(ANLS_opt$A), type="l")
# lines(diag(ANLS_LSM$A), col="red")
# lines(diag(ANLS_lcz$A), col="blue")
# lines(diag(ANLS_S$A), col="green")
# 
# plot((ANLS_opt$X_a), type="l")
# lines(x_true, lty=2, col="gold")
# lines((ANLS_LSM$X_a), col="red")
# lines((ANLS_lcz$X_a), col="blue")
# lines((ANLS_S$X_a), col="green")

#----------------------------------
# store anls-err stats

ms_anls_err_Banls_isB_true[i_repl] = Ex2(ANLS_opt$X_a - x_true)
ms_anls_err_Banls_isB_LSM [i_repl] = Ex2(ANLS_LSM$X_a - x_true)
ms_anls_err_Banls_isS_lcz[i_repl]  = Ex2(ANLS_lcz$X_a - x_true)
ms_anls_err_Banls_isB_med[i_repl]  = Ex2(ANLS_med$X_a - x_true)

Amx_diag_Banls_isB_true[i_repl] = mean(diag(ANLS_opt$A))
Amx_diag_Banls_isB_LSM [i_repl] = mean(diag(ANLS_LSM$A))
Amx_diag_Banls_isS_lcz [i_repl] = mean(diag(ANLS_lcz$A))
Amx_diag_Banls_isB_med [i_repl] = mean(diag(ANLS_med$A))

# plot(x_true)
# lines(ANLS_opt$X_a)
# lines(ANLS_LSM$X_a, col="red")
# lines(ANLS_lcz$X_a, col="blue")
# 
# plot(ANLS_lcz$X_a - x_true, col="blue", type="l")
# lines(ANLS_LSM$X_a - x_true, col="red")
# #lines(ANLS_opt$X_a - x_true)
# 
# plot(abs(ANLS_lcz$X_a - x_true) - abs(ANLS_LSM$X_a - x_true), col="blue", type="l")
# abline(h=0)
# mean(abs(ANLS_lcz$X_a - x_true) - abs(ANLS_LSM$X_a - x_true), col="blue", type="l") / mean(abs(ANLS_lcz$X_a - x_true))

#------------------------------------------------------------------
                                       } # end of n_repl BIG external LOOP
#---------------------------------------------------------
#---------------------------------------------------------
# Calibration: the averaged spat-ave crf & spectrum

if(CALIBRATE){
  crf_Msr = crf_SrMs / n_repl
  
  plot(crf_Msr, main="crf_Msr")
  lines(crf_median_1D, main="crf_median_1D")
  
  b_from_crf_Msr = fft(crf_Msr, inverse = FALSE) / nx
  
  max(abs(Im(b_from_crf_Msr)))
  b_from_crf_Msr = Re(b_from_crf_Msr)
  
  b_from_crf_Msr_nrm = b_from_crf_Msr / b_from_crf_Msr[1]
  
  plot(b_from_crf_Msr_nrm, main="b_from_crf_Msr_nrm")
  lines(b_median_1D_nrm, main="b_median_1D_nrm")
  
  b_from_crf_Msr_nrm_half = b_from_crf_Msr_nrm[1:nmaxp1]
  
  # Smoo the space-and-"time" mean spectrum, additionally, over n
  
  nsweep=0
  b_shape_estm = ThreePointSmoo_segment(b_from_crf_Msr_nrm_half, nsweep, maintainMx = FALSE)
  inm=nmax/3
  plot(b_shape_estm[1:inm], main="b_shape_estm (circ), median")
  lines(b_median_1D_nrm[1:inm])
  
  stop("Calibration run OK")
}

# Learn the neural network

# Data to learn from:
#
# TrueSpectra  = array(0, dim=c(nmaxp1, nx_thinned, n_repl))
# EnsmBandVars = array(0, dim=c(nband,  nx_thinned, n_repl))

if(LEARN){
  
  
  
  
}


#---------------------------------------------------------
# OVERALL STATS

# Mean field variances: tue & ensm

xi_Vt_Msr = mean(xi_Vt_Ms)
xi_Ve_Msr = mean(xi_Ve_Ms)

xi_Ve_AEsr   = mean(xi_Ve_AEs) 
xi_Ve_TAD_Msr = mean(xi_Ve_TAD_Ms) 


#=======================================
# Mean cvms of  phi

Test_cvm_phi = F
if(Test_cvm_phi){
  Gamma_true_Msr   = apply(GGamma_true_Ms[,,],   c(1,2), mean)
  cvm_phi_estm_Msr = apply(ccvm_phi_estm_Ms[,,], c(1,2), mean)
  
  mn = min(Gamma_true_Msr, cvm_phi_estm_Msr)
  mx = max(Gamma_true_Msr, cvm_phi_estm_Msr)
  image2D(Gamma_true_Msr, main="Gamma_true_Msr", zlim=c(mn,mx))
  image2D(cvm_phi_estm_Msr, main="cvm_phi_estm_Msr", zlim=c(mn,mx))
  
  image2D(cvm_phi_estm_Msr - Gamma_true_Msr, main="cvm_phi_estm_Msr - Gamma_true_Msr")
  
  rel_err = norm(cvm_phi_estm_Msr - Gamma_true_Msr, type = "F") / 
                               norm(Gamma_true_Msr, type = "F")
  bias = diag(cvm_phi_estm_Msr - Gamma_true_Msr)
  
  mn=min(diag(Gamma_true_Msr), bias)
  mx=max(diag(Gamma_true_Msr), bias)
  
  plot(diag(Gamma_true_Msr), type="l", ylim=c(mn,mx), main="True band variances")
  lines(bias, col="red")
  abline(h=0)
  
  rel_bias = bias / diag(Gamma_true_Msr)
  plot(rel_bias)
  
  message("rel_err estm Gamma_true: ", signif(rel_err,2))
}

#=======================================
# ANLS stats

# (1) A-mx (anls-err CVM) based stats

tt=Amx_diag_Banls_isB_true
ll=Amx_diag_Banls_isB_LSM
ss=Amx_diag_Banls_isS_lcz
vv=Amx_diag_Banls_isB_med


if(n_repl > 1){
  mx=max(max(ss-ll),0)
  mn=min(min(ss-ll),0)
  plot(ss-ll, ylim=c(mn,mx), type="l", col="red",
       xlab="Replicate number", ylab="S_lcz anls-err std   MINUS   LSM anls-err std",
       main="anls_DE(S_lcz) - anls_DE(B_LSM) \n(Difference of anls-err std for the 2 anls)")
  abline(h=0)
}

t=sqrt(mean(tt))
l=sqrt(mean(ll))
s=sqrt(mean(ss))
v=sqrt(mean(vv))

anls_err_Btrue = t
anls_err_LSM = l
anls_err_enkf = s
anls_err_3dv = v

REE_S_lcz =(l-t)/(s-t)
REE_B_med =(l-t)/(v-t)

# sampling error std: bootstrap

n_boo = 1000
rs_boo = c(1:n_boo) # init
rv_boo = c(1:n_boo) # init
for (i in 1:n_boo){
  ii_boo = sample(c(1:n_repl), n_repl, replace = TRUE)
  t=sqrt(mean(tt[ii_boo]))
  l=sqrt(mean(ll[ii_boo]))
  s=sqrt(mean(ss[ii_boo]))
  v=sqrt(mean(vv[ii_boo]))
  rs_boo[i] = (l-t)/(s-t)
  rv_boo[i] = (l-t)/(v-t)
}
REE_S_lcz_samplNoise = sd(rs_boo) 
REE_B_med_samplNoise = sd(rv_boo) 

#-----------------------
# (2) Sample based stats

tt = ms_anls_err_Banls_isB_true
ll = ms_anls_err_Banls_isB_LSM
ss = ms_anls_err_Banls_isS_lcz
vv = ms_anls_err_Banls_isB_med

t=sqrt(mean(tt))
l=sqrt(mean(ll))
s=sqrt(mean(ss))
v=sqrt(mean(vv))

REE_S_lcz_MCarlo =(l-t)/(s-t)
REE_B_med_MCarlo =(l-t)/(v-t)

for (i in 1:n_boo){
  ii_boo = sample(c(1:n_repl), n_repl, replace = TRUE)
  t=sqrt(mean(tt[ii_boo]))
  l=sqrt(mean(ll[ii_boo]))
  s=sqrt(mean(ss[ii_boo]))
  rs_boo[i] = (l-t)/(s-t)
  rv_boo[i] = (l-t)/(v-t)
}
REE_S_lcz_MCarlo_samplNoise = sd(rs_boo) 
REE_B_med_MCarlo_samplNoise = sd(rv_boo) 

#=======================================
# BANDSPACE stats

# 1) errors

band_Ve_MEs_Mr = apply(bband_Ve_MEs, 2, mean) # mean band_Ve 
band_Ve_AEs_Mr = apply(bband_Ve_AEs, 2, mean) # ave band_Ve MAE over n_repl
band_Vt_Ms_Mr  = apply(bband_Vt_Ms,  2, mean) # mean band_Vt
band_Ve_TD_Msr = apply(bband_Ve_TD_Ms, 2, mean) # mean band_Ve theor SD
band_Ve_TAD_Msr= apply(bband_Ve_TAD_Ms, 2, mean) # mean band_Ve theor MD

nb=nband/1
band_Ve_MEs_Mr_rel_Mb  = mean(band_Ve_MEs_Mr[1:nb])  / mean(band_Vt_Ms_Mr[1:nb]) 
band_Ve_AEs_Mr_rel_Mb  = mean(band_Ve_AEs_Mr[1:nb])  / mean(band_Vt_Ms_Mr[1:nb]) 
band_Ve_TAD_Msr_rel_Mb = mean(band_Ve_TAD_Msr[1:nb]) / mean(band_Vt_Ms_Mr[1:nb]) 

band_V_restored_AEs_Mr = apply(bband_V_restored_AEs, 2, mean)
band_V_restored_MEs_Mr = apply(bband_V_restored_MEs, 2, mean)


mn = min(band_Vt_Ms_Mr, band_Ve_AEs_Mr, band_Ve_MEs_Mr, band_V_restored_AEs_Mr, band_V_restored_MEs_Mr)
mx = max(band_Vt_Ms_Mr, band_Ve_AEs_Mr, band_Ve_MEs_Mr, band_V_restored_AEs_Mr, band_V_restored_MEs_Mr)
namefile=paste0("./Out/BandVeErr_nr", n_repl, "u", crftype, "ne", ne, "kapp", kappa, 
                "NSL", NSL, ".png")
png(namefile, width=7.48, height=5.47, units = "in", res=300)
par(mgp=c(2.5, 1, 0))
plot(band_Vt_Ms_Mr, type="l", xlab = "band", ylab = "Band variances",
     main=paste0(
    "Band variances V: Truth,  Errors (MAE and bias) MAD(Ve) \n",
    " Ensm and Restored from b_LSM \n",
    "u=", crftype, " ne=", ne, " NSL=",NSL, " kappa=", kappa, " nr=", n_repl), 
     sub=paste0(
       "Band-mean relative ensm MAE=", signif(band_Ve_AEs_Mr_rel_Mb,3),
       "  Rel bias=", signif(band_Ve_MEs_Mr_rel_Mb,3)),
     ylim=c(mn,mx), col="black", lwd=2)
lines(band_Ve_AEs_Mr, col="orange", lwd=1.5)
lines(band_Ve_MEs_Mr, col="springgreen4", lwd=1.5)
lines(band_V_restored_AEs_Mr, col="orange", lty=2)
lines(band_V_restored_MEs_Mr, col="springgreen4", lty=2)
lines(band_Ve_TAD_Msr, col="green", lty=3, lwd=3)

abline(h=0, lty=3)
leg.txt<-c('Truth', 'Ensm, MAE', 'Ensm, bias', 'Restored, MAE', 'Restored, bias', 'MAD(Ve)')
leg.col<-c("black", "orange", "springgreen4", "orange", "springgreen4", "green")
legend("topright", inset=0, leg.txt, col=leg.col, 
       lwd=c(2,1.5,1.5,1,1,3), lty=c(1,1,1,2,2,3), pch=c(NA,NA,NA,NA,NA),
       pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
dev.off()


# 2) Misfit of restored band_V with band_Ve

bband_V_restored_misfit_As_Mr = apply(bband_V_restored_misfit_As, 2, mean)
bband_V_restored_misfit_Ms_Mr = apply(bband_V_restored_misfit_Ms, 2, mean)
band_Ve_TAD_Ms_Mr = apply(bband_Ve_TAD_Ms, 2, mean)

mn = min(bband_V_restored_misfit_As_Mr, bband_V_restored_misfit_Ms_Mr, band_Ve_TAD_Ms_Mr, band_Vt_Ms_Mr)
mx = max(bband_V_restored_misfit_As_Mr, bband_V_restored_misfit_Ms_Mr, band_Ve_TAD_Ms_Mr, band_Vt_Ms_Mr)
namefile=paste0("./Out/BandVeMisfit_nr", n_repl, "u", crftype, "ne", ne, "kapp", kappa, 
                "NSL", NSL, ".png")
png(namefile, width=7.48, height=5.47, units = "in", res=300)
par(mgp=c(2.5, 1, 0))
plot(band_Vt_Ms_Mr, type="l", xlab = "Band", ylab = "Error",
     main=paste0(
       "Band vars misfits [V_restored - Ve]: MAD (red), bias (blu) \n",
       "True V (black). Sampling error MD(Ve-V_true) (orng) \n",
       "u=", crftype, "ne=", ne, " NSL=",NSL, " kappa=", kappa, " nr=", n_repl), 
     sub=paste0("NB: Misfit MAE should be comparable to RMS(Ve-V_true)"),
     ylim=c(mn,mx), col="black")
lines(bband_V_restored_misfit_As_Mr, col="red")
lines(bband_V_restored_misfit_Ms_Mr, col="blue")
lines(band_Ve_TAD_Ms_Mr, col="orange")
abline(h=0, lty=3)
dev.off()

#=======================================
# SPECTRAL-SPACE stats (LSM) :  Bias, MAE

b_true_Ms_Mr   = apply(bb_true_Ms, 2, mean)
b_LSM_MEs_Mr   = apply(b_LSM_MEs, 2, mean)

b_true_Ms_Mr_Sw = sum(b_true_Ms_Mr)
b_LSM_MEs_Mr_SAw = sum(abs(b_LSM_MEs_Mr))

b_LSM_AEs_Mr = apply(b_LSM_AEs, 2, mean)
b_LSM_AEs_Mr_Sw = sum(b_LSM_AEs_Mr)

nm=nmax/6
mx = max(b_LSM_MEs_Mr[1:nm], b_LSM_AEs_Mr[1:nm], b_true_Ms_Mr[1:nm])
mn = min(b_LSM_MEs_Mr[1:nm], b_LSM_AEs_Mr[1:nm], b_true_Ms_Mr[1:nm])
namefile=paste0("./Out/SpeErr_", "u", crftype, "ne", ne, "nr", n_repl, "kapp", kappa, 
                "NSL", NSL, ".png")
png(namefile, width=7.48, height=5.47, units = "in", res=300)
par(mgp=c(2.5, 1, 0))

plot(b_true_Ms_Mr[1:nm], type="l", ylim=c(mn,mx), col="black", lwd=2,
     main=paste0( "Spectral-space statistics",
                "\n u=", crftype, " ne=", ne, " NSL=",NSL, " kappa=", kappa, " nr=", n_repl ), 
     xlab="n+1", ylab = "Spectral variances",
     sub=paste0( "Total (sum abs): LSM  bias=", signif(b_LSM_MEs_Mr_SAw,3), 
                 "  MAE=", signif(b_LSM_AEs_Mr_Sw,3),
                "   Truth=Var(xi)=", signif(b_true_Ms_Mr_Sw,3) ))
lines(b_LSM_AEs_Mr[1:nm], col="red", lwd=1.5)
lines(b_LSM_MEs_Mr[1:nm], col="red", lty=2, lwd=1.5)

leg.txt<-c('Truth', 'LSM, MAE', "LSM, bias")
leg.col<-c("black", "red", "red")
legend("topright", inset=0, leg.txt, col=leg.col, 
       lwd=c(2,1.5,1.5), lty=c(1,1,2), pch=c(NA,NA,NA),
       pt.lwd=3, cex=1.3, pt.cex=1, bg="white")
abline(h=0, lty=3)
dev.off()

#=======================================
# PHYSICAL-SPACE stats: Correlations: Bias, MAE

aligned_C_Ms_Mr      = apply(aaligned_C_Ms, 2, mean)

aligned_C_LSM_MEs_Mr = apply(aaligned_C_LSM_MEs, 2, mean)
aligned_C_LSM_AEs_Mr = apply(aaligned_C_LSM_AEs, 2, mean)

aligned_CS_lcz_MEs_Mr = apply(aaligned_CS_lcz_MEs, 2, mean)
aligned_CS_lcz_AEs_Mr = apply(aaligned_CS_lcz_AEs, 2, mean)

RelMAE_Diag_LSM = sqrt(mean(RelMSE_Diag_LSM))
RelMAE_Diag_S_lcz = sqrt(mean(RelMSE_Diag_S_lcz))

nonmonot_crl_points_per_i_repl = sum(nn_nmon) / n_repl

# Aligned crfs

nx_mid = round(nx/2)
Dx_align=floor(L_xi_median *10 / dx)
if(Dx_align >= nx/2) Dx_align = floor(nx/2)-1
nx2_align=nx_mid + Dx_align
xx_align = c(0:Dx_align) * dx_km

# CRL Bias & MAE -- averaged over the plotted range of distances 

bias_crl_LSM   = mean(aligned_C_LSM_MEs_Mr[nx_mid:nx2_align])
bias_crl_S_lcz = mean(aligned_CS_lcz_MEs_Mr[nx_mid:nx2_align])

MAE_crl_LSM   = mean(aligned_C_LSM_AEs_Mr[nx_mid:nx2_align])
MAE_crl_S_lcz = mean(aligned_CS_lcz_AEs_Mr[nx_mid:nx2_align])


mn=min(aligned_C_Ms_Mr[nx_mid:nx2_align], aligned_C_LSM_MEs_Mr[nx_mid:nx2_align], aligned_CS_lcz_MEs_Mr[nx_mid:nx2_align])
mx=max(aligned_C_Ms_Mr[nx_mid:nx2_align], aligned_C_LSM_MEs_Mr[nx_mid:nx2_align], aligned_CS_lcz_MEs_Mr[nx_mid:nx2_align])

if(l_CRL_COV_plots == 1){
  namefile=paste0("./Out/CrlErr_", "u", crftype, "_B2S_", B2S_method, "_ne", ne, "nr", n_repl, "kapp", kappa, 
                 "NSL", NSL, ".png")
}else if(l_CRL_COV_plots == 2){
  namefile=paste0("./Out/CovErr_", "u", crftype, "_B2S_", B2S_method, "_ne", ne, "nr", n_repl, "kapp", kappa, 
                  "NSL", NSL, ".png")
}
png(namefile, width=7.48, height=5.47, units = "in", res=300)
par(mgp=c(2.5, 1, 0))

if(l_CRL_COV_plots == 1){
  plot(x=xx_align, 
       y=aligned_C_Ms_Mr[nx_mid:nx2_align], type="l", lwd=2, ylim=c(mn,mx),
       main=paste0("Spatial correlations\n",
                   "u=", crftype, " B2S=", B2S_method, " ne=", ne, " NSL=",NSL, " kappa=", kappa, " nr=", n_repl),
       xlab="Distance, km", ylab="Correlation",
       sub=paste0("Rel variance errors: Bias. LSM: ", signif(mean(RelBias_Diag_LSM),3),
                  "  S_lcz:", signif(mean(RelBias_Diag_S_lcz),3), 
                  ".  MAE. LSM: ",signif(RelMAE_Diag_LSM,3), ".  S_Lcz: ",signif(RelMAE_Diag_S_lcz,3)))
}else if(l_CRL_COV_plots == 2){
  plot(x=xx_align, 
       y=aligned_C_Ms_Mr[nx_mid:nx2_align], type="l", lwd=2, ylim=c(mn,mx),
       main=paste0("Spatial covariances\n",
                   "u=", crftype, " B2S=", B2S_method, " ne=", ne, " NSL=",NSL, " kappa=", kappa, " nr=", n_repl),
       xlab="Distance, km", ylab="Correlation",
       sub=paste0("Rel variance errors: Bias. LSM: ", signif(mean(RelBias_Diag_LSM),3),
                  "  S_lcz:", signif(mean(RelBias_Diag_S_lcz),3), 
                  ".  MAE. LSM: ",signif(RelMAE_Diag_LSM,3), ".  S_Lcz: ",signif(RelMAE_Diag_S_lcz,3)))
}
lines(x=xx_align, y=aligned_C_LSM_AEs_Mr[nx_mid:nx2_align], col="red", lwd=1.5)
lines(x=xx_align, y=aligned_CS_lcz_AEs_Mr[nx_mid:nx2_align], col="blue", lwd=1.5)
lines(x=xx_align, y=aligned_C_LSM_MEs_Mr[nx_mid:nx2_align], col="red", lwd=1.5, lty=2)
lines(x=xx_align, y=aligned_CS_lcz_MEs_Mr[nx_mid:nx2_align], col="blue", lwd=1.5, lty=2)

abline(h=0, lty=3)
leg.txt<-c('Truth', 'LSM, MAE', 'S_lcz, MAE', 'LSM, bias', 'S_lcz, bias')
leg.col<-c("black", "red", "blue", "red", "blue")
legend("topright", inset=0, leg.txt, col=leg.col, 
       lwd=c(2, 1.5, 1.5, 1.5, 1.5), lty=c(1,1,1,2,2), pch=c(NA,NA,NA,NA,NA),
       pt.lwd=3, cex=1.3, pt.cex=1, bg="white")

abline(h=0, lty=3)
dev.off()

#-------------------------------------------------------------------
#-------------------------------------------------------------------

message("Sample MAE(xi_Ve) = ", signif(xi_Ve_AEsr,5), 
        "  Theor MD(xi_Ve) = ", signif(xi_Ve_TAD_Msr,5), 
        " (need be really close in STATIO only)")

message("BANDSPACE: Rel. bias = ", signif(band_Ve_MEs_Mr_rel_Mb,3),
                  "  Rel MAE =",  signif(band_Ve_AEs_Mr_rel_Mb,3),
                  "   Rel theor MAD =",  signif(band_Ve_TAD_Msr_rel_Mb,3))

if(B2S_method == "SVshape") {
  message("rel_err_discard_nullspace_b = ",  
          signif(rel_err_discard_nullspace_b,3))
}

message("SPE-SPACE: \n LSM:  Bias: ", signif(b_LSM_MEs_Mr_SAw,3),
                          "  MAE: ", signif(b_LSM_AEs_Mr_Sw,3),
                          "  Truth=", signif(b_true_Ms_Mr_Sw,3),
                          "   -- sums over spectrum")

message("PHYS-SPACE CRL Bias: \n S_lcz: ", signif(mean(bias_crl_S_lcz),3),
        "  LSM: ", signif(mean(bias_crl_LSM),3))

message("PHYS-SPACE CRL MAE: \n S_lcz: ", signif(mean(MAE_crl_S_lcz),3),
        "  LSM: ", signif(mean(MAE_crl_LSM),3))

message("nonmonot_crl_points_per_i_repl=", nonmonot_crl_points_per_i_repl)

message("PHYS-SPACE Rel Variance Bias: \n S_lcz: ", signif(mean(RelBias_Diag_S_lcz),3),
        "  LSM: ", signif(mean(RelBias_Diag_LSM),3))

message("PHYS-SPACE Vars MAE:  \n S_lcz: ", signif(RelMAE_Diag_S_lcz,3),
        "  LSM: ", signif(RelMAE_Diag_LSM,3))

message("seed=", seed)

message("Anls RMSE:  LSM: ", signif(anls_err_LSM,3), 
        "  EnKF: ", signif(anls_err_enkf,3),
        "  3DV: ", signif(anls_err_3dv,3),
        "  KF: ", signif(anls_err_Btrue,3),
        "   fg_err (rough): ", signif(sd_xi,3)
)

if(kappa > 1 & NSL < 1e3)
  message("   Rm :  diag(A): ",signif(REE_B_med,3), 
          "   M-Carlo: ",signif(REE_B_med_MCarlo,3))

if(kappa > 1 & NSL < 1e3 & n_repl > 1)
  message("---SD(Rm):  diag(A):  ",signif(REE_B_med_samplNoise,1), 
          "   M-Carlo: ",signif(REE_B_med_MCarlo_samplNoise,1))

message("   Rs :  diag(A): ",signif(REE_S_lcz,3), 
        "   M-Carlo: ",signif(REE_S_lcz_MCarlo,3))

if(n_repl > 1) 
  message("---SD(Rs):  diag(A):  ",signif(REE_S_lcz_samplNoise,1), 
          "   M-Carlo: ",signif(REE_S_lcz_MCarlo_samplNoise,1))
