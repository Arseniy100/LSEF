# Run DSADM, Lorenz-2005, Lorenz-2005-TLM, and 
# filters: KF, EKF, EnKF, Var, EnVar, HBEF, and  
# HHBEF -- which includes all the above filters as special cases.
# Write output files to be examined by other programs.
# 
# A Rakitko
# M Tsyrulnikov (current code owner)
# Mar 2019, May 2021, Jun 2023


truth_worlds_and_filters = function(seed){
  
  # Main program called with the seed
  # Returns filters' performance scores: RMSEs:
  #     TRUTH_RMSE, KF_fRMSE, 
  #     HHBEF_fRMSE[1:3] : 1: Var,  2: EnKF,  3: EnVar 
  #     LSEF_fRMSE
  
  library(plot3D)
  library(limSolve)
  library(torch)
  
  source('./Functions/B2S/B2S_SVshape.R')
  source('./Functions/B2S/B2S_NN.R')
  source('./Functions/B2S/TransformSpectrum.R')
  
  source("./Functions/CreateBands/CreateExpqBands.R")
  source('./Functions/CreateBands/Omega_SVD.R')
  source('./Functions/CreateBands/tranfuBandpassExpq.R')
  
  source('./Functions/E2B/E2B.R')
  
  source("./Functions/LinAlgebra/BshiftMx_S1.R")
  source("./Functions/LinAlgebra/Cov2VarCor.R")
  source('./Functions/LinAlgebra/Superdiagonals.R')
  
  source('./Functions/Local/Sigma2WB.R')
  
  source("./Functions/RandProc/SpatAveCrf_S1.R")
  source("./Functions/RandProc/SpatAveCVM_S1.R")
  source('./Functions/RandProc/ThreePointSmoo_segment.R')
  
  source('./Functions/Stats/Ex2_ExAbs.R')
  
  source("./Functions/Varia/bisection.R")
  source('./Functions/Varia/evalFuScaledArg.R')
  source('./Functions/Varia/fitScaleMagn.R')
  source('./Functions/Varia/LinIntpl.R')
  
  source("./Functions/FILTERS/ext2mod_params4dsadm_statio.R")
  source("./Functions/FILTERS/ext2mod_params4transform.R")
  source('./Functions/FILTERS/dsadm_start_field_cvm.R')
  source('./Functions/FILTERS/dsadm_step.R')
  source('./Functions/FILTERS/dsadm_generator.R')
  source('./Functions/FILTERS/predictSpatialFldCVM.R')
  source('./Functions/FILTERS/kf_ekf.R')
  source('./Functions/FILTERS/HHBEF.R')
  source('./Functions/FILTERS/LSEF.R')
  source('./Functions/FILTERS/lclz_matrix.R')
  source('./Functions/FILTERS/symm_pd_mx_sqrt.R')
  source("./Functions/FILTERS/gfunction.R")
  source("./Functions/FILTERS/transform_function.R")
  source("./Functions/FILTERS/rmse.R")
  source('./Functions/FILTERS/lorenz05.R')
  source('./Functions/FILTERS/lorenz05_step.R')
  source('./Functions/FILTERS/lorenz05lin.R')
  source('./Functions/FILTERS/lorenz05lin_step.R')
  source('./Functions/FILTERS/ApplyJacobian_fd.R')
  source('./Functions/FILTERS/SpaSmooCVM.R')
  
  
  path = './Out'
  
  # Check that the two dirs do exist: ./Out and ./Out/Data.
  # Otherwise, uncomment the following two lines:
  #dir.create(path)
  #dir.create(paste0(path,'/DATA'))
  # -----------
  
  config <- read.table('./config.txt', sep = ';')
  
  # NB: time_filter is the number of assimilation steps (analyses), whereas
  #     time_model is the number of model time steps
  #
  #     dt_h is the model time step (in hours), so that
  #     the time interval between the consecutive analyses is dt_h*stride (h)
  
  mode            = config[config$V1 == "mode", 2]
  n               = config[config$V1 == "n", 2]
  stride          = config[config$V1 == "stride", 2]   # model time steps in one asml cycle
  time_filter     = config[config$V1 == "time_filter", 2] # nu of filter time steps (analyses)
  dt_h            = config[config$V1 == "dt_h", 2]
  U_mean          = config[config$V1 == "U_mean", 2]
  V_char          = config[config$V1 == "V_char", 2]
  L_mult          = config[config$V1 == "L_mult", 2]
  NSL             = config[config$V1 == "NSL", 2]
  sd_x            = config[config$V1 == "sd_x", 2]
  sd_U            = config[config$V1 == "sd_U", 2]
  kappa_rho       = config[config$V1 == "kappa_rho", 2]
  pi_rho          = config[config$V1 == "pi_rho", 2]
  kappa_nu        = config[config$V1 == "kappa_nu", 2]
  pi_nu           = config[config$V1 == "pi_nu", 2]
  kappa_sigma     = config[config$V1 == "kappa_sigma", 2]
  m               = config[config$V1 == "m", 2] #OBS GRID MESH SIZE
  sqrt_R          = config[config$V1 == "sqrt_R", 2]
  ne              = config[config$V1 == "ne", 2]
  perform_kf_ekf  = config[config$V1 == "perform_kf_ekf", 2]
  perform_HHBEF   = config[config$V1 == "perform_HHBEF", 2]
  w_cvr           = config[config$V1 == "w_cvr", 2]
  w_evp10         = config[config$V1 == "w_evp10", 2]
  perform_LSEF    = config[config$V1 == "perform_LSEF", 2]
  B2S_method_     = config[config$V1 == "B2S_method_", 2]
  w_evc10         = config[config$V1 == "w_evc10", 2]
  SaveClim        = config[config$V1 == "SaveClim", 2]
  ntime_EE_store  = config[config$V1 == "ntime_EE_store", 2]
  ntime_B_KF_store= config[config$V1 == "ntime_B_KF_store", 2]
  nband           = config[config$V1 == "nband", 2]
  inflation       = config[config$V1 == "inflation", 2]
  inflation_LSEF  = config[config$V1 == "inflation_LSEF", 2]
  spa_shift_mx_S  = config[config$V1 == "spa_shift_mx_S", 2]
  HHBEF_SelfClim  = config[config$V1 == "HHBEF_SelfClim", 2]
  F_Lorenz        = config[config$V1 == "F_Lorenz", 2]
  J_Lorenz        = config[config$V1 == "J_Lorenz", 2]        
  sd_noise_Lorenz = config[config$V1 == "sd_noise_Lorenz", 2]
  model_type      = config[config$V1 == "model_type", 2]
  M               = config[config$V1 == "M", 2] # number of replicates (worlds)
  
  if(B2S_method_ == 1){
    B2S_method = "NN"
  }else if(B2S_method_ == 2){
    B2S_method = "SVshape"
  }
  
  # seeds
  
  seed_for_secondary_fields = seed
  seed_for_filters          = seed + 12345
  
  # model selector 
  
  if(model_type == 1) {
    model_type="DSADM"
    
  } else if(model_type == 2) {
    model_type="Lorenz05"
    
  } else if(model_type == 3) {
    model_type="Lorenz05lin"
    
  } else{
    message("model_type=", model_type)  
    stop("Wrong model_type")
  }
  
  # KF filter type selector
  
  if(perform_kf_ekf == 1) {
    filter_type="KF"
    
  } else if(perform_kf_ekf == 2) {
    filter_type="EKF"
    
  }else{
    filter_type="KF" # "None"
  }
  
  #------------------------------------------------------
  # mode-dependent parameters
  # mode=0: single mdl&flt runs (produce Fields only), 
  # mode=1: predicted BBx (fields' Covariances), 
  # mode=2: predicted BB_KF (KF's background-error Covariances),
  # mode=3: worlds-averaged Fields' Covariances 
  # mode=4: worlds-averaged HHBEF's   Covariances 
  
  # By default and with mode=0: 
  # switch off time-specific Bx & B_flt computations (both recurrent and by worlds-averaging)
  
  if(mode == 0) {
    predict_BBx   = FALSE
    store_EE_EnKF = 0 # =-1 no, =0 only ntime_EE_store   cycles + all times in spat ave form, =1 all EEs
    store_B_KF    =0  # =-1 no, =0 only ntime_B_KF_store cycles + all times in spat ave form, =1 all CVMs
    worlds           = FALSE
    worldsAve_BBx    = FALSE
    worldsAve_BB_HHBEF = FALSE
  }
  
  # mode>0: ensure that ntime is small
  
  if(mode > 0){           # single runs of the truth and the filters
    #if(time_model > 400) time_model = 400
  } 
  
  # mode=1: predict BBx
  
  if(mode == 1){          # estm CVM of x by Predicting them
    M=1
    predict_BBx   = TRUE
    store_EE_EnKF = -1
    store_B_KF = -1
    worlds           = FALSE
    worldsAve_BBx    = FALSE
    worldsAve_BB_HHBEF = FALSE
    
    perform_kf_ekf=-1
    perform_HHBEF=-1
  }
  
  # mode=2: predict BB_KF
  
  if(mode == 2){          # estm CVM of KF' background errors by Predicting them
    M=1
    predict_BBx   = FALSE
    store_EE_EnKF = 1
    store_B_KF = -1
    worlds           = FALSE
    worldsAve_BBx    = FALSE
    worldsAve_BB_HHBEF = FALSE
    
    perform_kf_ekf=1
    perform_HHBEF=-1
    
  }
  
  # mode=3: estm BBx by worlds-averaging
  
  if(mode == 3 & M > 1){ # estm BBx_worldsAve (by averaging over worlds)
    predict_BBx   = FALSE
    store_EE_EnKF = -1
    store_B_KF = -1
    worlds           = TRUE
    worldsAve_BBx    = TRUE
    worldsAve_BB_HHBEF = FALSE
    
    perform_kf_ekf=-1
    perform_HHBEF=-1
  }
  
  # mode=4: estm BB_HHBEF by worlds-averaging
  
  if(mode == 4 & M > 1){ # estm BB_HHBEF_worldsAve (by averaging over worlds)
    predict_BBx   = FALSE
    store_EE_EnKF = -1
    store_B_KF = -1
    worlds           = TRUE
    worldsAve_BBx    = FALSE
    worldsAve_BB_HHBEF = TRUE
    
    perform_kf_ekf=-1
    perform_HHBEF=1
  }
  
  #------------------------------------------------------
  # time_model & synonyms
  
  nx = n
  time_model = time_filter * stride  # nu of model time steps
  
  time=time_model
  ntime_model=time_model
  ntime_cycles=time_filter
  ntime_filter=time_filter
  ntime_model_time_steps=time_model
  dt=dt_h *3600 # s
  dt_model = dt
  dt_filter = dt * stride
  
  spa_shift_mx_Bf = 0 # appeared to be not much useful
  
  #------------------------------------------------------
  # calculated basic parameters
  
  Rem       = 6.37e6           #     Earth radius, m
  a         = 1 / sqrt(2*pi*Rem)
  delta_s   = 2*pi*Rem/n       # spatial mesh, m
  L         = L_mult*delta_s   # mean spatial len scale of x, m
  L_perturb = L*NSL # spatial len scale of the scnd flds, m
  T_char    = L / V_char       # characteristic time scale for x, s
  
  ExtClim   = -HHBEF_SelfClim    # +-1; B_clim is taken from a KF run: short (-1) or long (+1)
  # NB: SelfClim for time_filter=10,000 seems to be almost as good as a 100,000 B_clim.
  
  nmax = nx/2
  nmaxp1=nmax +1
  
  #------------------------------------------------------
  # Derived-type variables containing external parameters
  # 
  # NB: U_i      is the same for all pre-secondary fields and the unperturbed model.
  #     V_char_i is the same for all pre-secondary fields and the unperturbed model.
  #     
  # 1) Tertiary==pre-secondary==pre-transform fields  (stationary)
  # Each tertiary field theta_i (i=1,2,3,4) is characterized by the four External Parameters:
  # (U_i, L_i, V_char_i, SD_i),
  # which are to be converted to the four Model Parameters:
  #  (U_i, rho_i, nu_i, sigma_i)
  #  
  # (i=1) theta_1* = U*
  
  tert1_extpar = list(U=U_mean, L=L_perturb, V_char=V_char, SD=sd_U)
  
  # (i=2) theta_2* =  rho*
  
  sd_tert2=log(kappa_rho)
  tert2_extpar = list(U=U_mean, L=L_perturb, V_char=V_char, SD=sd_tert2)
  
  
  # (i=3) theta_3* =  nu*
  
  sd_tert3=log(kappa_nu)
  tert3_extpar = list(U=U_mean, L=L_perturb, V_char=V_char, SD=sd_tert3)
  
  
  # (i=4) theta_4* = sigma*
  
  sd_tert4=log(kappa_sigma)
  tert4_extpar = list(U=U_mean, L=L_perturb, V_char=V_char, SD=sd_tert4)
  
  # 2) unperturbed model for x (stationary)
  
  x_unpert_extpar = list(U=U_mean, L=L, V_char=V_char, SD=sd_x)
  
  #---------
  # From ext params (U, L, V_char, SD) to model params (U, rho, nu, sigma)
  
  tert1_modpar = ext2mod_params4dsadm_statio(tert1_extpar, Rem, nmax)
  tert2_modpar = ext2mod_params4dsadm_statio(tert2_extpar, Rem, nmax)
  tert3_modpar = ext2mod_params4dsadm_statio(tert3_extpar, Rem, nmax)
  tert4_modpar = ext2mod_params4dsadm_statio(tert4_extpar, Rem, nmax)
  x_unpert_modpar = ext2mod_params4dsadm_statio(x_unpert_extpar, Rem, nmax)
  
  #---------
  # Transform params:
  
  epsilon_rho = ext2mod_params4transform(pi_rho, kappa_rho)
  epsilon_nu  = ext2mod_params4transform(pi_nu,  kappa_nu )
  
  pi_="piZero"
  if(pi_rho == 0.02 | pi_nu == 0.01){
    pi_="pi002001"
  }else if(pi_rho == 0.04 | pi_nu == 0.04){
    pi_="pi004004_U20"
  }
  
  #------------------------------------------------------
  # Obs-err CVM (diagonal) 
  
  R_diag = rep(sqrt_R^2, n%/%m + min(1,n%%m)) # vector on the diagonal of the R mx
  
  #------------------------------------------------------
  # initialize the output variable "filter"
  
  filter = list() # the output variable
  
  # create list for parameters (for output file)
  
  parameters = list()
  
  parameters$mode                 <- mode
  parameters$n                    <- n
  parameters$seed                 <- seed
  parameters$stride               <- stride 
  parameters$time_filter          <- time_filter
  parameters$dt_filter            <- dt_filter  # filter time step, s
  parameters$U_mean               <- U_mean
  parameters$V_char               <- V_char
  parameters$L_mean               <- L
  parameters$L_perturb            <- L_perturb
  parameters$sd_x                 <- sd_x
  parameters$sd_U                 <- sd_U
  parameters$kappa_rho            <- kappa_rho
  parameters$pi_rho               <- pi_rho
  parameters$kappa_nu             <- kappa_nu
  parameters$pi_nu                <- pi_nu
  parameters$kappa_sigma          <- kappa_sigma
  parameters$mesh_obs             <- m
  parameters$sqrt_R               <- sqrt_R
  
  nparam_not_HHBEF_specific = 19
  nparam_not_LSEF_specific = nparam_not_HHBEF_specific
  
  parameters$ne                   <- ne
  parameters$inflation            <- inflation 
  parameters$inflation_LSEF       <- inflation_LSEF
  parameters$spa_shift_mx_S       <- spa_shift_mx_S
  parameters$perform_kf_ekf       <- perform_kf_ekf
  parameters$SaveClim             <- SaveClim
  parameters$ntime_EE_store       <- ntime_EE_store
  parameters$ntime_B_KF_store     <- ntime_B_KF_store
  parameters$perform_HHBEF        <- perform_HHBEF
  parameters$perform_LSEF         <- perform_LSEF
  parameters$B2S_method           <- B2S_method
  parameters$nband                <- nband
  parameters$HHBEF_SelfClim       <- HHBEF_SelfClim
  parameters$w_cvr                <- w_cvr  
  parameters$w_evp10              <- w_evp10
  parameters$w_evc10              <- w_evc10
  parameters$F_Lorenz             <- F_Lorenz
  parameters$J_Lorenz             <- J_Lorenz
  parameters$model_type           <- model_type
  parameters$M                    <- M
  
  # Store the Parameters in the output "filter" variable
  
  filter$parameters = parameters
  
  
  set.seed(seed_for_secondary_fields)
  
  #------------------------------------------------------
  # DSADM: generate SECONDARY fields.
  # Technique: generate TERTIARY fields and (point-wise) transform them to the SECONDARY fields.
  # NB: Only ONE realization of each scnd field is generated: N_scnd=1
  
  if( model_type == "DSADM" ){ 
    
    message("Generate SECONDARY fields")
    
    N_scnd=1
    forcing = TRUE
    
    # Generate tert1(t,s)
    
    if(tert1_extpar$SD != 0){
      start_field = dsadm_start_field_cvm(tert1_modpar, n, Rem)$x_start
      
      tert_field  = dsadm_generator(as.matrix(start_field, nrow=n, ncol=N_scnd), 
                                    n, N_scnd, ntime_model, dt, 
                                    matrix(tert1_modpar$U,     nrow = n, ncol = time_model),
                                    matrix(tert1_modpar$rho,   nrow = n, ncol = time_model),
                                    matrix(tert1_modpar$nu,    nrow = n, ncol = time_model),
                                    matrix(tert1_modpar$sigma, nrow = n, ncol = time_model), Rem, forcing)
      
      # Transform tert1(t,s) --> UU(t,s)
      
      UU = x_unpert_modpar$U + tert_field
    }else{
      UU = matrix(tert1_modpar$U, nrow = n, ncol = time_model)
    }
    
    #  Generate tert2(t,s)
    
    if(tert2_extpar$SD != 0){
      start_field = dsadm_start_field_cvm(tert2_modpar, n, Rem)$x_start
      
      tert_field  = dsadm_generator(start_field, n, N_scnd, ntime_model, dt, 
                                    matrix(tert2_modpar$U,     nrow = n, ncol = time_model),
                                    matrix(tert2_modpar$rho,   nrow = n, ncol = time_model),
                                    matrix(tert2_modpar$nu,    nrow = n, ncol = time_model),
                                    matrix(tert2_modpar$sigma, nrow = n, ncol = time_model), Rem, forcing)
      
      # Transform tert2(t,s) --> rho(t,s)  
      
      rrho = x_unpert_modpar$rho * transform_function(tert_field, epsilon_rho)
    }else{
      rrho = matrix(x_unpert_modpar$rho, nrow = n, ncol = time_model)
    }
    
    #  Generate tert3(t,s)
    
    if(tert3_extpar$SD != 0){
      start_field = dsadm_start_field_cvm(tert3_modpar, n, Rem)$x_start
      
      tert_field  = dsadm_generator(start_field, n, N_scnd, ntime_model, dt, 
                                    matrix(tert3_modpar$U,     nrow = n, ncol = time_model),
                                    matrix(tert3_modpar$rho,   nrow = n, ncol = time_model),
                                    matrix(tert3_modpar$nu,    nrow = n, ncol = time_model),
                                    matrix(tert3_modpar$sigma, nrow = n, ncol = time_model), Rem, forcing)
      # Transform tert3(t,s) --> nu(t,s)
      
      nnu = x_unpert_modpar$nu * transform_function(tert_field, epsilon_nu)
    }else{
      nnu = matrix(x_unpert_modpar$nu, nrow = n, ncol = time_model)
    }
    
    # Generate tert4(t,s) 
    
    if(tert4_extpar$SD != 0){
      start_field = dsadm_start_field_cvm(tert4_modpar, n, Rem)$x_start
      
      tert_field  = dsadm_generator(start_field, n, N_scnd, ntime_model, dt, 
                                    matrix(tert4_modpar$U,     nrow = n, ncol = time_model),
                                    matrix(tert4_modpar$rho,   nrow = n, ncol = time_model),
                                    matrix(tert4_modpar$nu,    nrow = n, ncol = time_model),
                                    matrix(tert4_modpar$sigma, nrow = n, ncol = time_model), Rem, forcing)
      # Transform tert4(t,s) --> sigma(t,s)
      
      ssigma = x_unpert_modpar$sigma * transform_function(tert_field)
    }else{
      ssigma = matrix(x_unpert_modpar$sigma, nrow = n, ncol = time_model)
    }
    
    # store current parameters$... to be compared with in a follow-up run
    # (to save time when running the model in the same model setup)
    
  }
  
  #------------------------------------------------------
  # Setup params, ini field, and noise for Lorenz05
  
  if(model_type == "Lorenz05" | model_type == "Lorenz05lin"){
    
    dt_atm_h = min(3, dt_h)  #h
    if(F_Lorenz > 32 & n > 60) dt_atm_h = dt_atm_h /2 # for stability, may appear to be reduced
    
    dt_Lorenz = dt_atm_h/6*0.05 # unitless, "Lorenz time" (6h atmospheric time ~ 0.05 Lorenz time units)
    
    seed_ini_cond_Lorenz=seed
    seed_noise_Lorenz   =seed *12.3456
    
    # Specify ini cond that *minimize* the initial transient
    
    assumed_mean = 1.2 * F_Lorenz^(1/3) # from Lorenz-05, p.1577, bef Eq(5)
    assumed_ms = assumed_mean * F_Lorenz # from Lorenz-05, p.1577, bef Eq(5)
    assumed_rms = sqrt(assumed_ms)
    assumed_var=assumed_ms - assumed_mean^2
    assumed_sd = sqrt(assumed_var)
    
    set.seed(seed_ini_cond_Lorenz)
    X1=rnorm(n, mean=assumed_mean, sd=assumed_sd) # ini condition
    
    # smooth ini cond to reduce the Lorenz-05's initial transient period
    
    nsweep=ceiling( 3*(60/n) * J_Lorenz)
    xx2=X1
    
    if(nsweep >0){
      
      for(sweep in 1:nsweep){
        xx1=xx2
        
        for (i in 1:n){
          im1=i-1
          if(im1 < 1) im1=im1+n
          
          ip1=i+1
          if(ip1 > n) ip1=ip1-n
          
          xx2[i]=(xx1[im1] + 2* xx1[i] + xx1[ip1]) /4
        }
      }
      
      X1=xx2 * max(nsweep/3, 1) # *nsweep/3 -- because smoothing reduces the magnitude
    }
    
    if(model_type == "Lorenz05lin"){
      X1_lin=X1 /5  # ini field to start the fcst t=1:2 in the filter;  5 is smth >1
    }
    
    # Specify system noise (model error) space-time field
    #  for Lorenz 05 & Lorenz05lin
    # Note that sd_noise_Lorenz is specified per 6h atm time, 2pi/60 mesh size
    # while the noise is white both in space and in time.
    # The whiteness implies that 
    # 
    # Var x_dscr(t,s) = 1/(delta_t * delta_s) ~ n/delta_t
    # 
    # For the specified  delta_t  and  n  grid points, we obtain
    # 
    # Var x_dscr = Var x_dscr_6h_60points * (n/60) / (delta_t / delta_t_6h)  ==>
    # sd_noise = sd_noise_Lorenz * sqrt( (n/60) / (delta_t / delta_t_6h) )
    
    dt_h_6h = 6 # h
    sd_noise = sd_noise_Lorenz * sqrt( (n/60) / (dt_h / dt_h_6h) )
    
    set.seed(seed_noise_Lorenz)
    noise_Lorenz=rnorm(time_model*n, mean=0, sd=sd_noise)
    
    noise_Lorenz=matrix(noise_Lorenz, nrow=n)  # model error (system noise)
    noise_Lorenz_zero=noise_Lorenz
    noise_Lorenz_zero[,]=0   # for the reference trajectory
  }
  
  #------------------------------------------------------
  # Specify OBS network in space & time
  
  ind_obs_space = seq(1, n, m)             # spatial obs network
  ind_time_anls_1 = 2  # 1st mdl time step when anls is performed (1 or 2)
  ind_time_anls = seq(ind_time_anls_1, time_model, stride) # model time steps when anls is to be done
  
  #------------------------------------------------------
  # generate one-world TRUTH
  
  N_truth=1  # number of realizations when generating the truth
  
  # 1) Ini conditions
  #    X_mdl_start  is the state vector at t=1
  
  if(model_type == "DSADM"){
    
    Start = dsadm_start_field_cvm(x_unpert_modpar, n, Rem)
    X_mdl_start = Start$x_start
    Bx_start = Start$CVM
    
  }else if(model_type == "Lorenz05" | model_type == "Lorenz05lin"){
    X_mdl_start=X1
  }
  
  # 2) run the model
  
  if(model_type == "DSADM"){
    message("Main model run")
    forcing = TRUE
    X_true_mdl_steps = dsadm_generator(X_mdl_start, n, N_truth, ntime_model, dt, 
                                       UU, rrho, nnu, ssigma, Rem, forcing)
    
    if(predict_BBx){
      message("predict BBx")
      BBx = predictSpatialFldCVM(ntime_filter, n, dt, stride, Rem,
                                 UU, rrho, nnu, ssigma,
                                 Bx_start)
      filter$BBx   =  BBx  # anls steps
    }
    
    
  }else if(model_type == "Lorenz05"){ # NLIN mdl
    X_true_mdl_steps = lorenz05(X_mdl_start, n, time_model, dt_Lorenz, 
                                F_Lorenz, J_Lorenz, noise_Lorenz)  # perturbed
    
  }else if(model_type == "Lorenz05lin"){ # LINEARIZED mdl
    
    # (1) Reference trajectory for lorenz05lin  (non-perturbed !) and
    #     lin-lorenz05 TRUE PERTURBATION trajectory (UU)
    # Let the initial prtbn be =0
    
    n_pert=1 # one truth
    U_start=matrix(0, nrow=n, ncol=n_pert) # ncol=n_pert is one perturbation
    noise_spatim=array(noise_Lorenz, dim=c(n,n_pert, time_model))
    
    LIN = lorenz05lin(X_mdl_start, n, U_start, n_pert, time_model, dt_Lorenz, F_Lorenz, J_Lorenz, noise_spatim)
    UU              = LIN$UUU[,n_pert,]
    X_ref_mdl_steps = LIN$XX_ref
    
    # (3) The lin-lorenz05 TRUTH (X_ref_mdl_steps + UU)
    
    X_true_mdl_steps = X_ref_mdl_steps + UU
  }
  
  # TRUTH evaluated at the anls times
  
  X_true = X_true_mdl_steps[,ind_time_anls] # truth at the anls times
  TRUTH_RMSE = rmse(X_true[,], 0)
  message("TRUTH_RMSE=", signif(TRUTH_RMSE,4))
  
  ind_time_flt_plot = seq(from=1, to=min(400, time_model), by=stride)
  
  image2D(X_true_mdl_steps[,ind_time_flt_plot], main="X_true", xlab="Space", ylab="Time")
  image2D(rrho  [,ind_time_flt_plot], main="rho  ", xlab="Space", ylab="Time")
  image2D(nnu   [,ind_time_flt_plot], main="nu   ", xlab="Space", ylab="Time")
  image2D(ssigma[,ind_time_flt_plot], main="sigma", xlab="Space", ylab="Time")
  image2D(X_true_mdl_steps[,ind_time_flt_plot], main="X_true", xlab="Space", ylab="Time")
  
  
  kappa=max(kappa_rho, kappa_nu, kappa_sigma)
  namefile=paste0("./Out/X_true_seed",seed, "_L_mult", L_mult, "_kap", kappa, "_NSL", NSL,".png")
  png(namefile, width=7.48, height=5.47, units = "in", res=300)
  par(mgp=c(2.5, 1, 0))
  image2D(X_true_mdl_steps[,ind_time_flt_plot], main="X_true", xlab="Space", ylab="Time")
  abline(h=0, lty=3)
  dev.off()
  
  if(time_model > 10000){
    rarefa_plt = 100
    ind_time_flt_plot = seq(from=1, to=time_model, by=stride*rarefa_plt)
    
    image2D(rrho  [,ind_time_flt_plot], main="rho long ", xlab="Space", ylab="Time")
    image2D(nnu   [,ind_time_flt_plot], main="nu long ", xlab="Space", ylab="Time")
    image2D(ssigma[,ind_time_flt_plot], main="sigma long", xlab="Space", ylab="Time")
    image2D(X_true_mdl_steps[,ind_time_flt_plot], 
            main=paste0("X_true (every ", rarefa_plt, "th model \ntime step)"), 
            xlab="Space", ylab="Time")
    
  }
  
  max(abs(X_true_mdl_steps))
  mean(abs(X_true_mdl_steps))
  
  #-----------------------------------------------
  # Store the Fields in the output "filter" variable
  # All fields are written at the anls times only.
  
  if(model_type == "DSADM"){
    
    filter$rrho   = rrho  [,ind_time_anls]
    filter$nnu    = nnu   [,ind_time_anls]
    filter$UU     = UU    [,ind_time_anls]
    filter$ssigma = ssigma[,ind_time_anls]
  }
  
  filter$X_true_anlstimes = X_true
  
  #------------------------------------------------------
  # Generate OBS
  
  message("Generate OBS")
  
  n_obs=n%/%m+min(1,n%%m)
  OBS_NOISE = matrix(rnorm(n_obs*time_model, mean=0, sd=sqrt_R),
                     nrow=n_obs,
                     ncol=time_model)  # at the moment, OBS are generated at all mdl time steps (to be changed)
  OBS = X_true_mdl_steps[ind_obs_space,] + OBS_NOISE
  
  #------------------------------------------------------
  # FILTERING
  
  if(ntime_EE_store > (ntime_filter*3/4 -1)) ntime_EE_store=ntime_filter*3/4 -1
  if(ntime_EE_store == 0) store_EE_EnKF = -1
  
  if(ntime_B_KF_store > (ntime_filter*3/4 -1)) ntime_B_KF_store=ntime_filter*3/4 -1
  if(ntime_B_KF_store == 0) store_B_KF = -1
  
  #------------------------------------------------------
  # One-world KF/EKF
  
  parameters_KF=parameters
  KF_fRMSE = 0

  if(perform_kf_ekf > 0){
    
    message("Run KF")
    
    X_flt_start = X_mdl_start # the 1st FILTER fcst starts from the truth, X_flt_start at t=0
    A_start = diag(n);   A_start[,]=0 # correspondingly, the time=0 anls is error-free
    
    KF_res = KF_EKF(ntime_filter, n, dt, 
                    ind_time_anls_1, stride, ind_time_anls, 
                    ind_obs_space, Rem,
                    UU, rrho, nnu, ssigma,  
                    F_Lorenz, J_Lorenz, sd_noise,
                    R_diag, m, OBS, 
                    X_flt_start, A_start, 
                    model_type, filter_type, 
                    ntime_B_KF_store, store_B_KF)    
    
    KF_fRMSE  = rmse(KF_res$XXf[,], X_true[,])
    message("KF_fRMSE=", signif(KF_fRMSE,4))
    
    B_KF = KF_res$BB          #   [1:nx, 1:nx, 1:ntime_B_KF_store]
    ntime_B_KF = dim(B_KF)[3]
    spat_ave_cvfs = KF_res$spat_ave_cvfs
    
    KF_aRMSE  = rmse(KF_res$XXa[,], X_true[,]) 
    KF_aRMSE
    
    # CLIM CVM
    
    B_clim = SpatAveCVM_S1(KF_res$B_mean)
    # image2D(B_clim)
  
    # mn=min(KF_res$XXf[,1:min(200, time_filter)],
    #        KF_res$XXa[,1:min(200, time_filter)],
    #        X_true    [,1:min(200, time_filter)])
    # mx=max(KF_res$XXf[,1:min(200, time_filter)],
    #        KF_res$XXa[,1:min(200, time_filter)],
    #        X_true    [,1:min(200, time_filter)])
    # image2D(X_true    [,1:min(200, time_filter)], main="X_true", zlim=c(mn,mx))
    # image2D(KF_res$XXf[,1:min(200, time_filter)], main="XXf", zlim=c(mn,mx))
    # image2D(KF_res$XXa[,1:min(200, time_filter)], main="XXa", zlim=c(mn,mx))
    # image2D(X_true    [,1:min(200, time_filter)], main="X_true", zlim=c(mn,mx))
    
    # Test (permanent) KF_EKF:
    
    B_var_mean = sum(diag(KF_res$B_mean)) /n
    sqrt_B_var_mean=sqrt(B_var_mean)
    sqrt_B_var_mean

    A_var_mean = sum(diag(KF_res$A_mean)) /n
    sqrt_A_var_mean=sqrt(A_var_mean)
    sqrt_A_var_mean

    Consistensy_test = F
    if(Consistensy_test){
      print("KF_fRMSE: real fcst-err RMSE")
      print(KF_fRMSE)
      print("sqrt_B_var_mean: fcst-err RMSE assumed by the filter")
      print(sqrt_B_var_mean)
      print("-- sh.be close to each other")
      
      print("KF_aRMSE: real anls-err RMSE")
      print(KF_aRMSE)
      print("sqrt__var_mean: anls-err RMSE assumed by the filter")
      print(sqrt_A_var_mean)
      print("-- sh.be close to each other")
    }
    
    #--------------------------------
    # calc b_shape
   
    crf_mean = SpatAveCrf_S1(B_clim)$crf
    # plot(crf_mean, main="crf_mean")
    
    b_mean = Re( fft(crf_mean, inverse = FALSE) / nx )
    
    b_mean_nrm = b_mean / b_mean[1]
    b_shape = b_mean_nrm[1:nmaxp1]
    # plot(b_shape, main="b_shape")
    
    #--------------------------------
    # Save CLIM data to a file
    
    if(SaveClim > 0){   
       B_clim_long = B_clim

      B_clim_b_shape_and_params = 
        list(B_clim_long=B_clim_long, b_shape=b_shape, parameters=parameters)
      filename=paste0("Clim_", "nx", nx,
                      "_L_mult", L_mult,
                      "_tenKappa", kappa_rho*10, 
                      "_tenNSL", NSL*10, 
                      "_", pi_,
                      "_m", m, 
                      "_sqrtR", sqrt_R,  
                      "_sdx", sd_x,
                      ".RData")
      save(B_clim_b_shape_and_params, file=filename)
    }
    
    filter$KF = KF_res
    
    #-----------------------------------
    # Save KF_data: B_KF,  b_shape, ..

    save_KF_data = TRUE
    if(save_KF_data){
      filename=paste0("KF_data", "_nt", ntime_B_KF_store,
                      "_kap", kappa_rho, 
                      "_mu", NSL, "_pi", pi_rho*100,
                      # "_m", m, "_e", sqrt_R, "_sdx", sd_x,
                      ".RData")
      KF_data = list(B_KF=B_KF, B_clim=B_clim, 
                       b_shape=b_shape, spat_ave_cvfs=spat_ave_cvfs)
      save(KF_data, file=filename)    
    }
  }
  #-----------------------------------------------
  # Lcz -- for EnKF and for plotting
  
  if(ne < 10){               
    lclz_mult = 1.2        # 5
  }else if(ne < 20){        
    lclz_mult = 1.4        # 10
  } else if(ne < 40){
    lclz_mult = 1.6        # 20
  } else if(ne < 80){
    lclz_mult = 2.2        # 40
  } else {               
    lclz_mult = 2.6        # >= 80
  } 
  
  L_lcz = lclz_mult * L
  C_lcz = lclz_matrix(n, L_lcz, Rem)  # lcz mx
  
  #-----------------------------------------------
  # One-world HHBEF (Generalized HBEF, which includes EnKF, HBEF, Var, and EnVar as special cases)
  
  n_HBEF = 3
  HHBEF_fRMSE = c(1:n_HBEF)
  HHBEF_fRMSE[] = 0
  
  if(perform_HHBEF > 0){
    
    message("Run HHBEF")
    
    X_flt_start = X_mdl_start # the 1st flt fcst starts from the truth (at step 0)
    
    # select B_clim. If 
    
    if(HHBEF_SelfClim > 0){
      
      # B_clim = B_clim  # KF's CVM from the current run
      
    }else if(w_cvr > 0){
      
      ClimFile=paste0("Clim_", "nx", nx,
                      "_L_mult", L_mult,
                      "_tenKappa", kappa_rho*10, 
                      "_tenNSL", NSL*10, "_", pi_,
                      "_m", m, "_sqrtR", sqrt_R, "_sdx", sd_x,
                      ".RData")
      load(ClimFile, verbose= TRUE) # Loading  B_clim_b_shape_and_params
      
      B_clim = B_clim_b_shape_and_params$B_clim_long
    }else{
      ClimFile = NULL
    }
    
    # Initial conditions
    
    Ba_start = matrix(0, nrow=nx, ncol=nx) # start flt at t=0 from the truth, as in the KF
    A_start = Ba_start # assume no obs at t=0
    
    # Compute the step-0 anls ensm members with zero perturbations
    
    # sqrt_A_start = symm_pd_mx_sqrt(A_start)$sq
    # N01 = matrix(rnorm(n*ne, mean=0, sd=1), nrow=n, ncol=ne)
    # dXae_start = sqrt_A_start %*% N01    # perturbations yet
    # Xae_start = X_flt_start + dXae_start # ensm members
    
    Xae_start = matrix(0, nrow=nx, ncol=ne)
    for(ie in 1:ne){
      Xae_start[,ie] = X_flt_start
    }
    
    # loop over 3 HBEF versions: Var, EnKF, EnVar
    
    i_HBEF_EnKF = 2
    ww_evp10 = c(0, 1, 0.5)
    
    if(perform_HHBEF == 3){ # =1: Var, =2: EnKF, =3: EnVar
      i1 = 1 
      i2 = n_HBEF
    }else{ # perform_HHBEF = 1: only EnKF
      i1 = i_HBEF_EnKF 
      i2 = i_HBEF_EnKF
    }
    
    detFc_emean = 1 # deterministic fcst (1) or ensm mean (2)?
    
    for(i_HBEF in i1:i2){
      set.seed(seed_for_filters)
      w_evp10 = ww_evp10[i_HBEF]
      
      HHBEF_res = HHBEF(ntime_filter, nx, dt, 
                        ind_time_anls_1, stride, ind_time_anls,  
                        ind_obs_space, Rem,  
                        UU, rrho, nnu, ssigma,  
                        F_Lorenz, J_Lorenz, sd_noise,
                        R_diag, m, OBS, 
                        X_flt_start, Ba_start, Xae_start, B_clim,
                        ne, w_cvr, w_evp10, detFc_emean, 
                        inflation, spa_shift_mx_Bf, spa_shift_mx_S, C_lcz, 
                        ntime_EE_store, store_EE_EnKF, 
                        model_type)
      
      HHBEF_fRMSE[i_HBEF]  = rmse(HHBEF_res$XXf[,], X_true[,])
      message("i_HBEF=", i_HBEF, " HHBEF_fRMSE=", signif(HHBEF_fRMSE[i_HBEF],4))
      
      S_mean = HHBEF_res$S_mean
      
      # mean fc spread
      
      spread = sqrt( mean(diag(S_mean)) )
      message("i_HBEF=", i_HBEF, " spread=", signif(spread,3))
      
      if(i_HBEF == i_HBEF_EnKF){
        spat_ave_cvfs_EnKF = HHBEF_res$spat_ave_cvfs 
        EE = HHBEF_res$EE
        
        # Calc  b_shape_EnKF  from  S_mean
        
        crf_mean = SpatAveCrf_S1(S_mean)$crf
        plot(crf_mean, main="crf_mean")
        
        b_mean = Re( fft(crf_mean, inverse = FALSE) / nx )
        b_mean_nrm = b_mean / b_mean[1]
        b_shape_EnKF = b_mean_nrm[1:nmaxp1]
        # plot(b_shape_EnKF, main="b_shape")
        # lines(b_shape)
      }
    }
    
    HHBEF_one_world = HHBEF_res
    filter$HHBEF_one_world = HHBEF_one_world
    B_EnKF = HHBEF_res$EE
    
    # image2D(HHBEF_res$XXf[,1:min(200, time_filter)], main="XXf")
    # image2D(HHBEF_res$XXa[,1:min(200, time_filter)], main="XXa")
    # image2D(X_true       [,1:min(200, time_filter)], main="X_true")
    # image2D(HHBEF_res$XXf[,1:min(200, time_filter)], main="XXf")
    
    # HHBEF_fRMSE_assumed = sqrt( mean( diag( HHBEF_res$B_mean ) ) )
    # HHBEF_fRMSE_assumed
    # 
    # HHBEF_aRMSE  = rmse(HHBEF_res$XXa[,], X_true[,]) 
    # HHBEF_aRMSE
    
    #----------------------
    # save EnKF's B_EnKF and spat averaged spat covs (cvfs)
    
    save_EnKF_data = TRUE
    if(save_EnKF_data){
      filename=paste0("EnKF_data", "_nt", ntime_EE_store,
                      "_kap", kappa_rho, 
                      "_mu", NSL, "_pi", pi_rho*100,
                      # "_m", m, "_e", sqrt_R, "_sdx", sd_x,
                      ".RData")
      
      EnKF_data = list(EE=EE, S_mean=S_mean, b_shape_EnKF=b_shape_EnKF,
                      spat_ave_cvfs_EnKF = spat_ave_cvfs_EnKF)
      save(EnKF_data, file=filename) 
    }
  }
  
  #-----------------------------------------------
  # One-world LSEF 
  
  LSEF_fRMSE = 0
  
  if(perform_LSEF == 1){
    
    message("Run LSEF")
    
    X_flt_start = X_mdl_start # the 1st model fcst starts from the truth (at step 0)
    
    if(B2S_method == "NN"){  # Read a pre-trained NN from a file
      NN = torch_load("NN.pt")
    }else{
      NN = NULL
    }
    
    # select B_clim (needed for b_shape -- needed by B2S_SVshape)
    
    ClimFile=paste0("Clim_", "nx", nx,
                    "_L_mult", L_mult,
                    "_tenKappa", kappa_rho*10, 
                    "_tenNSL", NSL*10, "_", pi_,
                    "_m", m, "_sqrtR", sqrt_R, "_sdx", sd_x,
                    ".RData")
    load(ClimFile, verbose= TRUE) # Loading  B_clim_b_shape_and_params
    
    B_clim  = B_clim_b_shape_and_params$B_clim_long
    b_shape = B_clim_b_shape_and_params$b_shape
    
    # Initial conditions
    
    Xae_start = matrix(0, nrow=nx, ncol=ne)
    for(ie in 1:ne){
      Xae_start[,ie] = X_flt_start
    }
    
    true_field_available = TRUE
    
    set.seed(seed_for_filters)
    
    LSEF_res = LSEF(ntime_filter, nx, dt, 
                    ind_time_anls_1, stride, ind_time_anls,  
                    ind_obs_space, Rem,  
                    UU, rrho, nnu, ssigma,  
                    F_Lorenz, J_Lorenz, sd_noise,
                    R_diag, m, OBS, 
                    B2S_method, NN, 
                    X_flt_start,  Xae_start, B_clim, w_evc10,
                    ne, nband, b_shape, inflation_LSEF, C_lcz,
                    true_field_available, X_true_mdl_steps,
                    model_type)
    
    LSEF_fRMSE  = rmse(LSEF_res$XXf[,], X_true[,])
    message("LSEF_fRMSE=", signif(LSEF_fRMSE, 4))
    
    S_mean = HHBEF_res$S_mean
    
    # mean fc spread
    
    spread = sqrt( mean(diag(S_mean)) )
    message("LSEF spread=", signif(spread,3))
    
    LSEF_one_world = LSEF_res
    
    filter$LSEF_one_world = LSEF_one_world
    
    # image2D(LSEF_res$XXf[,1:min(200, time_filter)], main="XXf")
    # image2D(LSEF_res$XXa[,1:min(200, time_filter)], main="XXa")
    # image2D(X_true      [,1:min(200, time_filter)], main="X_true")
    # image2D(LSEF_res$XXf[,1:min(200, time_filter)], main="XXf")
    
    LSEF_aRMSE  = rmse(LSEF_res$XXa[,], X_true[,]) 
    LSEF_aRMSE
    
  }
  
  #------------------------------------------------------
  # Worlds: generate TRUTH & perform filtering
  
  if(worlds){
    
    message("Run worlds")
    
    X     = list()
    X_HHBEF = list()
    
    set.seed(seed_for_filters)
    
    for(iter in 1:M){  
      cat("\r",paste0(round(iter/M*100,0),'%'))
      
      # Gen truth (one world -- one truth)
      
      X_start    = dsadm_start_field_cvm(x_unpert_modpar, n, Rem)$x_start
      
      forcing = TRUE
      X_true_mdl_steps  = dsadm_generator(X_start, n, N_truth, ntime_model, dt, 
                                          UU, rrho, nnu, ssigma, Rem, forcing)  # model steps
      X_true = X_true_mdl_steps[,ind_time_anls] # truth at the anls times
      X[[iter]] = X_true  # only anls steps
      
      #save(X_true, file = paste0(path,'/DATA/truth_',iter, '.Rdata'))
      
      #-----------
      # HHBEF filtering
      
      if(perform_HHBEF == 1){
        
        # gen OBS
        OBS_NOISE = matrix(rnorm((n%/%m+min(1,n%%m))*time_model, mean=0, sd=sqrt_R),
                           nrow=n%/%m+min(1,n%%m), ncol=time_model)
        OBS       = X_true_mdl_steps[ind_obs_space,] + OBS_NOISE
        
        # Initial conditions
        
        Ba_start = matrix(0, nrow=n, ncol=n) # start flt at t=0 from the truth, as in the KF
        A_start = Ba_start # assume no obs at t=0
        
        # Compute the step-0 anls ensm members as sqrt(A)*N(0,I)
        
        sqrt_A_start = symm_pd_mx_sqrt(A_start)$sq
        N01 = matrix(rnorm(n*ne, mean=0, sd=1), nrow=n, ncol=ne)
        dXae_start = sqrt_A_start %*% N01    # perturbations yet
        
        X_flt_start = X[[iter]][,1]
        Xae_start   = X_flt_start + dXae_start # ensm members
        
        detFc_emean = 1 # deterministic fcst (1) or ensm mean (2)?
        
        HHBEF_res = HHBEF(ntime_filter, nx, dt, 
                          ind_time_anls_1, stride, ind_time_anls,  
                          ind_obs_space, Rem, 
                          UU, rrho, nnu, ssigma,  
                          F_Lorenz, J_Lorenz, sd_noise,
                          R_diag, m, OBS, 
                          X_flt_start, Ba_start, Xae_start, B_clim,
                          ne, w_cvr, w_evp10,  detFc_emean, 
                          inflation, spa_shift_mx_Bf, spa_shift_mx_S, C_lcz,
                          ntime_EE_store, store_EE_EnKF, 
                          model_type)
      
        #save(HHBEF_res, file = paste0(path,'/DATA/HHBEF_',iter, '.Rdata'))
        
        X_HHBEF[[iter]] = HHBEF_res$XXf  # only anls steps
        
      } # end if_HHBEF
    } # end loop_iter
    
  } # end if_worlds
  
  #-----------------------------------------------
  # Estm B_x_true by averaging over the worlds
  
  if(worldsAve_BBx){
    
    message('Covariance matrix of x: averaging over the worlds')
    
    X_arr <- array(NA, dim = c(M, n, time_filter))
    
    for(iter in 1:M){
      cat("\r",paste0(round(iter/M*100,0),'%'))
      X_arr[iter,,] <- X[[iter]]
    }
    
    BBx_worlds_ave = array(NA, dim = c(n, n, time_filter))
    
    for(t in 1:time_filter){
      BBx_worlds_ave[,,t] <- cov(X_arr[,,t])
    }
    
    image2D(BBx_worlds_ave[,,time_filter/2], main="CVM of x at t=time_filter/2")
    image2D(BBx_worlds_ave[,,time_filter],   main="CVM of x at t=time_filter")
    
    CVM_x_mean = apply(BBx_worlds_ave, c(1,2), mean)
    image2D(CVM_x_mean, main="Time mean spatial CVM of x")
    
    var_x_mean = mean(diag(CVM_x_mean))
    sqrt(var_x_mean)
    sd_x_true = sd(as.vector(X_true))  # one world only..
    sd_x_true  # sh.be close to sqrt(var_x_mean). OK.
    
    filter$BBx_worlds_ave = BBx_worlds_ave
  }
  
  #-----------------------------------------------
  # Estm BB_HHBEF by averaging over the worlds
  
  if(worldsAve_BB_HHBEF){
    
    message('HHBEF: background-error (sample covariance) matrix S and one-world flt stats')
    
    BB_HHBEF_worldsAve = array(0, dim = c(n, n, time_filter))
    
    for(iter in 1:M){
      #load(paste0(path,'/DATA/truth_',iter,'.Rdata'))  
      #load(paste0(path,'/DATA/HHBEF_',iter,'.Rdata'))  
      
      X_true = X[[iter]] 
      Xf_HHBEF = X_HHBEF[[iter]]
      
      for(i in 1:time_filter){
        BB_HHBEF_worldsAve[,,i] = BB_HHBEF_worldsAve[,,i] + 
          ((Xf_HHBEF[,i] - X_true[,i])) %*% t(Xf_HHBEF[,i] - X_true[,i])
      }
      cat("\r",paste0(round(iter/M*100,0),'%'))
    }
    
    BB_HHBEF_worldsAve = BB_HHBEF_worldsAve / M
    
    filter$BB_HHBEF_worldsAve = BB_HHBEF_worldsAve
  } 
  
  #--------------------------------------------
  # Write output txt file
  
  outfile=paste0("./RMSE.txt")
  
  if(SaveClim > 0) outfile=paste0("./RMSE_long_", seed, ".txt")
  
  unlink(outfile)
  sink(outfile) # , append=TRUE)
  
  cat("\n")
  
  if(perform_kf_ekf > 0){
    
    cat("TRUTH_RMSE=")
    print(TRUTH_RMSE)
    
    cat("\n")
    
    cat("KF_fRMSE: real fcst-err RMSE")
    print(KF_fRMSE)
    cat("sqrt_B_var_mean: fcst-err RMSE assumed by the filter")
    print(sqrt_B_var_mean)
    cat("-- sh.be close to each other")
    
    cat("\n")
    
    cat("KF_aRMSE: real anls-err RMSE")
    print(KF_aRMSE)
    cat("sqrt__var_mean: anls-err RMSE assumed by the filter")
    print(sqrt_A_var_mean)
    cat("-- sh.be close to each other")
    
    # cat("\n")
    
    # cat("FG VARIANCE reduction in the anls")
    # print(100*(KF_fRMSE^2 - KF_aRMSE^2) / KF_fRMSE^2)
  }
  
  cat("\n")
  
  if(perform_HHBEF > 0){
    
    # cat("TRUTH_RMSE=")
    # print(TRUTH_RMSE)
    
    cat("\n")
    
    cat("HHBEF_fRMSE=")
    print(HHBEF_fRMSE)
    
    # cat("\n")
    # 
    # cat("HHBEF_fRMSE_assumed=")
    # print(HHBEF_fRMSE_assumed)
    # 
    # cat("\n")
    # 
    # cat("HHBEF_aRMSE=")
    # print(HHBEF_aRMSE)
    
    # cat("\n")
    
    # same_parameters = all(parameters[1:nparam_not_HHBEF_specific]    %in% 
    #                         parameters_KF[1:nparam_not_HHBEF_specific]) & 
    #   all(parameters_KF[1:nparam_not_HHBEF_specific] %in% 
    #         parameters[1:nparam_not_HHBEF_specific])
    
    # if(same_parameters){
    #   cat("(HHBEF_fRMSE - KF_fRMSE) / KF_fRMSE, %:")
    #   print(100*(HHBEF_fRMSE - KF_fRMSE) / KF_fRMSE)
    # }else{
    #   print("Rerun KF for this parameter set!")
    #   print("Rerun KF for this parameter set!")
    #   print("Rerun KF for this parameter set!")
    # }
    
  }
  
  
  # cat("\n")
  
  if(perform_LSEF > 0){
    
    # cat("TRUTH_RMSE=")
    # print(TRUTH_RMSE)
    
    cat("\n")
    
    cat("LSEF_fRMSE=")
    print(LSEF_fRMSE)
    
    cat("\n")
    
    # cat("LSEF_fRMSE_assumed=")
    # print(LSEF_fRMSE_assumed)
    # 
    # cat("\n")
    # 
    # cat("LSEF_aRMSE=")
    # print(LSEF_aRMSE)
    
    cat("\n")
    
    # same_parameters = all(parameters[1:nparam_not_LSEF_specific]    %in% 
    #                                          parameters_KF[1:nparam_not_LSEF_specific]) & 
    #                   all(parameters_KF[1:nparam_not_LSEF_specific] %in% 
    #                                          parameters[1:nparam_not_LSEF_specific])
    
    # if(same_parameters){
    #   cat("(LSEF_fRMSE - KF_fRMSE) / KF_fRMSE, %:")
    #   print(100*(LSEF_fRMSE - KF_fRMSE) / KF_fRMSE)
    # }else{
    #   print("Rerun KF for this parameter set!")
    #   print("Rerun KF for this parameter set!")
    #   print("Rerun KF for this parameter set!")
    # }
    
  }
  
  sink()
  
  #-----------------------------------------------
  # write output : filter
  
  save=FALSE # FALSE TRUE
  
  if(save){
    if(mode == 0) {                            # single-run model/filters output
      save(filter, file = paste0(path,"/fields_", seed, "_mode",mode, ".RData"))
    }
    
    if(mode > 0) {                             # Fields and Covs output
      save(filter, file = paste0(path,"/fields_covs_", seed,   "_mode",mode, ".RData"))
    }
  }
  #-----------------------------------------------
  
  return(list(TRUTH_RMSE=TRUTH_RMSE, KF_fRMSE=KF_fRMSE, 
         HHBEF_fRMSE=HHBEF_fRMSE, LSEF_fRMSE=LSEF_fRMSE))
}
