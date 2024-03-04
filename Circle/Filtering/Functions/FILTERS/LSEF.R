
LSEF <- function(ntime_filter, nx, dt, 
                  ind_time_anls_1, stride, ind_time_anls,  
                  ind_obs_space, Rem,  
                  UU, rrho, nnu, ssigma,  
                  F_Lorenz, J_Lorenz, sd_noise,
                  R_diag, m, OBS, 
                  B2S_method, NN, 
                  X_flt_start, Xae_start, B_clim, w_evc10,   
                  ne, tranfu, b_shape, inflation_LSEF, C_lcz,
                  true_field_available, X_true_mdl_steps,
                  model_type){
  #-----------------------------------------------------------------------------------
  # LSEF: 
  # 
  # Note that the "anls-fcst" cycle starts here from an timestep-0 field X_flt_start
  # and one cycle is {(i) fcst, (ii) anls}.
  # 
  # NB: We assume that the model-error and obs-err statistics are perfect,
  #     therefore, the deterministic (control) forecast is preferred here 
  #     over the ensemble mean -- both in the fcst and anls.
  #  -- check this!!
  #--------------------------------------------------
  # Args:
  # 
  # ntime_filter - nu of ANLS (filter) time steps 
  #                (onefilter time step is
  #               a multiple (stride) of the model time step)
  # dt - MODEL time step (atmospheric time), sec.
  # ind_time_anls_1 -  1st mdl time step when anls is performed
  # stride - nu of model time steps between consecutive analyses
  # ind_obs_space - vector of indices of the state vector, where (in space) OBS are present
  # ind_time_anls - model time steps at which the anls is to be performed 
  #                (= ind_time_anls_1, ind_time_anls_1 + stride, ind_time_anls_1 + 2*stride, ...)
  # 
  # UU, rrho, nnu,  ssigma - scnd flds for DSADM
  # F_Lorenz, J_Lorenz, sd_noise - Lorenz-2005 params
  # 
  # R_diag - diagonal (a vector) of the obs-err CVM
  # m - distance between adjacent obs in space (in grid meshes, integer)
  # OBS - obs at ALL MODEL time steps at the obs locations defined by ind_obs_space
  # B2S_method - (Bands to Spectrum) how to compute local spectrum from ensm band variances: SVshape or NN (default)
  # NN- torch neural network trained on synthetic data
  # B_clim - static B
  # X_flt_start - the fcst valid at time step 1 starts from  X_flt_start (ie at  the virtual time step 0)
  # Xae_start - the "anls" ensemble at the virtual time step 0 
  #           (from which the 1st ensm fcst starts)
  # B_clim - static B
  # w_evc10 - relative weight of B_LSM vs B_clim
  # ne - ensm size
  # tranfu - transfer functions of the bandpass filters [wvn=1:nx, j=1:J],  J=nband
  # inflation_LSEF - covariance inflation_LSEF coeficient 
  #       (defined as the multiplier of the fcts-ensm perturbations, i.e.
  #       the covariances are effectively multiplied by inflation_LSEF^2): =1 by default
  # C_lcz - localization mx (for plotting only)
  # true_field_available - is truth at model time steps provided? (for verif)
  # X_true_mdl_steps - truth at model time steps (for verif)
  # model_type = "DSADM" or "Lorenz05" or "Lorenz05lin"
  # 
  # return: arrays at the anls times only.
  # 
  # M Tsyrulnikov (current code owner),
  # A Rakitko
  # 
  # May 2021, 2023
  #-----------------------------------------------------------------------------------
  
  ntime_model = ntime_filter *stride   # nu of mdl time steps
  nmax = nx/2
  nband = dim(tranfu)[2] ;  J = nband
  true_spectra_available = FALSE
  
  # Checks (only DSADM model permitted at the moment)
  
  if(nx != dim(tranfu)[1]){
    print(str(tranfu))  
    stop("LSEF: wrong nx in tranfu")
  }
  
  if(model_type != "DSADM"){ # & model_type != "Lorenz05" & model_type != "Lorenz05lin"){
    print(model_type)  
    stop("LSEF: wrong model_type")
  }
  
  #========================================================
  # Preliminaries
  
  h = 2*pi*Rem/nx                    # spatial mesh size
  
  XXf = matrix(NA, nrow = nx, ncol = ntime_model)
  XXa = matrix(NA, nrow = nx, ncol = ntime_model)
  S_mean = matrix(0,  nrow = nx, ncol = nx)
  B_mean = matrix(0,  nrow = nx, ncol = nx)
  
  # Obs related variables
  
  n_obs=length(ind_obs_space)        # number of obs
  H = matrix(0, nrow=n_obs, ncol=nx)  # obs oprt
  for (i in 1:n_obs){
    H[i, ind_obs_space[i]] = 1
  }
  
  R=diag(R_diag)                     # obs-err CVM
  sqrt_R_diag=sqrt(R_diag)           # obs-err st.devs
  
  # Starting (step-0) conditions
  
  Xa  = X_flt_start # the 1st deterministic fcst starts at time step 0 from this field
  Xae = Xae_start   # step-0 anls ensm
  
  w_evc = ne / (10/w_evc10 - 10 + ne) # ensm-vs-clim weight in computing Ba
  
  #========================================================
  
  lplot=F
  iplot_stride = 100
  
  tranfu2 = (abs(tranfu))^2 # [i_n, j=1:J]
  band_centers_n = NULL
  
  # Omega and its SVD
  
  lplot_=F
  Omega_SV = Omega_SVD(tranfu2, lplot_)
  
  Omega_nmax = Omega_SV$Omega_nmax
  Omega_S1   = Omega_SV$Omega_S1
  SVD_Omega_nmax = Omega_SV$SVD_Omega_nmax
  SVD_Omega_S1   = Omega_SV$SVD_Omega_S1
  
  #========================================================
  # Prms for  B2S
  
  if(B2S_method == "NN"){
    TransformSpectrum_type_x = "sqrt" # "none" "log" "sqrt" "pow"  "RELU" -- none or sqrt 
    TransformSpectrum_type_y = "sqrt" # "none" "log" "sqrt" "pow" "RELU" -- sqrt good
    TransformSpectrum_pow_x = 1 # not used
    TransformSpectrum_pow_y = 1 # not used
    
  }else if(B2S_method == "SVshape"){
    moments = "012" # "01" , "12" , "012"
    a_max_times = 5 # max deviation of the scale multiplier  a  from 1 in times 
    w_a_fg = 0.05  #  weight of the ||a-1||^2 weak constraint in fitScaleMagn
    nSV_discard = 0 # discard trailing SV to filter sampling noise
  }
  
  BOOTENS = F
  band_Ve_B = NULL
  nB = 0

  #========================================================
  # The main loop over MODEL time steps
  
  ntm10  = floor(ntime_model /10)
  ntm100 = max(1, floor(ntime_model /100))
  
  forcing_fcst = FALSE
  forcing_ensm = TRUE
  itime_filter = 0
  
  for(i in (1:ntime_model)){
    
    if(i %% ntm10 == 0){
      message(i / ntm100)
    }
    
    #-------------------------
    # (1) Fcst (valid at time step i)
    # (1.1) run deterministic fcst started from the previous anls
    
    N_det=1
    Xf = dsadm_step(Xa, nx, N_det, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing_fcst)
     
    # (1.2) Fcst ensm
    
    Xfe = dsadm_step(Xae, nx, ne, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing_ensm)
  
    #=========
    # plot(X_true_mdl_steps[,i], type="l")
    # lines(Xf, col="red")
    # lines(Xfe[,1], col="green", lty=2)
    # lines(Xfe[,5], col="blue", lty=2)
    
    
    # Separate model time steps i when the anls is to be or not to be performed:
    # ANLS are to be done at    i=stride*k +ind_time_anls_1,    where k=0,1,2,...
    # Therefore at the anls times, i-ind_time_anls_1 should divisible by stride:
    
    if(((i-ind_time_anls_1) %% stride) != 0){  # no anls, continue fcst
      
      Xa  = Xf
      Xae = Xfe
      
      
    }else{                       # time step when anls is performed
      
      itime_filter = itime_filter +1
      
      #-------------------------
      # (2) Anls
      
      # inflation_LSEF
      
      dXfe=(Xfe - matrix(rep(Xf, ne), nrow = nx)) * inflation_LSEF  # inflated fcst-ensm perturbations
      Xfe_inflated = matrix(rep(Xf, ne), nrow = nx) + dXfe
      
      S=tcrossprod(dXfe, dXfe) /ne
      S_mean = S_mean + S           
      
      # E2B: Band variances
      
      bandData = E2B(dXfe, tranfu, BOOTENS, nB, true_spectra_available)
      band_Ve = bandData$band_Ve
      # image2D(band_Ve, main="band_Ve")
      # image2D(dXfe, main="dXfe")
      
      # B2S: local spectra
      
      # Normalize band vars so that NN be in the regime it has been trained in
      # (in which Var(xi_FG) ~ 1)
      
      # evars = apply(dXfe^2, 1, mean) # ensm variances
      # NRM = median(evars)
      # NRM = quantile(evars, probs=0.2)
      # NRM = max(evars)
      NRM=1
      band_Ve_nrm = band_Ve / NRM
      # image2D(band_Ve_nrm, main="band_Ve_nrm")
      
      lplot_B2S = F
      
      if(B2S_method == "SVshape"){
        SPECTRUM_SVshape = B2S_SVshape(band_Ve_nrm, Omega_S1, SVD_Omega_S1,
                                       b_shape, moments,  band_centers_n,
                                       BOOTENS, band_Ve_B,
                                       nSV_discard, a_max_times, w_a_fg, 
                                       true_spectra_available, lplot_B2S)
        b_LSM = SPECTRUM_SVshape$b_fit
        
      }else if(B2S_method == "NN"){
        
        SPECTRUM_NN = B2S_NN(band_Ve_nrm, NN, 
                             TransformSpectrum_type_x, TransformSpectrum_type_y,
                             TransformSpectrum_pow_x, TransformSpectrum_pow_y,
                             Omega_nmax, true_spectra_available, lplot_B2S)
        b_LSM = SPECTRUM_NN$b_fit
      }
      
      # Renormalize the resulting spectrum, b_LSM[ix,i_n]
      b_LSM = b_LSM * NRM   
      
      # B_LSM, W_LSM
      
      Sigma_LSM = sqrt(b_LSM)
      WB = Sigma2WB(Sigma_LSM)
      B_LSM = WB$B
      # W_LSM = WB$W
      Ba = (1-w_evc)*B_clim + w_evc*B_LSM
      
      if(lplot & itime_filter %% iplot_stride == 0){
        
        S = tcrossprod(dXfe, dXfe) / ne
        # S=dXfe %*% t(dXfe) /ne  
        S_lcz = S * C_lcz
        
        d=nx/6
        ix=sample(c((d+1):(nx-d)),1)
        # ix=(ix+2) %% nx
        mx=max(S_lcz[ix, (ix-d):(ix + d)], B_LSM[ix, (ix-d):(ix + d)])
        mn=min(S_lcz[ix, (ix-d):(ix + d)], B_LSM[ix, (ix-d):(ix + d)])
        if(mn > 0) mn=0
        plot(B_LSM[ix, (ix-d):(ix + d)], type="l", lwd=2, main="Row: B_LSM(red), S_lcz(blu)", 
             sub=paste0("itime_filter=", itime_filter, "  ix=",ix), 
             col="red", xlab="Distance, meshes", ylim=c(mn,mx))
        lines(S_lcz[ix, (ix-d):(ix + d)], col="blue")
      }
      
      # (2.3) KF anls with Ba
      
      # Kalman gain
      BHT  = Ba[             , ind_obs_space]  # B*H^T
      HBHT = Ba[ind_obs_space, ind_obs_space]  # H*B*H^T
      HBHTpR = HBHT + R
      K = BHT %*% solve(HBHTpR)
      
      # Deterministic anls
      Xa = Xf + K %*% (OBS[,i] - Xf[ind_obs_space])

      
      # rmse(Xf, X_true_mdl_steps[,i])
      # rmse(Xa, X_true_mdl_steps[,i])
      
      
      # Ensm anls
      # Generate simulated obs errs
      
      obsN01 = rnorm(n_obs*ne, mean=0, sd=1) # N(0,1) noise
      simOBS_err = matrix(obsN01*sqrt_R_diag, nrow=n_obs, ncol=ne)
      
      # Generate perturbed obs
      
      simOBS = OBS[,i] + simOBS_err
      
       
      # Anls ensm
      
      Xae = Xfe_inflated + K %*% (simOBS - Xfe_inflated[ind_obs_space,])
      
      # Averaging of Ba
      B_mean = B_mean + B_LSM
      
    }                # end anls time step
  
    #-------------------------
    # Store arrays
    
    XXf[,i] = Xf
    XXa[,i] = Xa
    
  }  # end time loop
  
  #========================================================
  
  B_mean = B_mean / ntime_filter
  S_mean = S_mean / ntime_filter
  
  
  return(list(XXf = XXf[, ind_time_anls], 
              XXa = XXa[, ind_time_anls],
              B_mean = B_mean, 
              S_mean = S_mean
  ))

}
