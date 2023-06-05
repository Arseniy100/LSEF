
HHBEF <- function(ntime_filter, n, dt, 
                  ind_time_anls_1, stride, ind_time_anls,  
                  ind_obs_space, Rem,  
                  UU, rrho, nnu, ssigma,  
                  F_Lorenz, J_Lorenz, sd_noise,
                  R_diag, m, OBS, 
                  X_flt_start, Ba_start, Xae_start, B_clim,
                  ne, w_cvr, w_evp10, detFc_emean,
                  inflation, spa_shift_mx_Bf, spa_shift_mx_S, C_lcz, 
                  ntime_B_store, store_BB_KF_EnKF, 
                  model_type){
  #-----------------------------------------------------------------------------------
  # HHBEF: a Generalized HBEF, which includes EnKF, Var, EnVar, HBEF as special cases
  # and blends sample cvm with any or all of
  # (i) B_clim
  # (ii) time-smoothed covs
  # (iii) space-smoothed covs
  # As compared to the HBEF, in the HHBEF:
  # 
  # 1) The scnd filter treats B rather than P and Q separately,
  # 2) The fcst step of the scnd filter involves a "regression to the mean"
  #    ("climatological") background-error covariance matrix:
  #-------------------------------------------------------------------    
  #  B_f(t) = w_cvr*B_clim + (1-w_cvr)*B_a(t-1)               (1)
  #-------------------------------------------------------------------  
  #  whr B_clim is the static time-mean climatological CVM,
  #      B_a(t-1) is the final estimate of B at the previous time step, and
  #      w_cvr weighs the static CVM B_clim vs the recent-past CVM B_a(t-1).
  #  
  #  The anls step of the scnd flt is 
  #-------------------------------------------------------------------  
  # B_a(t) = (1-w_evp)*B_f(t) + w_evp*S(t)                   (2)
  #-------------------------------------------------------------------  
  # whr 
  # w_evp = ne /(theta + ne), find from 
  # w_evp10 = 10 / (theta +10)  ==>
  # theta = 10/w_evp10 - 10  ==>
  #---------------------------------- 
  # w_evp = ne /(10/w_evp10 - 10 + ne)
  #----------------------------------
  # w_pve=1-w_evp 
  #  whr ne is the ensm size.
  # 
  # Note that the "anls-fcst" cycle starts here from an timestep-0 field X_flt_start
  # and one cycle is {(i) fcst, (ii) anls}.
  # 
  # NB: We assume that the model-error and obs-err statistics are perfect,
  #     therefore, the deterministic (control) forecast is preferred here 
  #     over the ensemble mean -- both in the fcst and anls.
  # 
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
  # X_flt_start - the fcst valid at time step 1 starts from  X_flt_start (ie at  the virtual time step 0)
  # Ba_start - "anls"  B  (in terms of the scnd flt) at  the virtual time step 0
  # Xae_start - the "anls" ensemble at the virtual time step 0 
  #           (from which the 1st ensm fcst starts)
  # B_clim - static B
  # ne - ensm size
  # w_cvr - in computing the prior Bf(t): the relative weight of the static "Climatological" CVM 
  #         vs the evolved Recent past CVM Ba(t-1), see Eq(1) above
  # w_evp10 - relative weight of the (localized) Ensemble sample CVM S vs the Prior Bf ("ensm-vs-prior"),
  #           see Eq(2) above 
  # detFc_emean - switch: =1: use deterministic fcst instead of ensm mean
  #                       =2: use ensm mean instead of deterministic fcst
  # inflation - covariance inflation coeficient 
  #       (defined as the multiplier of the fcts-ensm perturbations, i.e.
  #       the covariances are effectively multiplied by inflation^2)
  # spa_shift_mx_Bf, spa_shift_mx_S - number of spatial shifts to be made in one of the 
  #          two directions in order to spatially smooth the cvm (Bf and S, resp.)
  #          with an even triangular weighting function
  # C_lcz - localization matrix 
  # ntime_B_store - number of asml time steps uniformly selected 
  #    between ntime_filter/4 and ntime_filter, at which the prior CVM B is to be stored & returned
  # store_BB_KF_EnKF - compute & return prior filtering cvms (S_lcz)? :
  #   =-1 don't
  #   =0 only ntime_B_store time instants
  #   =1 full CVMs are stored at each filter time step
  # model_type = "DSADM" or "Lorenz05" or "Lorenz05lin"
  # 
  # return: arrays XXf, XXa  at anls times only,
  #   B_mean, SS_lcz, S_mean (non-lcz), spat_ave_cvfs
  # 
  # M Tsyrulnikov (current code owner),
  # A Rakitko
  # 
  # Mar 2019
  #-----------------------------------------------------------------------------------
  
  ntime_model = ntime_filter *stride   # nu of mdl time steps
  
  # Checks (only DSADM model permitted at the moment)
  
  if(model_type != "DSADM"){ # & model_type != "Lorenz05" & model_type != "Lorenz05lin"){
    print(model_type)  
    stop("HHBEF: wrong model_type")
  }
  
  #---------------------------------------------------
  # Preliminaries
  
  h = 2*pi*Rem/n                    # spatial mesh size
  nx = n
  
  XXf = matrix(NA, nrow = n, ncol = ntime_model)
  XXa = matrix(NA, nrow = n, ncol = ntime_model)
  B_mean = matrix(0,  nrow = n, ncol = n)
  S_mean = matrix(0,  nrow = n, ncol = n)
  #BBf = array(NA,dim=c(n, n, ntime_model))
  #BBa = array(NA,dim=c(n, n, ntime_model))

  if(store_BB_KF_EnKF == 1) {
    SS_lcz  = array(NA, c(n, n, ntime_filter))
  }else if(store_BB_KF_EnKF == -1){                                   
    SS_lcz  = NULL
  }else if(store_BB_KF_EnKF == 0){
    SS_lcz  = array(NA, c(n, n, ntime_B_store))
    ind_B_store = seq(from=ceiling(ntime_filter/4), to=ntime_filter, length.out=ntime_B_store)
    ind_B_store = floor(ind_B_store)
  }
  
  # Obs related variables
  
  n_obs=length(ind_obs_space)        # number of obs
  H = matrix(0, nrow=n_obs, ncol=n)  # obs oprt
  for (i in 1:n_obs){
    H[i, ind_obs_space[i]] = 1
  }
  
  R=diag(R_diag)                     # obs-err CVM
  sqrt_R_diag=sqrt(R_diag)           # obs-err st.devs
  
  # Starting (step-0) conditions
  
  Xa  = X_flt_start # the 1st deterministic fcst starts at time step 0 from this field
  Xae = Xae_start   # step-0 anls ensm
  Ba  = Ba_start    # step-0 anls estimate of B(t=0)
  
  w_evp = ne / (10/w_evp10 - 10 + ne) # ensm-vs-prior weight in computing Ba
  
  # save spat averaged covs
  spat_ave_cvfs = matrix(0, nrow = nx, ncol = ntime_filter)
  
  #========================================================
  # The main loop over MODEL time steps
  
  ntm10 =floor(ntime_model /10)
  ntm100= max(1, floor(ntime_model /100))
  
  forcing_fcst = FALSE
  forcing_ensm = TRUE
  i_filter = 0
  i_store_B_partly = 0
  
  for(i in (1:ntime_model)){
    
    if(i %% ntm10 == 0){
      message(i / ntm100)
    }
    
    #-------------------------
    # (1) Fcst (valid at time step i)
    #  Fcst ensm
    
    Xfe = dsadm_step(Xae, n, ne, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing_ensm)
    
    if(detFc_emean == 1){
      # run deterministic fcst started from the previous anls
      N_det=1
      Xf = dsadm_step(Xa, n, N_det, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing_fcst)
    
    }else if(detFc_emean == 2){
      
      # take ensm mean
      Xf = apply(Xfe, 1, mean)
      
    }else{
      print(detFc_emean)  
      stop("HHBEF: wrong detFc_emean")
    }
    
    # Separate model time steps i when the anls is to be or not to be performed:
    # ANLS are to be done at    i=stride*k +ind_time_anls_1,    where k=0,1,2,...
    # Therefore at the anls times, i-ind_time_anls_1 should divisible by stride:
    
    if(((i-ind_time_anls_1) %% stride) != 0){  # no anls, continue fcst
      
      Xa  = Xf
      Xae = Xfe
      
      
    }else{                       # time step when anls is performed
      
      i_filter = i_filter +1
      
      #-------------------------
      # (1.3) Scnd flt B: fcst, compute the Prior B, i.e. Bf
      
      Bf = w_cvr*B_clim + (1-w_cvr)*Ba
      
      # Spa smoo Bf
      
      if(spa_shift_mx_Bf > 0){
        C=Bf
        spa_shift_max = spa_shift_mx_Bf
        Bf = SpaSmooCVM(n, C, spa_shift_max)
      }
      
      #-------------------------
      # (2) Anls
      
      # (2.0.0) Inflation
      
      dXfe=(Xfe - rep(Xf, ne)) * inflation  # inflated fcst-ensm perturbations
      Xfe_inflated = rep(Xf, ne) + dXfe
      
      # (2.0.1) Sample CVM
      
      if(detFc_emean == 1){     # deterministic fcst used (ie the expectation is known)
        S=tcrossprod(dXfe, dXfe) /ne  # non-shifted sample CVM
        
      }else{                    # ensm mean used
        S=tcrossprod(dXfe, dXfe) /(ne-1)  # non-shifted sample CVM
      }
      
      # Calc and store spat_ave_cvfs
      spat_ave_cvm = SpatAveCVM_S1(S)
      spat_ave_cvfs[, i_filter] = spat_ave_cvm[1,] # store 1st row (cvf)
      
      # (2.0.2) Spa smoo S?
      
      if(spa_shift_mx_S > 0){
        C=S
        spa_shift_max = spa_shift_mx_S
        S = SpaSmooCVM(n, C, spa_shift_max)
      }
      
      # (2.0.3) Covariance Localization

      S_lcz = S * C_lcz       
      S_mean = S_mean + S           
      
      if(store_BB_KF_EnKF == 1) {
        SS_lcz[,,i_filter] = S_lcz 
      }else if(store_BB_KF_EnKF == 0){
        
        if(is.element(i_filter, ind_B_store)){
          i_store_B_partly = i_store_B_partly +1
          SS_lcz[,,i_store_B_partly] = S_lcz
        }
      }
      
      # (2.1) Scnd flt: B: anls, compute the Posterior B, i.e. Ba
      
      Ba = (1-w_evp)*Bf + w_evp*S_lcz
      
      # (2.3) KF anls with Ba
      
      # Kalman gain
      BHT  = Ba[             , ind_obs_space]  # B*H^T
      HBHT = Ba[ind_obs_space, ind_obs_space]  # H*B*H^T
      HBHTpR = HBHT + R
      K = BHT %*% solve(HBHTpR)
      
      # Deterministic anls
      Xa = Xf + K %*% (OBS[,i] - Xf[ind_obs_space])

      # (2.4) Ensm anls
      # (2.4.1) Simulate obs errs
      
      obsN01 = rnorm(n_obs*ne, mean=0, sd=1) # N(0,1) noise
      simOBS_err = matrix(obsN01*sqrt_R_diag, nrow=n_obs, ncol=ne)
      
      # (2.4.2) Generate perturbed obs
      
      simOBS = OBS[,i] + simOBS_err
      
      # (2.4.3) Anls ensm
      
      Xae = Xfe_inflated + K %*% (simOBS - Xfe_inflated[ind_obs_space,])
      
      # Averaging of Ba
      B_mean = B_mean + Ba
      
    }                # end anls time step
  
    #-------------------------
    # Store arrays
    
    XXf[,i] = Xf
    XXa[,i] = Xa
    #BBf[,,i] = Bf
    #BBa[,,i] = Ba

  }  # end time loop
  
  #========================================================
  
  B_mean = B_mean / ntime_filter
  S_mean = S_mean / ntime_filter
  
  return(list(XXf = XXf[, ind_time_anls], 
              XXa = XXa[, ind_time_anls],
              B_mean = B_mean,
              SS_lcz  = SS_lcz,
              S_mean = S_mean, # non-lcz
              spat_ave_cvfs = spat_ave_cvfs # spat ave cvf (pos-def fu), all i_filter
              #BBa = BBa[,,ind_time_anls],
              #BBf = BBf[,,ind_time_anls]
  ))

}
