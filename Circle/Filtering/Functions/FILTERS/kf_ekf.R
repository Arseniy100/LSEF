
KF_EKF = function(ntime_filter, n, dt, 
                  ind_time_anls_1, stride, ind_time_anls,  
                  ind_obs_space, Rem, 
                  UU, rrho, nnu, ssigma,  
                  F_Lorenz, J_Lorenz, sd_noise,
                  R_diag, m, OBS, 
                  X_flt_start, A_start, 
                  model_type, filter_type, ntime_B_store, 
                  store_BB_KF_EnKF){
  #-----------------------------------------------------------------------------------
  # KF/EKF
  # 
  # Note that the "anls-fcst" cycle starts here from an timestep-0 field X_flt_start
  # and one cycle is {(i) fcst, (ii) anls}.
  # 
  # Args:
  # 
  # ntime_filter - nu of ANLS (filter) time steps 
  #                (one filter time step is
  #               a multiple (stride) of the model time step)
  # n - dim-ty of the state vector x
  # dt - MODEL time step (atmospheric time), sec.
  # ind_time_anls_1 -  1st mdl time step when anls is performed
  # stride - nu of model time steps between consecutive analyses
  # ind_obs_space - vector of indices of the state vector, where (in space) OBS are present
  # ind_time_anls - model time steps at which the anls is to be performed 
  #                (= ind_time_anls_1, ind_time_anls_1 + stride, ind_time_anls_1 + 2*stride, ...)
  # Rem - Earth radius, m
  # UU, rrho, nnu,  ssigma - scnd flds for DSADM
  # F_Lorenz, J_Lorenz, sd_noise - Lorenz-2005 params
  # 
  # R_diag - diagonal (a vector) of the obs-err CVM
  # m - distance between adjacent obs in space (in grid meshes, integer)
  # OBS - obs at ALL MODEL time steps at the obs locations defined by ind_obs_space
  # X_flt_start - the fcst valid at time step 1 starts from X_flt_start 
  # A_start - the imposed "anls-err CVM" at the virtual time step 0
  # model_type = "DSADM" or "Lorenz05" or "Lorenz05lin"
  # filter_type - "KF" or "EKF" 
  #     NB: with model_type="Lorenz05", only filter_type ="EKF" is acceptable
  # ntime_B_store - number of asml time steps uniformly selected 
  #    between ntime_filter/4 and ntime_filter, at which the prior CVM B is to be stored & returned
  # store_BB_KF_EnKF - store & return prior filtering cvms? :
  #   =-1 don't
  #   =0 only ntime_B_store time instants
  #   =1 full CVMs are stored at each filter time step
  # 
  # return: XXf & XXa (valid at anls times), BB_KF, and 
  #         B_mean, A_mean.
  #         vars (variances at all grid points all times),
  #         spat_ave_cvfs (statio spat ave cvfs)
  # 
  # A Rakitko, 
  # M Tsyrulnikov (current code owner)
  # June 2018, May 2021, March 2023
  #-----------------------------------------------------------------------------------
  
  h = 2*pi*Rem/n
  nx = n
  
  ntime_model = ntime_filter * stride
  
  XXf    = matrix(NA, nrow = n, ncol = ntime_model)
  XXa    = matrix(NA, nrow = n, ncol = ntime_model)
  B_mean = matrix(0,  nrow = n, ncol = n)
  # C_mean = matrix(0,  nrow = n, ncol = n)
  A_mean = matrix(0,  nrow = n, ncol = n)
  
  if(store_BB_KF_EnKF == 1) {
    BB  = array(NA, c(n, n, ntime_filter))
  }else if(store_BB_KF_EnKF == -1){                                   
    BB  = NULL
  }else if(store_BB_KF_EnKF == 0){
    BB  = array(NA, c(n, n, ntime_B_store))
    ind_B_store = seq(from=ceiling(ntime_filter/4), to=ntime_filter, length.out=ntime_B_store)
    ind_B_store = floor(ind_B_store)
  }
  vars          = matrix(0, nrow = nx, ncol = ntime_filter)
  spat_ave_cvfs = matrix(0, nrow = nx, ncol = ntime_filter)
  
  #AA         <- array(NA,c(n, n, ntime_model))
  
  n_obs=length(ind_obs_space)        # number of obs
  H = matrix(0, nrow=n_obs, ncol=n)  # obs oprt
  for (i in 1:n_obs){
    H[i, ind_obs_space[i]] = 1
  }
  
  R=diag(R_diag)                     # obs-err CVM
  
  Xa = X_flt_start # the 1st fcst starts at time step 0 from this field
  A  = A_start

  eps=1e-9 # EKF: Jacobian assessment through finite differences:
           #      scale down anls-CVM columns to reach the linear regime 1e-7..1e-9 are ok
 
  #---------------------------------------------------
  # Checks
  
  if(model_type != "DSADM" & model_type != "Lorenz05" & model_type != "Lorenz05lin"){
    print(model_type)  
    stop("KF_EKF: wrong model_type")
  }
  
  if(filter_type != "KF" & filter_type != "EKF"){
    print(filter_type)  
    stop("KF_EKF: wrong filter_type")
  }
  
  if(model_type == "Lorenz05" & filter_type == "KF"){
    print(model_type)
    print(filter_type)  
    stop("KF_EKF: wrong filter_type/model_type pair")
  }
  
  if(model_type == "Lorenz05lin" & filter_type == "EKF"){
    print(model_type)
    print(filter_type)  
    stop("KF_EKF: wrong filter_type/model_type pair")
  }
  
  #---------------------------------------------------
  # Lorenz: From atmospheric time to Lorenz time
  
  if(model_type == "Lorenz05" | model_type == "Lorenz05lin") {
    dt_atm_h = dt /3600
    dt_Lorenz = dt_atm_h/6*0.05    # unitless, "Lorenz time" 
                                   # (6h atmospheric time ~ 0.05 Lorenz time units)
    Q_Lorenz=sd_noise^2 * diag(n)  # Lorenz's Q
  }
  
  Q_DSADM_diag = ssigma^2  /h *dt # model-error variances per model time step
  
  #---------------------------------------------------
  # main loop over MODEL time steps
  
  ntm10 =floor(ntime_model /10)
  ntm100= max(1, floor(ntime_model /100))
  
  forcing = FALSE
  i_filter=0
  i_store_B_partly = 0
  
  for(i in 1:ntime_model){    
    
    if(i %% ntm10 == 0){
      message(i / ntm100)
    }
    
    # (1) Fcst
    # (1.1) run deterministic fcst started from the previous anls
    
    if(model_type == "DSADM"){ 
      N_det=1
      XXf[,i] = dsadm_step(Xa, n, N_det, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing)

    }else if(model_type == "Lorenz05"){
      XXf[,i] = lorenz05_step(Xa, n, dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n))
      
    }else if(model_type == "Lorenz05lin"){
      XXf[,i] = lorenz05lin_step(Xa, X_ref, n, dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n))
    }
    
    # (1.2) fcst covs
    
    if(model_type == "DSADM"){
      # In the implicit DSADM time integration scheme, 
      # model error is added Before the mdl oprt is applied ==>
      
      CVM_forcing=diag( Q_DSADM_diag[,i] ) 
      AQ = A + CVM_forcing   # A is the previous-cycle anls-err CVM
                             # NB: CVM_forcing is not exactly Q 
      
      # B = F * AQ * F^T 
      # (1) PHI := F * AQ
      # (2) B = PHI * F^T = (F * PHI^T)^T = F * PHI^T
      
      if(filter_type == "KF"){
        
        #PHI = apply( AQ,    2, function(x) dsadm_step(x, n, 1, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], 
        #                                              Rem, forcing) )
        #B   = apply( t(PHI),2, function(x) dsadm_step(x, n, 1, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], 
        #                                              Rem, forcing) ) 
        
        PHI = dsadm_step(AQ,     n, n, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing)
        B   = dsadm_step(t(PHI), n, n, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], Rem, forcing)
      
      }else if(filter_type == "EKF"){  # for testing only
        PHI = apply( AQ,     2, 
                     function(x) ApplyJacobian_fd( dsadm_step, Xa, XXf[,i], x, eps,
                                                   n, n, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], 
                                                   Rem, forcing) )
        B   = apply( t(PHI), 2, 
                     function(x) ApplyJacobian_fd( dsadm_step, Xa, XXf[,i], x, eps,
                                                   n, n, dt, UU[,i], rrho[,i], nnu[,i], ssigma[,i], 
                                                   Rem, forcing) )
      }
      

            
    }else if(model_type == "Lorenz05"){
      
      # In the Lorenz model, system noise is added after the fcst ==>
      # # B = F * A * F^T  + Q
      # (1) PHI := F * AQ
      # (2) B = PHI * F^T = (F * PHI^T)^T = F * PHI^T
      
      PHI = apply( A,  2, 
                   function(x) ApplyJacobian_fd(lorenz05_step, Xa, XXf[,i], x, eps,
                                                n, dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n)) )
      P   = apply( t(PHI), 2, 
                   function(x) ApplyJacobian_fd(lorenz05_step, Xa, XXf[,i], x, eps,
                                                n, dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n)) )
      P = (P + t(P)) /2 # eliminate computational non-symmetry
      B = P + Q_Lorenz
      
      
      
    }else if(model_type == "Lorenz05lin"){
      
      if(filter_type == "KF"){
        
        PHI = apply(AQ,    2, function(x) lorenz05lin_step(x, Xref, n, 
                                          dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n)) )
        P   = apply(t(PHI),2, function(x) lorenz05lin_step(x, Xref, n,  
                                          dt_Lorenz, F_Lorenz, J_Lorenz, rep(0,n)) )
        B = P + Q_Lorenz
      }
    }
    
    
    # (2) Anls
    
    # Separate model time steps when the anls is to be or not to be performed.
    # ANLS are to be done at    t=stride*k +ind_time_anls_1,    where k=1,2,3,...
    # Therefore at the anls times, t-ind_time_anls_1 should divisible by stride:
    
    if(((i-ind_time_anls_1) %% stride) != 0){  # no anls, continue fcst
      Xa = XXf[,i]
      A = B

    }else{                       # perform anls
      i_filter=i_filter +1
      
      BHT  = B[             , ind_obs_space]  # B*H^T
      HBHT = B[ind_obs_space, ind_obs_space]  # H*B*H^T
      HBHTpR = HBHT + R
      K = BHT %*% solve(HBHTpR)
      
      Xa = XXf[,i] + K %*% (OBS[,i] - XXf[ind_obs_space,i])
    
      A = B - K%*%B[ind_obs_space,] # (I-KH)B
      
      # use  the Joseph form: 
      ## A=(I-KH)B(I-KH)^T + KRK^T 
      ## --> Yields the same results.
      
      #ImKH=diag(n) - K %*% H
      #A=ImKH %*% B %*% t(ImKH) + K %*% R %*% t(K)
    
      # Averaging of B, A
      
      B_mean = B_mean + B # cvm
      
      # C = Cov2VarCor(B)$C
      # C_mean = C_mean + C # crm
      
      A_mean = A_mean + A # cvm
      
      # Store  B  and its spat ave form (cvf_statio)
      
      vars[, i_filter] = diag(B)
      
      spat_ave_cvm = SpatAveCVM_S1(B)
      spat_ave_cvfs[, i_filter] = spat_ave_cvm[1,] # store 1st row (cvf)
      
      if(store_BB_KF_EnKF == 1) {
        BB[,,i_filter] = B 
      }else if(store_BB_KF_EnKF == 0){
        
        if(is.element(i_filter, ind_B_store)){
          i_store_B_partly = i_store_B_partly +1
          BB[,,i_store_B_partly] = B
        }
      }
    }
    
    XXa[,i] = Xa
    #AA[,,i] = A
    
  }  # end time loop
  
  B_mean = B_mean / ntime_filter
  # C_mean = C_mean / ntime_filter
  A_mean = A_mean / ntime_filter
  
  return(list(XXa   = XXa  [, ind_time_anls], 
              XXf   = XXf  [, ind_time_anls], 
              B_mean = B_mean,
              # C_mean = C_mean, # FG err CVM
              A_mean = A_mean,
              BB = BB,    # [1:nx, 1:nx, 1:ntime_B_store]
              vars=vars,
              spat_ave_cvfs = spat_ave_cvfs # [1:nx, 1:ntime_filter]
              #AA = AA[,,ind_time_anls]
              ))
}
