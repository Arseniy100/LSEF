
fit_LSM = function(B, ne, tranfu, BOOTENS, nB, true_spectra_available,
                  perform_B2S, B2S_method, b_shape, lplot, 
                  B2S_bbc, niter_bbc, correc_bbc,  # prms for B2S_lines
                  moments, a_max_times, w_a_fg,    # prms for B2S_shape
                  Omega_S1, SVD_Omega_S1, band_centers_n, nSV_discard, # prms for B2S_SVshape
                  NN, TransformSpectrum_type_x, TransformSpectrum_type_y,
                  TransformSpectrum_pow_x, TransformSpectrum_pow_y, Omega_nmax, # prms for B2S_NN
                  a_search_times, na, eps_V_sd, band_Ve_TD, eps_smoo, w_a_rglrz){ #prms for B2S_PARAM
  #--------------------------------------------------------------
  # Given cvm  B,  fit LSM:
  # 1) Generate the ENSM
  # 2) E2B (bandpass flt)
  # 3) B2S (estm local spectra b_LSM)
  # 
  #   Args:
  # 
  # B - cvm
  # tranfu[i_n=1:nx, j=1:J] - bands' spectral transfer functions
  #
  #
  # BOOTENS - TRUE if the bootstrap band variances  band_Ve_B  are available
  # nB - size of the bootstrap sample to be generated
  # true_spectra_available - TRUE spectra and true band variances are provided testing
  # perform_B2S - do B2S?
  # B2S_method = SVshape NN LINSHAPE LINES PARAM MONOT
  # b_shape - mean shape of local spectra (for some B2S functions) 
  # lplot - plotting?
  # moments, a_max_times, w_a_fg - prms to be passed to  B2S_shape
  # 
  # 
  #   Return:
  # ENSM, S, band_Ve, b_LSM, band_V_restored
  # 
  # 
  # M Tsy 2023 June
  #--------------------------------------------------------------
  
  nx = dim(B)[1]
  nband = dim(tranfu)[2] ;  J = nband

  tranfu2 = (abs(tranfu))^2 # [i_n, j=1:J]
  
  # Calc symm-pos-def sqrt of CVM
  
  sqB = symm_pd_mx_sqrt(B)$sq
  
  # CREATE the ENSM
  
  gau_mx_N01=matrix(nrow=nx, ncol=ne, data=rnorm(nx*ne))
  ENSM=sqB %*% gau_mx_N01
  
  # Ensm sample CVM
  # Calc ensm perturbations.
  # Always subtract the mean.
  
  # ensm mean
  
  ENSM_Me = rowMeans(ENSM) # ENSM[ix,ie], ave over ie: ENSM_Me[ix]
  
  d_ENSM = ENSM - matrix(ENSM_Me, nrow = nx, ncol = ne)
  S = 1/(ne-1) * d_ENSM %*% t(d_ENSM)
  
  # Calc ensm band vars (and, optionally, their bootstrap versions)
  
  bandData = E2B(ENSM, tranfu, BOOTENS, nB, true_spectra_available)
  band_Ve   = bandData$band_Ve
  if(BOOTENS) band_Ve_B = bandData$band_Ve_B
  
#----------------------------------------------------------------------   
  if(perform_B2S){  # B2S:  FROM band_Ve[x, 1:nband] to bLSM_n(x) section
    
    #----------------------------------------------------------------------
    #    "LINES"
    # Draw segments of straight lines through (nc[j, bmean[j]])
    
    if(B2S_method == "LINES" | B2S_method == "LINSHAPE"){
      
      band_V=band_Ve # band_Ve, band_Vt - for debug

      SPECTRUM_e_lines = B2S_lines(tranfu2, band_V, ne,
                                   B2S_bbc, niter_bbc, correc_bbc, lplot)
      
      b_lines               = SPECTRUM_e_lines$b_lines
      band_V_lines_restored = SPECTRUM_e_lines$band_V_restored
      if(true_spectra_available){
        norm(b_lines - b_true, "F") / norm(b_true, "F")
        mean(b_lines - b_true) / mean(b_true)
      }
    }
    #----------------------------------------------------------------------
    #   LINSHAPE
    # Use the true spectrum's SHAPE to fit b_lines and get a smooth and
    # better-behaving-near-the-origin estimate of b_n 
    
    if(B2S_method == "LINSHAPE"){
      b_tofit = b_lines
      band_V  = band_Ve
      
      SPECTRUM_e_shape = B2S_shape(b_tofit, b_shape, tranfu2, band_V, 
                                   moments, a_max_times,w_a_fg, lplot)
      
      b_linshape            = SPECTRUM_e_shape$b_fit
      band_V_shape_restored = SPECTRUM_e_shape$band_V_restored
      
      if(true_spectra_available){
        norm(b_linshape - b_true, "F") / norm(b_true, "F")
        mean(b_linshape - b_true) / mean(b_true)
        
        norm(b_lines - b_true, "F") / norm(b_true, "F")
        mean(b_lines - b_true) / mean(b_true)
      }
    }
    
    #----------------------------------------------------------------------
    #    SVshape (pseudo-inversion + pararametric fit)
    # Estimate  b_n  using SVD of  Omega
    
    if(B2S_method == "SVshape" | B2S_method == "MONOT" | B2S_method == "PIP"){
      band_V = band_Ve # band_Ve, band_Vt (debug)
      # band_Vt_S1 = cbind(band_Vt, band_Vt[, (J-1):2])
      
      SPECTRUM_SVshape = B2S_SVshape(band_V, Omega_S1, SVD_Omega_S1,
                                     b_shape, moments, band_centers_n,
                                     BOOTENS, band_Ve_B,
                                     nSV_discard, a_max_times, w_a_fg, 
                                     true_spectra_available, lplot)
      b_SVshape = SPECTRUM_SVshape$b_fit
      band_V_SVshape_restored = SPECTRUM_SVshape$band_V_restored
      rel_err_discard_nullspace_b = SPECTRUM_SVshape$rel_err_discard_nullspace_b
      b_svd = SPECTRUM_SVshape$b_svd
    }
    
    #----------------------------------------------------------------------
    #    NN
    # Estimate  b_n  using NN
    
    if(B2S_method == "NN"){
      band_V = band_Ve # band_Ve, band_Vt (debug)
      
      SPECTRUM_NN = B2S_NN(band_V, NN, 
                           TransformSpectrum_type_x, TransformSpectrum_type_y,
                           TransformSpectrum_pow_x, TransformSpectrum_pow_y,
                           Omega_nmax, true_spectra_available, lplot)
      
      b_NN               = SPECTRUM_NN$b_fit
      band_V_NN_restored = SPECTRUM_NN$band_V_restored
      
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
      
    }else if(B2S_method == "PARAM"){
      
      b_LSM=b_param
      band_V_restored = band_V_param_restored
    } 
  }else{ # doesn't perform B2S
    b_LSM = NULL
    band_V_restored = NULL
  }
  
  return(list(ENSM=ENSM,
              S=S,
              band_Ve=band_Ve, # [ix,j]
              b_LSM=b_LSM,
              band_V_restored=band_V_restored))
}