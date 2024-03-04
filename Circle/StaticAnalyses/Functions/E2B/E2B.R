
E2B = function(ENSM, tranfu, BOOTENS, nB, true_spectra_available){
  #---------------------------------------------------------------------------------
  # From the Ensemble to Band-filtered Fields  phi (Re and Im parts, pphi1, pphi2),
  # their Variances  band_Ve, and their  cvms.
  # Calc (if  BOOTENS=T) bootstrap version of  band_Ve:  band_Ve_B.
  #
  #   Args
  # 
  # ENSM[ix,ie] - ensemble
  # tranfu[i_n=1:nx, j=1:J] - bands' spectral transfer functions
  #
  #
  # BOOTENS - TRUE if the bootstrap band variances  band_Ve_B  are available
  # nB - size of the bootstrap sample to be generated
  # true_spectra_available - TRUE spectra and true band variances are provided testing
  # (if TRUE,  b_true  and  band_Vt  are taken from the environment!)
  #
  #
  # return: band_Ve, band_Ve_B, H1, H2, H_big1, H_big2, pphi1, pphi2, 
  #         Gamma_phi_1_true, Gamma_phi_2_true, cvm_phi1_estm, cvm_phi2_estm
  #         b_Ms_estm
  #
  #  M Tsy 2021 Mar
  #---------------------------------------------------------------------------------
  
  nx = dim(ENSM)[1]
  ne = dim(ENSM)[2]
  nband = dim(tranfu)[2] ; J = nband
  Jm1=J-1 ;  Jm2=J-2
  
  nmax = nx/2   ; lmax=nmax
  nmaxp1 = nmax +1   ; lmaxp1 = nmaxp1
  nmaxm1 = nmax -1   ; lmaxm1 = nmaxm1
  
  #----------------------------------------------------------------------
  # Fourier transform the ensm
  
  ENSM_fft = mvfft(ENSM, inverse=FALSE) /nx # [i_n, ie]
  
  # sum((Re(ENSM_fft))^2)
  # sum((Im(ENSM_fft))^2)
  
  # estm the spatially mean spectrum
  
  b_Ms_estm = apply(ENSM_fft, 1, Ex2)
  
  #----------------------------------------------------------------------
  # BOOTSTRAP the ensm
  
  if(BOOTENS){
    EEB = array(0, dim = c(nx, ne, nB))
    EEB_fft = array(0, dim = c(nx, ne, nB)) 
    #EB_fft = matrix(0, nrow=nx, ncol=ne*nB) 
    # SSB = array(0, dim = c(nx, nx, nB))
    members = matrix(0, nrow = ne, ncol = nB) # of the ensemble, for all bootstrap samples
    
    for(iB in 1:nB){
      members[,iB] = sample(c(1:ne), ne, replace = TRUE)
      EEB[,,iB] = ENSM[,members[,iB]]
      # dEB = d_ENSM[,members[,iB]]   # centered (ensm perturbations)
      # SSB[,,iB] = 1/(ne-1) * dEB %*% t(dEB)
      EEB_fft[,,iB] =  ENSM_fft[,members[,iB]] 
    }
    
    # SSB_MB = apply(SSB, c(1,2), mean)
    # SSB_DB = apply(SSB, c(1,2), sd)
    
    # image2D(S, main="S")
    # image2D(SSB_MB, main="SSB_MB")
    # image2D(SSB_DB, main="SSB_DB")
    # image2D(abs(S-B), main="S-B")
    # cor(as.vector(abs(S-B)), as.vector(SSB_DB)) #  crl~0.7
    
    # Bootstrap sample (of size nB) of Ve fields 
    
    xi_Ve_B = apply(EEB, c(1,3), var) # EEB[ix,ie,iB], xi_Ve_B[ix,iB]
  }
  #----------------------------------------------------------------------
  # Bandpass FLT the ensm members
  # NB: The complex spectral coefficients produced by the R's function fft 
  #
  # {\bf\tilde f} := {\tilde f_0}, \tilde f_1, \tilde f_2, \tilde f_3, ..., 
  # \tilde f_{n/2-1}, \, {\tilde f_{n/2}},  
  #    \tilde f_{-n/2+1}, \dots, \tilde f_{-3}, \tilde f_{-2}, \tilde f_{-1}  (*)
  #
  # that is, from wvn=0 go to the right up to $n_{max}:=nx/2$, 
  # then jump to the very left to (-nmax+1) (but not to -nmax!)
  # and then go up till wvn==n=nx-1 (equivalently, n=-1). 
  # Thus, all wavenumbers are counted only once, running the whole circle.
  # 
  # NB: With tranfu(n) not symmetric about 0, ENSM_bandpass are CPLX valued!
  # 
  # NB: It is essential that the non-centered ensm is filtered --
  #     otherwise, band_Ve become negatively biased.
  
  ENSM_bandpass = array(data=0, dim=c(nx,ne,nband)) # [ix, ie, band]
  ENSM_fft_tmp = matrix(data=0, nrow=nx, ncol=ne) # [i_n,ie]
  if(BOOTENS) ENSM_bandpass_B=array(data=0, dim=c(nx,ne,J,nB)) # [ix, ie, j, iB]
  
  for (band in 1:nband){
    ENSM_fft_tmp = apply(ENSM_fft, 2, function(t) t*tranfu[,band])
    ENSM_bandpass[,,band] = mvfft(ENSM_fft_tmp, inverse=TRUE)
  }
  
  if(BOOTENS){
    for (band in 1:nband){
      for(iB in 1:nB){
        ENSM_bandpass_B[,,band,iB] = ENSM_bandpass[,members[,iB], band]
      }
    }
  }
  
  # band=J # 1 J
  # plot (ENSM[,band], type="l", xlab="Grid point",
  #       main=paste0("xi & xi_bandpass (red). band=", band,
  #                   "\n Band's center=", band_centers_n[band], " Band's halfwidth=", hhwidth[band]))
  # lines(Re(ENSM_bandpass[,1,band]), type="l", col="red")
  # 
  # e_diff = diff(ENSM[,band])
  # Le = sd(ENSM[,band]) / sd(e_diff)
  # Le
  # 
  # eb_diff = diff(Re(ENSM_bandpass[,1,band]))
  # Leb = sd(Re(ENSM_bandpass[,1,band])) / sd(eb_diff)
  # Leb
  # 
  # band=round(nband/3)
  # plot(Re(ENSM_bandpass[,1,band]), type="l", main=paste0("ie=1: Re(bandpass-flt). band=", band), xlab="x")
  # plot(Im(ENSM_bandpass[,1,band]), main=paste0("ie=1: Im(bandpass-flt). band=", band), xlab="x")
  #----------------------------------------------------------------------
  # Band variances.
  # Explicilty calc and subtract the mean over each band - to compute Var
  # using Ex2==Ex(abs()) - this is essential because  ENSM_bandpass  are CPLX-VALUED
  # (the R's function var() doesn't work well if its argument is complex valued!)
  
  # Ensm mean: each band separately
  
  ENSM_bandpass_Me = apply(ENSM_bandpass, c(1,3), mean) # ENSM_bandpass[ix,ie,j]
  
  # Centering: each band separately
  
  d_ENSM_bandpass = array(data=0, dim=c(nx,ne,nband)) # [ix, ie, band]
  for (ie in 1:ne){
    d_ENSM_bandpass[,ie,] = ENSM_bandpass[,ie,] - ENSM_bandpass_Me
  }
  
  # 
  # image2D(abs(ENSM_bandpass_Me), main=paste0("ENSM_bandpass_Me"))
  # j=3
  # image2D(abs(ENSM_bandpass[,,j]), main=paste0("ENSM_bandpass, j=", j))
  # image2D(abs(d_ENSM_bandpass[,,j]), main=paste0("d_ENSM_bandpass, j", j))
  
  band_Ve = apply(d_ENSM_bandpass, c(1,3), Ex2) *ne/(ne-1) # ENSM_bandpass[ix,ie,j]; band_Ve[ix,j]
  
  #----------------------------------------------------------------------
  # Bootstrap band_Ve_B.
  # NB: ENSM_bandpass_B[ix, ie, j, iB] is cplx valued.
  # Calc ensm mean for each band and each iB
  
  if(BOOTENS){
    # Ensm mean by band
    
    ENSM_bandpass_B_Me = apply(ENSM_bandpass_B, c(1,3,4), mean) # ENSM_bandpass_B[ix,ie,j,iB]
    
    # Centering by band
    
    d_ENSM_bandpass_B = array(data=0, dim=c(nx,ne,J,nB)) 
    
    for (ie in 1:ne){
      d_ENSM_bandpass_B[,ie,,] = ENSM_bandpass_B[,ie,,] - ENSM_bandpass_B_Me
    }
    band_Ve_B = apply(d_ENSM_bandpass_B, c(1,3,4), Ex2) *ne/(ne-1) # d_ENSM_bandpass_B[ix,ie,j,iB]
  
  }else{
    band_Ve_B = NULL
  }
  
  # Play with the bandpass fltred fields
  
  phiCVMs = F
  if(phiCVMs){
    #----------------------------------------------------------------------
    # For each  ix  and each ensm member  ie,  compute the centered 
    # bandpass filtred ensm members: 
    #    pphi[j, ie, ix] = d_ENSM_bandpass[ix, ie, j] -- cplx valued!
    
    pphi = array(0, dim=c(J, ne, nx))  # NB: reversed indices in  pphi   wrt  b_true etc.!
    
    for (ie in 1:ne){
      pphi[,ie,] = t(d_ENSM_bandpass[,ie,]) # reverse indices: pphi[j,ie,ix]
    }
    pphi1 = Re(pphi)
    
    # With symmetric bands 1 and J, the I part, 
    # pphi2[1,,] = pphi2[J,,] = 0
    # Therefore, take only the  [2:(J-1)]  section in Im(pphi)
    
    pphi2 = Im( pphi[2:(J-1),,] )  # bands from 2 to J-1
    
    #----------------------------------------------------------------------
    # Calc sample covs.
    # For each  ix,  compute  
    #  the sample cvm of pphi1, pphi2, & cross-cvm-phi12   
    #    cvm_phi1_estm = 1/(ne-1) * pphi1 *pphi^T  etc.
    
    cvm_phi1_estm  = array(0, dim=c(J,J,nx))
    cvm_phi2_estm  = array(0, dim=c(Jm2,Jm2,nx))
    cvm_phi12_estm = array(0, dim=c(J,Jm2,nx))
    
    for (ix in 1:nx){
      cvm_phi1_estm[,,ix] =  1/(ne-1) * tcrossprod(pphi1[,,ix], pphi1[,,ix])
      cvm_phi2_estm[,,ix] =  1/(ne-1) * tcrossprod(pphi2[,,ix], pphi2[,,ix])
      cvm_phi12_estm[,,ix] = 1/(ne-1) * tcrossprod(pphi1[,,ix], pphi2[,,ix])
    }
    
    #----------------------------------------------------------------------
    # Calc H1 and H2.
    # H[1:J, 1:nx] - the whole circle of wvns
    # 
    # H1[j=1:J, i_n= 1:nmaxp1] - only nonneg wvns
    # H2[j'=1:(J-2), i_n= 1:nmaxm1] - only nonneg wvns w/o n=0 and n=nmax
    #                                 and with j=1 & j=J bands withheld
    #                                 (as these bands are SYMM ==> Im(pphi)=0 there)
    # NB: pphi2 = 0 for the 2 symm bands: j=1 and j-J.
    # 
    # H1(n) = 1/sqrt(2) * (H(n) + H(-n))
    # H2(n) = 1/sqrt(2) * (H(n) + H(-n))
    # 
    # i_n_symm = 2*nmaxp1 - i_n
    
    H = t(tranfu)
    
    H1 = matrix(0, nrow = J,   ncol = nmaxp1)
    H2 = matrix(0, nrow = Jm2, ncol = nmaxm1)
    
    for (n in c(0, nmax)){ # the 2 extreme wvns, whr Im parts =0
      i_n=n+1
      H1[,i_n] = H[,i_n] 
    }
    
    for (n in 1:nmaxm1){  # all other wvns
      i_n = n+1
      i_n_symm = 2*nmaxp1 - i_n  # same as -n
      H1[,i_n] = (H[,i_n] + H[, i_n_symm]) / sqrt(2)
      H2[,n]   = (H[2:(J-1), i_n] - H[2:(J-1), i_n_symm]) / sqrt(2)
    }
    
    # image2D(H1)
    #----------------------------------------------------------------------
    # Calc true theoretic cvms of  phi
    #    Gamma_phi_1 = H1 diag(f) H1^T
    #    Gamma_phi_2 = H2 diag(f) H2^T
    #
    # NB H1 * diag(f)  is multiplying the 1st column of H1 by f1, the 2nd by f2, etc.
    
    Gamma_phi_1_true = array(0, dim = c(J,J,nx))
    Gamma_phi_2_true = array(0, dim = c(Jm2,Jm2,nx))
    
    if(true_spectra_available){
      
      for(ix in 1:nx){
        F_true_columns = matrix(b_true[ix,1:nmaxp1], nrow = nmaxp1, ncol = J)
        F_true_rows = t(F_true_columns)
        
        H1F = H1 * F_true_rows   # Compute H1 \circ F: Schur product to speed up
        Gamma_phi_1_true[,,ix] = tcrossprod(H1F, H1)
      }
      
      for(ix in 1:nx){
        Fm2_true_columns = matrix(b_true[ix,2:nmax], nrow = nmaxm1, ncol = Jm2)
        Fm2_true_rows = t(Fm2_true_columns)
        
        H2F = H2 * Fm2_true_rows   # Compute HF: Schur product to speed up
        Gamma_phi_2_true[,,ix] = tcrossprod(H2F, H2)
      }
      
      lplot = F
      if(lplot){
        
        Gamma_phi_1_true_Ms = apply(Gamma_phi_1_true, c(1,2), mean)
        Gamma_phi_2_true_Ms = apply(Gamma_phi_2_true, c(1,2), mean)
        
        cvm_phi1_estm_Ms = apply(cvm_phi1_estm, c(1,2), mean)
        cvm_phi2_estm_Ms = apply(cvm_phi2_estm, c(1,2), mean)
        cvm_phi12_estm_Ms = apply(cvm_phi12_estm, c(1,2), mean)
        
        
        mx=max(cvm_phi1_estm_Ms, Gamma_phi_1_true_Ms)
        mn=min(cvm_phi1_estm_Ms, Gamma_phi_1_true_Ms)
        image2D(Gamma_phi_1_true_Ms, main="Gamma_phi_1_true_Ms", zlim=c(mn,mx))
        image2D(cvm_phi1_estm_Ms, main="cvm_phi1_estm_Ms", zlim=c(mn,mx))
        plot(diag(Gamma_phi_1_true_Ms), ylim=c(mn,mx), main="Gam1_true_Ms (circ), _estm")
        lines(diag(cvm_phi1_estm_Ms))
        
        
        mx=max(cvm_phi2_estm_Ms, Gamma_phi_2_true_Ms)
        mn=min(cvm_phi2_estm_Ms, Gamma_phi_2_true_Ms)
        image2D(Gamma_phi_2_true_Ms, main="Gamma_phi_2_true_Ms", zlim=c(mn,mx))
        image2D(cvm_phi2_estm_Ms, main="cvm_phi2_estm_Ms", zlim=c(mn,mx))
        plot(diag(Gamma_phi_2_true_Ms), ylim=c(mn,mx), main="Gam2_true_Ms (circ), _estm")
        lines(diag(cvm_phi2_estm_Ms))
        
        
        image2D(cvm_phi12_estm_Ms, main="cvm_phi12_estm_Ms")
        mx=max(cvm_phi12_estm_Ms)
        mn=min(cvm_phi12_estm_Ms)
        plot(cvm_phi12_estm_Ms[1,], ylim=c(mn,mx), main="rows of cvm_phi12_estm_Ms")
        for (j in 2:J){
          lines(cvm_phi12_estm_Ms[j,])
        }
        
        ix=sample(1:nx, 1)
        
        mx=max(cvm_phi1_estm[,,ix], Gamma_phi_1_true[,,ix])
        mn=min(cvm_phi1_estm[,,ix], Gamma_phi_1_true[,,ix])
        image2D(Gamma_phi_1_true[,,ix], main="Gamma_phi_1_true", zlim=c(mn,mx))
        image2D(cvm_phi1_estm[,,ix], main="cvm_phi1_estm", zlim=c(mn,mx))
        plot(diag(Gamma_phi_1_true[,,ix]), ylim=c(mn,mx), main="Gam1_true (circ), _estm")
        lines(diag(cvm_phi1_estm[,,ix]))
        
        
        mx=max(cvm_phi2_estm[,,ix], Gamma_phi_2_true[,,ix])
        mn=min(cvm_phi2_estm[,,ix], Gamma_phi_2_true[,,ix])
        image2D(Gamma_phi_2_true[,,ix], main="Gamma_phi_2_true", zlim=c(mn,mx))
        image2D(cvm_phi2_estm[,,ix], main="cvm_phi2_estm", zlim=c(mn,mx))
        plot(diag(Gamma_phi_2_true[,,ix]), ylim=c(mn,mx), main="Gam2_true (circ), _estm")
        lines(diag(cvm_phi2_estm[,,ix]))
        
        
        mx=max(cvm_phi12_estm[,,ix], Gamma_phi_2_true[,,ix])
        mn=min(cvm_phi12_estm[,,ix], Gamma_phi_2_true[,,ix])
        image2D(cvm_phi12_estm[,,ix], main="cvm_phi12_estm", zlim=c(mn,mx))
        plot(diag(Gamma_phi_2_true[,,ix]), ylim=c(mn,mx), main="Gam2_true (circ), CROSS-12")
        lines(diag(cvm_phi12_estm[,,ix]))
        
        max(abs(cvm_phi12_estm[,,ix])) / max(abs(Gamma_phi_2_true[,,ix]))
        
        # Test band_Vt, band_Ve
        
        band_Vt_ = matrix(0, nrow = nx, ncol = J)
        band_Ve_ = matrix(0, nrow = nx, ncol = J)
        
        for (ix in 1:nx){
          band_Vt_1 = diag(Gamma_phi_1_true[,,ix])
          band_Vt_2 = diag(Gamma_phi_2_true[,,ix])
          
          band_Vt_[ix,] = band_Vt_1 
          band_Vt_[ix,2:Jm1] = band_Vt_[ix,2:Jm1] + band_Vt_2
          
          band_Ve_1 = diag(cvm_phi1_estm[,,ix])
          band_Ve_2 = diag(cvm_phi2_estm[,,ix])
          
          band_Ve_[ix,] = band_Ve_1 
          band_Ve_[ix,2:Jm1] = band_Ve_[ix,2:Jm1] + band_Ve_2
        }
        
        max(abs(band_Vt - band_Vt_)) # OK
        max(abs(band_Ve - band_Ve_)) # OK
        
        # band_Vt_Ms_ = apply(band_Vt_, 2, mean)
        # band_Ve_Ms_ = apply(band_Ve_, 2, mean)
        # plot(band_Vt_Ms_)
        # lines(band_Ve_Ms_)
      }
    } else{ # true_spectra_available = FALSE
      
    }
    
    #----------------------------------------------------------------------
    # Calc  H_big  - obs oprt for CVMs of  pphi1, pphi2:
    # 
    #    H_big * f = ( \vech Gamma_phi_1, \vech Gamma_phi_2)
    #    
    # (\vech vectorizes the lower triangle of a mx)
    # 
    #    H1 * diag(f) * H1^T = Gamma_phi_1 
    # 
    # Gamma_phi_1_j1,j2 = sum_l H1_j1,l * H1_j2,l * f_l   ==>
    #    
    #    H_big(i1D(j1,j2)) = H1_j1,l * H1_j2,l
    
    n_big1 = J*(J+1)/2
    H_big1 = matrix(0, nrow = n_big1, ncol = lmaxp1)
    
    i1D=0
    for (j1 in 1:J){
      for (j2 in j1:J){
        i1D = i1D +1
        H_big1[i1D,] = H1[j1,] * H1[j2,]
      }
    }
    
    # ix=1
    # v1 = H_big1 %*% b_true[ix,1:nmaxp1]
    # v2 = vechMat(Gamma_phi_1_true[,,ix])
    # max(abs(v1-v2))
    
    n_big2 = (J-2)*(J-1)/2
    H_big2 = matrix(0, nrow = n_big2, ncol = lmaxp1)
    
    i1D=0
    for (j1 in 1:Jm2){
      for (j2 in j1:Jm2){
        i1D = i1D +1
        H_big2[i1D,2:lmax] = H2[j1,] * H2[j2,]
      }
    }
    
    # ix=1
    # v1 = H_big2 %*% b_true[ix,1:nmaxp1]
    # v2 = vechMat(Gamma_phi_2_true[,,ix])
    # max(abs(v1-v2))
    
    #----------------------------------------------------------------------
    # TEST  B2S_costfu_lik
    # Only the Re-component:  H1, pphi1, etc. is tested at one ix
    
    ltest_B2S_costfu=F
    
    if(ltest_B2S_costfu & true_spectra_available){ 
      ix=sample(1,1:nx)
      f_true = b_true[ix,1:nmaxp1]
      phi = pphi1[,,ix]
      test_B2S_lik(f_true, phi, H1)
      
      
      f_clim = b_median_1D[1:nmaxp1]
      
      test_B2S_prior(f_true, f_clim)
    }
  }
  #----------------------------------------------------------------------
  
  return(list("band_Ve"=band_Ve,      #[ix=1:nx, j=1:J]
              "band_Ve_B"=band_Ve_B,  #[ix=1:nx, j=1:J, iB=1:nB]
              "b_Ms_estm"=b_Ms_estm)) # [i_n=1:nx]
              
              # "H1"=H1,       # [j=1:J, lp1=1:lmaxp1]
              # "H2"=H2,       # [jm1=1:(J-2), l=1:lmaxm1]  (w/o bands 1 & J, wvns l=0 & lmax)
              # "H_big1"=H_big1, # [i1D=1:n_big1, 1:lmaxp1]
              # "H_big2"=H_big2, # [i1D=1:n_big2, 1:lmaxp1]
              # "pphi1"=pphi1, # [j=1:J,       ie=1:ne, ix=1:nx]
              # "pphi2"=pphi2, # [jm1=1:(J-2), ie=1:ne, ix=1:nx]
              # "Gamma_phi_1_true"=Gamma_phi_1_true,  # [1:J,     1:J,     ix=1:nx]
              # "Gamma_phi_2_true"=Gamma_phi_2_true,  # [1:(J-2), 1:(J-2), ix=1:nx] (w/o bands 1 & J)
              # "cvm_phi1_estm"=cvm_phi1_estm,   # [1:J,     1:J,     ix=1:nx]
              # "cvm_phi2_estm"=cvm_phi2_estm))  # [1:(J-2), 1:(J-2), ix=1:nx] (w/o bands 1 & J)
}
