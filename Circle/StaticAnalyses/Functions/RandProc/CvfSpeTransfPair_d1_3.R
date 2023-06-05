CvfSpeTransfPair_d1_3 <- function(dim_phy_space, n, fun, inverse){
  
  # Compute forw & backw FFT of ISOTROPIC functions in R^1 or R^3.
  #
  # dim_phy_space - dim-ty of the phys space: =1 or =3 are now allowed.
  # n - gridsize
  # fun(1:n) - input cvf or spe: NB: on the whole circle!
  # inverse: =FALSE for the Forward  transf (cvf -> spe), 
  #          =TRUE  for the Backward transf (spe -> cvf)
  #
  # M Tsy Mar 2017
  
  # Debug
  #dim_phy_space=3
  # fun=crf_tru
  # fun=transform
  # EndDebug

  #------------------------------------------------
  # d=1. Tested ok.
  
  if(dim_phy_space == 1){
    if(inverse == FALSE) { # forw transf
      
      spe = fft(fun, inverse = FALSE)/n
      
      how_im = sqrt( sum((Im(spe))^2) / sum((Mod(spe))^2) ) # check
      if( abs(how_im) < sqrt(.Machine$double.eps) ){        # nrm: spe density sh.be real
        
        transform=Re(spe)
        
      }else{                                                # smth is wrong
        message("Non-real spe density")
        print(how_im)  
        print(spe)
        stop("iNon-real spe density")
      }
    
    }else{                 # backw transf
      
      cvf = fft(fun, inverse = TRUE)
      
      how_im = sqrt( sum((Im(cvf))^2) / sum((Mod(cvf))^2) ) # check
      if( abs(how_im) < sqrt(.Machine$double.eps) ){        # nrm: cvf sh.be real
      
        transform=Re(cvf)
        
      }else{                                                # smth is wrong
        message("Non-real cvf")
        print(how_im)  
        print(cvf)
        stop("Non-real cvf")
      }
    }                       # end d=1 forw-backw block
    
    #------------------------------------------------
    # d=3. Tested ok.
    
  }else if(dim_phy_space == 3){
    
    # Construct the odd function y=x written to xfu in the same way the input to fft 
    # is to be written:
    # (1) from 0 to pi and then 
    # (2) back to 0. In (2), do not repeat neither pi nor 0.
    
    xfu = c(1:n) # init, to be fr -n/2 to n/2
    xfu[1:((n/2)+1)]=c(0:(n/2))  # first half
    xfu[((n/2)+2):n]=rev(-xfu[2:(n/2)]) # second half 
    xfu_rad = xfu * pi/(n/2)  # fr -pi to pi
    xfu_k = xfu               # fr -n/2 to n/2
    
    xfu_k_nozero_inv = xfu         # fr -n/2 to n/2
    xfu_k_nozero_inv[1]=1  # instead of 0, so that the inversion here yields 1, not infty
    xfu_k_nozero_inv = 1/xfu_k_nozero_inv
    
    xfu_rad_nozero_inv = xfu_rad   # fr -pi to pi
    xfu_rad_nozero_inv[1]=1  # instead of 0, so that the inversion here yields 1, not infty
    xfu_rad_nozero_inv = 1/xfu_rad_nozero_inv
    
    if(inverse == FALSE) { # forw transf
      
      # Forward 1-D FFT of the 3-D isotropic cvf:
      # Define phi(x)=x*B(x), x>0
      # Extend phi to x<0 (as an odd fu)
      # Perform FFT of phi(x), getting Phi(k)
      # Compute the resulting spe density:
      # f(k)=i/(2*pi*k) * Phi(k)
      #
      # For k=0, recompute spe(k=0)=1/(4*pi^2) * sum_j=1^n rho_j^2 * B_j * (2\pi)/n
      
      phi = xfu_rad*fun
      phi[(n/2)+1]=0  # for phi to be truly odd
      Phi = fft(phi, inverse = FALSE)/n
      spe = 1i /(2*pi) * xfu_k_nozero_inv * Phi
      spe[1] = 1/(2*pi*n) * sum( xfu_rad^2 * fun )
      
      how_im = sqrt( sum((Im(spe))^2) / sum((Mod(spe))^2) ) # a check
      if( abs(how_im) < sqrt(.Machine$double.eps) ){        # nrm: spe density sh.be real
        
        transform=Re(spe)
        
      }else{                                                # smth is wrong
        message("Non-real spe density")
        print(how_im)  
        print(spe)
        stop("iNon-real spe density")
      }
      
    }else{                 # backw transf
      
      # Backward 1-D FFT of the 3-D isotropic cvf:
      # Define psi(k)=k*f(k), k>0
      # Extend psi to k<0 (as an odd fu)
      # Perform FFT of psi(k), getting Psi(x)
      # Compute the resulting cvf:
      # B(x)=(2*pi)/(i*x) * Psi(x)
      #
      # For x=0, recompute cvf(0)=2*pi * sum_k k^2 * f_k 
      
      psi = xfu_k*fun
      psi[(n/2)+1]=0  # for psi to be truly odd
      Psi = fft(psi, inverse = TRUE)
      cvf = -1i*2*pi * xfu_rad_nozero_inv * Psi
      cvf[1] = 2*pi * sum( xfu_k^2 * fun ) 
      
      how_im = sqrt( sum((Im(cvf))^2) / sum((Mod(cvf))^2) ) # a check
      if( abs(how_im) < sqrt(.Machine$double.eps) ){        # nrm: cvf sh.be real
        
        transform=Re(cvf)
        
      }else{                                                # smth is wrong
        message("Non-real cvf")
        print(how_im)  
        print(cvf)
        stop("Non-real cvf")
      }
    }                       # end d=3 forw-backw block
  #------------------------------------------------
  }else{
    message("Only dim_phy_space =1 or =3 are allowed")
    print(dim_phy_space)  
    stop("Wrong dim_phy_space")
  }
  
  return(transform)
}
