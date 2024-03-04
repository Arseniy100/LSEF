
tranfuBandpassExpq = function(nmax, nc, halfwidth, 
                              q_tranfu=3, rectang = FALSE)   {
  #-------------------------------------------------------------------------------------
  # Specifies transfer function for a dscr linear  Lowpass  filter on S1 & S2.
  # tranfu needs to be smooth (to prevent oscillations in the filtered signal).
  # The functional form of tranfu is
  # 
  #       f(n) = exp(-|(n-nc)/halfwidth|^q_tranfu)                      (1)
  #       
  #------
  # In order to ensure positive definiteness of tranfu
  # (we may need this for FFT^{-1} [tranfu^2] to be positive)
  # (at least for q_tranfu=2),
  # we may replace the circular distance  dn=n-nc  by the chordal distance
  # r = 2*sin(rho/2) 
  # where        rho = 2*pi*dn/nx
  # and return to the n-scale: 
  #    
  #    dn_chordal = nx*r / (2*pi) = nx/pi * sin(dn/(nx/pi))
  #    
  # NB: this option is not active now (commented) 
  #------
  #  tranfu  
  # |        
  # |                         .
  # |                  .             .
  # |                              
  # |              .                     .
  # |
  # |            .                         .
  # |             
  # |          .                             .
  # |         
  # |      .                                     .
  # |_._______________________________________________.________________ n
  #                           nc       
  # 
  # NB: For the S1 application, 
  #     we allow for a non-even tranfu(n) so that respfu will be cplx valued.
  # NB: For the S2 application, 
  #     just take tranfu[i_n=1:nmaxp1]  and ignore the rest, i_n=nmaxp2:nx, and
  #     ignore respfu.
  #        
  #    Args
  #    
  # nmax - defines support of tranfu: nmax = max wvn resolvable on the grid on S1 or S2:
  #        S1: for the grid with nx points, nmax=nx/2 and supp(tranfu) is [-nmax+1, nmax].
  #            Correspondingly, tranfu is a vector of length nx, whr the wns go as follows:
  #            0,1,...,nmax (the upper half-circle), then
  #            nmax-1, nmax-2,..., 1. 
  #        S2: for the regular grid with nlon grid points over a lat circle, nmax = nlon/2.
  # nc - center wvn (nc >= 0 !)
  # halfwidth - see Eq(1)
  # q_tranfu - exponent, see Eq(1)
  #     NB: If q_tranfu>100, tranfu is rectangular!
  # rectang - if TRUE, tranfu is rectangular, =1 iff |n-nc| < halfwidth
  #           NB: The endpoints nc +- halfwidth are NOT included in the supp.
  #
  # return: tranfu (of size 2*nmax)
  #         respfu (cplx, check that Im(respfu) \approx 0)
  #         
  # 
  # M Tsy May 2020, Feb 2021
  #-------------------------------------------------------------------------------------

  # debug
    # nmax=60
    # nc=20 
    # halfwidth=10
    # q_tranfu=3
    # rectang = FALSE # TRUE
  # end debug
  
  nx=2*nmax
  nmaxp1 = nmax +1
  
  tranfu = c(1:nx) ;  tranfu[] = 0 
  
  #--------------------------------------------------------------------
  # Main part
  
  for (n in 0:(nx-1)) {
    i_n=n+1
    
    if(rectang){ # rectangular tranfu case

      if(abs(n-nc) < halfwidth)  tranfu[i_n] = 1
     
    }else{       # "normal" case
      
      dn = n - nc
      if(dn >  nx/2) dn = dn - nx
      if(dn < -nx/2) dn = dn + nx
      tranfu[i_n] = exp( - ( abs( dn/halfwidth ) )^q_tranfu )
      
      # nxdpi=nx/pi
      # dn_chordal = nxdpi * sin(dn/nxdpi)
      #tranfu[i_n] = exp( - ( abs( dn_chordal/halfwidth ) )^q_tranfu )
    }
  }
  
  #--------------------------------------------------------------------
  # Plots
  
  #plot(tranfu[1:nmaxp1], main="tranfu, upper")
  #tranfu[1:55]
  #sum(tranfu>0)

  #plot(tranfu[1:nx], main="tranfu")
  #sum(tranfu>0)
  
  #--------------------------------------------------------------------
  # response fu
  
  respfu = fft(tranfu, inverse=TRUE)  # [i_n]  cplx !

  # nm=nmax/1
  # plot(tranfu[1:nm], type="l", 
  #      main=paste0("Transfer fu. Center wvn=", nc),
  #      xlab = "wvn")
  # plot(abs(respfu[1:nm]), type="l", 
  #      main=paste0("Response fu. Center wvn=", nc),
  #      xlab = "Distance, meshes")
  
  return( list("tranfu"=tranfu, "respfu"=respfu) ) # NB: CPLX valued!
  
}

  
