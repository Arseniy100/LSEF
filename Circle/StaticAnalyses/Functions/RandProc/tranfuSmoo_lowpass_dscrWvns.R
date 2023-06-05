
tranfuSmoo_lowpass_dscrWvns = function(domain, nmax, nflat, 
                                       d = 0, 
                                       transition_type = 'cos', 
                                       ASYMM = FALSE)   {
  #-------------------------------------------------------------------------------------
  # Specifies transfer function for a dscr linear  Lowpass  filter on S1 & S2
  # with a tunable degree of smootness (which is inteded to prevent oscillations
  # in the filtered signal).
  # 
  # tranfu
  # |        flat              transition            zero
  # |         area                 area              area
  # |--------------------------.
  # |                              .
  # |                              
  # |                                 .
  # |
  # |                                   .
  # |
  # |                                     .
  # | 
  # |                                        .
  # |____________________________________________.________________ n
  #                          nflat            nflat+d
  #           
  #
  # (1) The flat area extends from n=0 to |n| = nflat (inclu), whr
  #    tranfu(n)=1
  # (2) The transition (of length d) area is from |n| = nflat to |n| = nflat+d,
  #  where 
  #    tranfu(n) = f_transition(x), whr
  #    x=(n - n_flat)/d) and
  #    f_transition = 0.5* ( 1 + cos(pi*(|n|-nflat)/d)  ) if transition_type = 'cos' or
  #    f_transition = 1 - (|n|-nflat)/d)                  if transition_type = 'lin'
  #    f_transition = exp( -mult**(|n|-nflat)/d)^2 )      if transition_type = 'exp2' (mult is set up here)
  # (3) The zero area: |n| > nflat+d or |n|>nmax:
  #    tranfu(n) = 0
  #    
  # NB: If the transition zone goes beyond nmax, it is not applied
  #     (so that if nflat=nmax, then tranfu=1 everywhere).
  # NB: If d=0, tranfu is rectangular (there is no transition zone at all).
  #    
  #    Methodology
  #    
  # On S1, first fill in the positive-wvn half-circle. 
  #        At the end copy the positive-wvn half-circle to the negative wvns.
  # 
  # 
  #    Args
  #    
  # domain - 'S1' or 'S2'
  # nmax - defines support of tranfu: nmax = max wvn resolvable on the grid on S1 or S2:
  #        S1: for the grid with nx points, nmax=nx/2 and supp(tranfu) is [-nmax+1, nmax].
  #            Correspondingly, tranfu is a vector of length nx, whr the wns go as follows:
  #            0,1,...,nmax (the upper half-circle), then
  #            nmax-1, nmax-2,..., 1. 
  #        S2: for the regular grid w nlon grid points over a lat circle, nmax = nlon/2.
  #            Correspondingly, tranfu is a vector of length nmax+1, whr the wns go as follows:
  #            0,1,...,nmax.
  # nflat - size of the flat (tranfu=1) area (in terms of n, not np1)
  # d - size of the transition area (an integer), whr 
  #     tranfu(n) goes from 1 to 0 (including both ends).
  #     NB: d is recommended to be at least 6 (to be resolved on the integer-wvn grid),
  #         with d>=12 looking smoother on the dscr wvn grid.
  #     It's better (for interpretation by the user) to specify an EVEN d.
  # transition_type - type of the function in the transition area, 
  #              = 'cos' or 'lin' or exp2'
  #              NB: transition_type = cos or exp2 makes little difference in the flt respfu.
  #              NB: 'lin' is present here only for comparison: it produces much worse spat lcz then 'cos'. 
  #              # ===> smoother tranfu implies much shorter impulse response function's effective supp!
  # ASYMM - if FALSE, then the range is symmetric wrt n=0
  #         if TRUE and if d=0, then the lowest negative-wvn is withheld from supp(tranfu)
  #
  # return: tranfu (of size 2*nmax in case of S1 and nmax+1 in case of S2)
  # 
  # M Tsy Apr 2020
  #-------------------------------------------------------------------------------------
  
  nmaxp1 = nmax +1
  nflatp1 = nflat+1
  
  if(domain == 'S1'){
    nx=2*nmax
    tranfu = c(1:nx)
  }else if(domain == 'S2'){
    tranfu = c(1:nmaxp1)
  }else{
      message("tranfuSmoo_lowpass_dscrWvns. Wrong domain=")
      print(domain)  
      stop("stop")
  }
  
  tranfu[] = 0  # zero domain everywere by default.
    
  # Flat-tranfu wvn domain  
  
  tranfu[1:nflatp1] = 1
  
  # Transition domain
  
  mult=2 # 1.5...3 the greater the faster the decrease in f_transition(x) if it is 'exp2'
  
  if( d > 0){
    for ( np1 in (nflatp1 +1):(nflatp1 +d) ) {
      
      if(np1 <= nmaxp1){
        
        if(transition_type == 'cos'){
          tranfu[np1] = 0.5* ( 1 + cos(pi*(np1 - nflatp1)/d) )
        }else if(transition_type == 'lin'){
          tranfu[np1] = 1 - (np1 - nflatp1)/d
        }else if(transition_type == 'exp2'){
          tranfu[np1] = exp( - ( mult * (np1 - nflatp1)/d )^2 )
        }
        
      }else{            # beyond the spectrum
        tranfu[np1] = 0
      }
    }
  }

  # Fill in tranfu for negative wvns (tranfu(n) is an Even fu)
  
  if(domain == 'S1'){
    for ( np1_upp in 2:nmax ) {  # upper half-circle
      np1 = nx - np1_upp +2
      tranfu[np1] = tranfu[np1_upp] 
    }
    
    if(ASYMM == TRUE & d == 0){ # withhold the lowest neg wvn from supp(tranfu)
      np1_upp = nflat+1
      np1 = nx - np1_upp +2
      tranfu[np1] = 0
    }
  }
  
  #plot(tranfu)
  
  return(tranfu)
  
}

  
