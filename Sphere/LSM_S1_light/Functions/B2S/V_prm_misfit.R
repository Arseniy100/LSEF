
V_prm_misfit = function(V, e, d_e, d_V, A, a, c, cd, eps_smoo, w_a_rglrz){
    
    # ***************************************************************************
    # Compute a norm of the misfit between the DATA (ensm band vars V) and 
    # the fitting model (V_mod, see below).
    # 
    #     V_mod[] = A * e[], whr
    #    
    #  e_j := Omega * b_shape_scaled_by_a[]
    #    
    #     delta[] := A * e[] - V[]
    #     d_delta = A * d_e[] - d_V[]
    #     
    #     d_delta[j] := delta[j] - delta[j-1]
    # -----------------------------------------------------
    # misfit(A,a) = crossprod(c[], delta[]^2) +
    #               eps_smoo * crossprod(c[], d_delta[]^2)     (*)
    # -----------------------------------------------------
    #
    #   Args
    # 
    # V[1:J] - band variances (the Data)
    # e[1:J] - sum Omega_{jn} g(n/a) -- V_mod wothout multiplying by  A
    # d_e, d_V - differences [j]-[j-1]
    # A - magnitude parameter  
    # a - scale prm
    # c[1:J] - weights of V[j] (inverse sampling error variances)
    # eps_smo - weight of the smoothing (ie 2nd) term in misfit (*)
    # w_a_rglrz - weight of the rglrz constraint ||a-1||
    # 
    # 
    # returns the misfit of the spectrum   A*b_shape(n/a)  with the data V
    #    
    # M Tsy 2020 Nov
    # ***************************************************************************
    
    J=length(V)
    
    #     delta[] := A * e[] - V[]
    #     d_delta = A * d_e[] - d_V[]
    # misfit(A,a) = crossprod(c[], delta[]^2) +
    #               eps_smoo * crossprod(c[], d_delta[]^2)   
    
    delta = A * e - V
    d_delta = A * d_e - d_V
    
    misfit =       crossprod(c,   delta^2) + 
        eps_smoo * crossprod(cd, d_delta^2) +
        w_a_rglrz* (a-1)^2
    
    return(misfit)
  }