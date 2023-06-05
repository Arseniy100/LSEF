
data_misfit = function(prm, prm_rglrz, b_shape, Omega, V, w_rglrz, V_esd){
    
    # ***************************************************************************
    # Compute a norm of the misfit between the DATA (V) and 
    # the fitting model (V_mod, see below).
    # 
    # prm=c(A,a)
    # prm_rglrz - FG for prm -- used for rglrz
    #  
    # In terms of b[] (the modal spectrum), the fitting model is
    #    y_mod = A*b_shape(n/a)
    # In terms of band variances, the fitting model is
    #    V_mod = Omega %*% y_mod
    # The discrepancy is
    #    dV = V_mod - V
    # The discrepancy norm squared is  
    # 
    #    misfit = crossprod(dV/V_esd, dV/V_esd)
    #    
    # returns the misfit of the spectrum   A*b_shape(n/a)  with the data V
    #    
    # M Tsy 2020 Nov
    # ***************************************************************************
    
    A = prm[1]
    a = prm[2]
    N = length(b_shape)
    
    ii_n=c(1:N)
    
    tt = ii_n / a # the scaled i_n-grid
    b = A * spline(x=ii_n, y=b_shape, xout=tt)$y
    V_mod = Omega %*% b
    dV = V_mod - V
    dV_scaled = dV / V_esd
    misfit = crossprod(dV_scaled, dV_scaled) + 
        w_rglrz*crossprod(prm - prm_rglrz, prm - prm_rglrz)
    
    return(misfit)
  }