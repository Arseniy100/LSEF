
B2S_monot = function(band_V, Omega_nmax, 
                   ne, eps_V_sd, band_Ve_TD,
                   b_fg, c_z_fg_rel, 
                   k0_log, c_smoo_monot, 
                   LS_solver, truth_available, lplot){
#*****************************************************************************
# From band variances  band_V, estimate 
# on S1: the modal spectrum  b_l
# on S2: the variance spectrum E_l
# by applying a least-squares optimization approach with non-negative constraints.
# 
# The resulting spectrum is 
# (a) fits the data  z = band_V
# (b) is smooth
# (c) is monotone (the degree is controlled by  c_smoo_monot) 
# (d) fits FG, b_fg
# 
#    Method
#    
# 0) Notation.
#    Let N = nmax
#    At each grid point ix, let
#    
#        y[1:N] = b[n=0 : n=nmax-1]
#       
#   NB: y[nmax] = 0 by assumption.
# 
# 1) Monotonicity.
#    Represent  
# 
#    y[i] = sum z[k=i:N]              (1)
# 
# with z[k] subject to n inequality constraints:  
# 
#    z[k] >= 0                        (2)
#    
# Eq(1) implies that
# 
#    y = Y * z,                        (3)
#   
# whr 
#      |1 1 1 1 1|
#      |0 1 1 1 1|
#  Y = |0 0 1 1 1|
#      |0 0 0 1 1|
#      |0 0 0 0 1| 
#   
# is an N*N mx  
#      
# 2) Impose the data (d==data=ensm variances V) constraint:
# 
# The obs eqn is
#  
#    d = d_obs = Omega * y + err
#    d_mod = Omega * y
# 
# Here d_obs is the ensm band variances vector (of length J), 
#      y is the spectrum we seek (at each ix independently).
#      Omega == Omega_nmax.
#      Omega is a J*N mx.
#      
# The obs eqn wrt z is 
# 
#      d = A*z + err
#      
#      A = Omega * Y
#      
# The data misfit function:     
#      
#    J_d(z) = (Az - d, Az - d)
# 
# 
# 3) Impose a smoothness constraint on z. Let
# 
#   J_s = c[1]*(z_2 - z_1)^2 + ... + c[N]*(z_N - z_{N-1})^2 
# 
# Here c[] is the weighting sequence (=c_smoo_monot  by default, specified here)
#   
# Then,
# 
#   J_s = (Tz, z)
#   
# whr T is the 3-diagonal N*N mx;
# the main diagonal is (c1, (c1+c2), (c2+c3),...,cN)
# the sub and super diagonals are (-c1, -c2, -c3,...,-cN)
# 
# 4) The FG constraint
# 
# J_fg = c_z_fg * ||z - z_fg||^2 == 
#      = c_z_fg_rel / mean_k(z_fg^2) * sum (z[k] - z_fg[k])^2
# 
# Here,  
# c_z_fg = c_z_fg_rel / mean(z_fg[ix,]^2) -- individual for each grid point ix,
# z_fg[N] = y_fg[N] 
# k<N:  z_fg[k] = y_fg[k] - y_fg[k+1]
# 
# Now, we formulate the problem (for each grid point ix individually) as
# =====================================================================
#     J(z) = J_d(z) + J_s(z) + J_fg(z) --> min   subject to z[k] >= 0.
# =====================================================================
# This amounts to the problem, which can be formulated in 2 equiv ways:
# 
# 1) ||C*z - psi|| --> min |    subject to z[j]>=0           (**)
# 
# whr 
# 
#    C = A^T A + T + c_z_fg * I
#    psi = A^T * d + c_z_fg * z_fg
#  
# or
# 
# 2) ||G*z - phi|| --> min |   subject to z[j]>=0           (***)
# 
# whr 
# 
#    G = sqrt(C)
#    phi = 1/G * psi
#    
# With a square invertible G, these 2 formulations coincide.
# ...........................   
# Technically, we call function  
# lsqnonneg(C, d)  from  pracma  or  nnls
# with C=G.
#-------------------------  
#    Args
# 
# band_V[1:nx, 1:J] - [ix, j=band] BAND variances 
#               (normally, estimated from the ensemble or may be the true band varinces) 
# Omega_nmax[1:J, 1:nmaxp1] -- d=Omega_nmax * b[1:nmaxp1]
# ne - ensm size
# eps_V_sd - portion of mean TSD(band_Ve) used as an additive rglrzr 
# band_Ve_TD - theor SD of sampling errors in band_Ve
# b_fg[ix=1:nx, k=1:nx] - FG
# c_fg_rel - weight of FG penalty (rel b_fg_max)
# k0_log - prm in the log-transform of  k=l+1:  t:=log(k+k0_log)
# c_smoo_monot - overall weight of the smoothing penalty
# LS_solver - character string indicating how to solve the variat problem:
#            "ls" - ordinary least-squares
#            "nnls" - nnls from nnls-package
#            "pracma" - lsqnonneg from the pracma-package
# band_Ve_TD - theor SD of sampling errors in band_Ve
# truth_available - TRUE if there is TRUTH available for testing
# lplot - plotting: draw plots only if TRUE
# 
# Return: b_fit[ix, i_n] (the resulting spectra at all grid points),
#         band_V_restored (band variances restored from b_fit)
# 
# 
# M Tsy 2020 Nov, 2021 Mar
#*****************************************************************************

  nx = dim(band_V)[1] 
  J  = dim(band_V)[2]
  nband=J

  # Prelims
  
  nmax=nx/2
  nmaxp1=nmax+1
  nmaxp2=nmax+2
  N = nmax # dim-ty of y
  
  #-----------------------------------------------------------------
  # build Y - transform z \mapsto y
  
  Y = matrix (data = 0, nrow = N, ncol = N)
  
  for (i in 1:N){
    Y[i,i:N] = 1
  }
  
  #-----------------------------------------------------------------
  # build A=Omega*Y
  # Drop the last column in  Omega_nmax  because we postulate here
  # that the spectrum vanishes at the highest wvn
  
  Omega = Omega_nmax[,1:N]
  
  A = Omega %*% Y
  
  #-----------------------------------------------------------------  
  # Calc  z_fg  from  y_fg.
  # (Assuming that  z_fg = 0 at the highest resolvable by the grid wvn.)
  # k=N:  z_fg[N] = y_fg[N]                   (1)
  # k<N:  z_fg[k] = y_fg[k] - y_fg[k+1]       (2)
  # Let in  z & y  the 1st dimension be wvns, not is as in b   
  
  y_fg = t(b_fg[,1:N]) # we treat only nonneg wvns in this function
  z_fg = y_fg  # init and assign (1)
  z_fg[1:(N-1),] = y_fg[1:(N-1),] - y_fg[2:N,]
  
  # # Test 
  # y_fg1_ = Y %*% z_fg[,1]
  # max(abs(y_fg1_-y_fg[,1])) # OK
  
  #-----------------------------------------------------------------
  # Weights for the data constraint.
  # for any ix, Lambda_d = diag(d_weights[,ix])
  
  V_td_min = eps_V_sd * apply(band_Ve_TD, 1, mean)  # mean over bands, [ix] 
  V_sd = band_Ve_TD + V_td_min     # [ix,j]
  
  d_weights = 1/t(V_sd^2)          # [j,ix]

  #-----------------------------------------------------------------  
  # FG weights
  # c_z_fg = c_z_fg_rel / mean(z_fg[ix,]^2) -- individual for each grid point ix,
  # for any ix,   Lambda_fg = diag(z_fg_weights[,ix])

  z_fg_td = sqrt(2/(ne-1)) * z_fg # "theoretical" SD of z_fg   [k,ix]
                                  # (using the formula for SD(sample var) 
  z_fg_td_min = eps_V_sd * apply(z_fg_td, 2, mean) 
  z_fg_sd = t(t(z_fg_td) + z_fg_td_min)
  
  z_fg_weights = c_z_fg_rel / z_fg_sd^2      # [k,ix]
  
  #-----------------------------------------------------------------  
  # Smoothness weights
  
  c_smoo = c_smoo_monot * (c(1:N) + k0_log) 
  
  #-----------------------------------------------------------------  
  # build TT, which defines the smoothness constraint
  # 
  #   J_s = (Tz, z)
  #   
  # TT is the 3-diagonal N*N mx;
  # the main diagonal is (c1, (c1+c2), (c2+c3),...,cN)
  # the sub and super diagonals are (-c1, -c2, -c3,...,-cN)
  
  TT = matrix (data = 0, nrow = N, ncol = N)

  for (k in 1:N){
    
    if(k == 1){
      
      TT[1,1] =  c_smoo[1]
      TT[1,2] = -c_smoo[1]
    
    }else if(k == N){
      
      TT[N,N]   =  c_smoo[N-1]
      TT[N,N-1] = -c_smoo[N-1]
      
    }else{                # all but first & last rows
      
      TT[k,k] =  c_smoo[k-1] + c_smoo[k]
      TT[k,k-1] = -c_smoo[k-1]
      TT[k,k+1] = -c_smoo[k]
    }
  }
  # isSymmetric.matrix(TT) # OK
  #-----------------------------------------------------------------
  # Now, solve the problem for all ix
  # C = G
  # ?? = 1/G * A^TT * d
   
  b_fit = matrix(0, nrow=nx, ncol=nx) # [ix, i_n]
  
  for (ix in 1:nx){
 
    # Build  C = A^T Lambda_d A + Lambda_fg + TT
    #        G = sqrt(C)  (take symm pos def sqrt)
    #        psi = A^T Lambda_d d + Lambda_fg z_fg
    #        phi = 1/G psi (solve G*phi=psi)
  
# ix=(ix+33) %%  nx +1
    LA =  d_weights[,ix] * A
    AtLA = crossprod(A,LA)
    G2 = AtLA + TT + diag(z_fg_weights[,ix])
    G = symm_pd_mx_sqrt(G2)$sq
    
    # norm(TT)/norm(AtLA)
    # norm(diag(z_fg_weights[,ix]))/norm(AtLA)
    # norm(TT)/norm(diag(z_fg_weights[,ix]))
    
    d = band_V[ix,]     # data 
    ATLd = drop( crossprod(A, d_weights[,ix] * d) )
    psi = ATLd + z_fg_weights[,ix] * z_fg[,ix]
    phi = solve(G, psi)  
    
    # Main solver
    # nnls & pracma's lsqnonneg yield exactly the same result but nnls is much faster
    
    if(LS_solver == "nnls"){
      solution = nnls(G, phi)
      
    }else if(LS_solver == "ls"){
      z = solve(G, phi)
      solution = list("x"=z)
      
    # }else if(LS_solver == "pracma"){
    #   solution = lsqnonneg(G, phi)
    } 
    
    z = solution$x
    y=drop( Y %*% z )

    # if(lplot){
    #   y_true = b_true[ix,1:nmax]
    #   mx=max(y_true, y)
    #   plot(y_true, ylim=c(0,mx))
    #   lines(y, col="red")
    #   max(abs(y-y_true)) / max(abs(y_true))
    # }

    b_fit[ix, 1:nmax] = y
    b_fit[ix, (nmaxp1+1):nx] = b_fit[ix, rev(2:nmax)]
  }
  b_fit[, nmaxp1] = b_fit[, nmax] # a patch
  
  
  if(truth_available){
    norm(b_fg - b_true, type="F") / norm(b_true, type="F")
    norm(b_fit - b_true, type="F") / norm(b_true, type="F")
  }
 
  #-----------------------------------------------------------------------
  # Plots
  
  if(lplot & truth_available){
    
    b_fg_Ms = apply(b_fg, 2, mean)
    b_fit_Ms = apply(b_fit, 2, mean)
    
    inm=nx/6
    mx=max(b_fg_Ms[1:inm], b_fit_Ms[1:inm], b_true_Ms[1:inm])
    plot(b_fit_Ms[1:inm], main="b_true_Ms (black) \n b_fit_Ms(red), b_fg_MS(blu)", 
         type="l", col="red", lwd=2, ylim = c(0,mx))
    lines(b_true_Ms[1:inm], lwd=2)
    lines(b_fg_Ms[1:inm], lwd=2, col="blue")
    
    
    mx=max(b_true, b_fit)
    image2D(b_true, main="b_true", zlim=c(0,mx))
    image2D(b_fit, main="b_fit", zlim=c(0,mx))
    
    ix=sample(c(1:nx), 1)
    inm=nx/6
    mn=min(b_true[ix,1:inm], b_fit[ix,1:inm], b_fg[ix,1:inm])
    mx=max(b_true[ix,1:inm], b_fit[ix,1:inm], b_fg[ix,1:inm])
    plot(b_true[ix,1:inm], 
         main=paste0("b_true (circ) \n b_fit(red), b_fg(blu) \n ix=", ix),
         ylim=c(mn,mx), xlab="n+1")
    lines(b_fit[ix,1:inm], col="red")
    lines(b_fg[ix,1:inm], col="blue")
    abline(h=0)
    
    # db=b_fit - b_true
    # ind = which(max(db) == db, arr.ind = TRUE)
    # ix=ind[1]
    # im=ind[2]
    # plot(b_true[ix, 1:im])
    # lines(b_fit[ix, 1:im])
    # b_true[ind]
    # db[ind]
    
    # ix=sample(c(1:nx), 1)
    # inm=nx/6
    # plot(b_true[ix,1:inm]/b_true[ix,1], 
    #      main=paste0("b_true/b_true[1] (circ), b_fit/b_fit[1] \n ix=", ix),
    #      ylim=c(0,1), xlab="n+1")
    # lines(b_fit[ix,1:inm]/b_fit[ix,1], col="red")
  }
  
  sum(b_fit < 0) / sum(b_fit >= 0)
  b_fit[b_fit < 0] = 0
  
  #-----------------------------------------------------------------------
  # Check how well input Band Vars are fitted (restored) by  b_fit
  
  band_V_restored = t( apply( b_fit[,1:N], 1, function(t) drop(Omega %*% t) ) )

  # mx=max(band_V, band_V_restored)
  # image2D(band_V, main="band_V", zlim=c(0,mx), xlab='x', ylab="band")
  # image2D(band_V_restored, main="band_V_restored", zlim=c(0,mx), xlab='x', ylab="band")
  
  norm(band_V_restored - band_V, "F") / norm(band_V, "F")
  
  # j=1
  # plot(band_V[,j])
  # lines(band_V_restored[,j])
  
  #-----------------------------------------------------------------
  
  return(list("b_fit"=b_fit,                      # [ix, i_n]
              "band_V_restored"=band_V_restored)) # band variances restored from b_fit
}