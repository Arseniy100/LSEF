
wL2_loss = function(ff, gg, w0, nw1){
    #-------------------------------------------------------------------------------
    # Weighted L2 loss (squared weighted L2 norm of the difference ff-gg).
    # Assign more weight to low wvns:
    # 
    # n >  nw1 : w(n) = 1
    # n <= nw1 : w(n) = 1 + (w0 - 1)* ( cos(n*pi/nw1) + 1 ) /2
    #  
    #   Args
    # 
    # ff, gg [1:n_sample, 1:nmaxp1] - torch_tensors (dtype = torch_float())
    # w0 - weight for n=0
    # nw1 - wvn above which the weights are =1 
    # 
    # ==> Doesn't really improve the results compared to the usual L2 loss.
    # 
    #  M Tsy 2024 Jan
    #-------------------------------------------------------------------------------
  
    nmaxp1   = dim(ff)[2] 
    nsamples = dim(ff)[1]
    
    ww = c(1:nmaxp1) ;  ww[] = 1
    nw1p1 = nw1 +1
    ww[1:nw1p1] = sqrt(1 + (w0 - 1) * ( cos(c(0:nw1) * pi/nw1) + 1 ) /2)
    ww2D = t(matrix(ww, nrow = nmaxp1, ncol = nsamples))
    
    fmg = torch_subtract(ff, gg)
    fmg_m_ww = fmg * ww2D
    d2 = fmg_m_ww$pow2
    d2 = torch_mul(fmg_m_ww, fmg_m_ww)
    
    # loss_samples = apply( ff-gg, 1, function(t) sum((t*ww)^2) )
    # loss = mean(loss_samples)
    
    loss = mean(d2)
  }
  
