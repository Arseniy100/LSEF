
AEV_loss = function(ff, gg, r){
    
    # deviances_per_wvn = r^2 * (f-g)^2 /  ( (f+r)*(g+r)^2 )
    # AEV_LOSS = deviance_AEV_per_wvn[1] + deviance_AEV_per_wvn[nmaxp1] +
    #            2* sum (  deviance_AEV_per_wvn[2:nmax] )
    #          
    #          
    # try 
    # 
    #    (f-g)^2 /  ( (1+ f/r)*(1+ g/r)^2 )
    #  
    #  
    #  Var(f) = sum  (r * f[n]) / (f[n] +r) = sum f / (1 +f/r)
    #  
    #  try loss = delta_var / Var(f)
    #            
    # Works with Torch tensors  ff, gg.
    # 
    #   Args
    # 
    # ff, gg [1:n_sample, 1:nmaxp1] - torch_tensors (dtype = torch_float())
    # r - obs-err variance per wvn squared  (r)  (numeric, scalar)
    # 
    #  M Tsy 2023 Jan
  
    r_tensor = torch_tensor(r, dtype = torch_float())  
    tensor_1 = torch_tensor(1.0, dtype = torch_float()) 
  
    fmg = torch_subtract(ff, gg)
    fmg2 = fmg$pow(2)
    
    #    (f-g)^2 /  ( (1+ f/r)*(1+ g/r)^2 )
    
    fdr = torch_div(ff, r_tensor)
    fdrp1 = torch_add(fdr, tensor_1)
    gdr = torch_div(gg, r_tensor)
    gdrp1 = torch_add(gdr, tensor_1)
    gdrp12 = gdrp1$pow(2)
    denom = torch_mul(fdrp1, gdrp12)
    
    # fpr = torch_add(ff, r_tensor)
    # 
    # gpr = torch_add(gg, r_tensor)
    # gpr2 = gpr$pow(2)
    # 
    # denom = torch_mul(fpr, gpr2)
    # 
    loss_all_wvn_contrib = torch_true_divide(fmg2, denom)

    dVar = loss_all_wvn_contrib$sum()
    
    #  Var(f) = sum [ f / (1 +f/r) ]
    Var_f_contrib = torch_true_divide(ff, fdrp1)
    Var_f = Var_f_contrib$sum()
    
    loss = dVar / Var_f
    
  }
  