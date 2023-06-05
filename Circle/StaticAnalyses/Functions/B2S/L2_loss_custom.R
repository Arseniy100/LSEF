
L2_loss_custom = function(ff, gg, c){
  
  # loss = sum( (c*(ff-gg))^2 )
  #            
  # Works with Torch tensors.
  # 
  #   Args
  # 
  # ff, gg [1:n_sample, 1:nmaxp1] - torch_tensors (dtype = torch_float())
  # r - obs-err variance per wvn squared  (r)  (numeric, scalar)
  # 
  #  M Tsy 2023 Jan
  
  c_tensor = torch_tensor(c, dtype = torch_float())  
  
  fmg = torch_subtract(ff, gg)
  fmg2 = fmg$pow(2)
  
  # loss = torch_mul(fmg2, c_tensor$pow(2))$sum()
  loss = torch_mul(fmg2, c_tensor$pow(2))$mean()
  
}
  