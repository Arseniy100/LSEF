#------------------------------------------------------------------
#------------------------------------------------------------------
# ANLS
# Perform the secondary KF's anls with:
# B=B_true
# B=S_lcz
# B=B_LSM
# 
# I.e. compute the 3 respective anls-err cvms A
# A = (I-KH) B_true (I-KH)^T + K R K^T
# (B_true=B here)

#----------------------------------
# Preparations to the ANLS

# FG=0

x_f = c(1:nx) ;  x_f[] =0

# Generate truth using the true fcst-err mdl: xi=W*N(0,I):
# x_true = x_f - e_f = -xi (minus the FG error)
# Since the CVM of xi is B=W*W^T, we simulate xi as 
# xi=W*alpha, whr alpha is the N(0,I) noise

alpha = rnorm(nx, mean=0, sd=1) 
xi = W %*% alpha
x_true = -xi

# Specify obs-err SD

var_FG_err = median(diag(B))
sd_obs = sd_obs_rel_FG * sqrt(var_FG_err)

#--------------
# Generate OBS: x_obs, H, R

OBS = gen_obs(x_true, n_obs, sd_obs)

H = OBS$H
R = OBS$R
x_obs = OBS$x_obs

#----------------------------------
# ANLS

ANLS_opt = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, B, R, B)
ANLS_LSM = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, B_LSM, R, B)
ANLS_lcz = lin_determ_anls(as.matrix(x_f), as.matrix(x_obs), H, S_lcz, R, B)
  