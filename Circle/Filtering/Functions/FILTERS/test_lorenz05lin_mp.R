# Test lorenz05lin using the fin-diff test.
# M Tsy May 2018


library(mixAK)
library(MASS)
library(stats)
library(plot3D)
library(psych)

source('lorenz05.R')
source('lorenz05_.R')
source('lorenz05_step.R')
source('lorenz05lin.R')
source('lorenz05lin_step.R')


# setup and starting values for Lorenz

rem=6.37*10^6 # m
rekm=rem/10^3 # km

n=60
xx1_prev_ntime=c(1:n)
xx1_prev_ntime[]=0
F_Lorenz=32
J_Lorenz=3 # defines spatial scale for Lorenz05

step_increase=5
if(F_Lorenz > 32 & n > 60) step_increase=step_increase /2


dt_atm_h=6 /10 *step_increase /10  # model time step,h, "atmospheric time" (fraction of 6h)

T_h=20 # integration intvl, h
#ntime=2000 # the desired mdl integration length

mesh <- 2*pi/n  # rad
mesh_km <- mesh * rekm # grid spacing, km
grid_km <- c(0:(n-1))*mesh_km

step_h=dt_atm_h
ntime=ceiling(T_h / step_h)
ntime


xi=matrix(0, nrow=n, ncol=ntime)
xx1_prev = matrix(0, nrow=n, ncol=ntime)


tgrid_h=c(1:ntime)*step_h
  # 6h atmospheric time ~ 0.05 lorenz ==>
dt_Lorenz = dt_atm_h/6*0.05 # unitless, "Lorenz time"
  
sd_noise_Lorenz=1e-2 # 
seed_ini_cond_Lorenz=4454631
seed_noise_Lorenz=1231441
  
  #set.seed(seed_ini_cond_Lorenz)
  
assumed_mean = 1.2 * F_Lorenz^(1/3) # from Lorenz-05, p.1577, bef Eq(5)
assumed_ms = assumed_mean * F_Lorenz # from Lorenz-05, p.1577, bef Eq(5)
assumed_rms = sqrt(assumed_ms)
assumed_var=assumed_ms - assumed_mean^2
assumed_sd = sqrt(assumed_var)

# Gen ini cond

set.seed(seed_ini_cond_Lorenz)
x1=rnorm(n, mean=assumed_mean, sd=assumed_sd) # ini condition

# smooth ini cond to reduce the Lorenz-05's initial transient period

nsweep=ceiling( 3*(60/n) * J_Lorenz)
xx2=x1

for(sweep in 1:nsweep){
  xx1=xx2
  for (i in 1:n){
    im1=i-1
    if(im1 < 1) im1=im1+n
    
    ip1=i+1
    if(ip1 > n) ip1=ip1-n
    
    xx2[i]=(xx1[im1] + 2* xx1[i] + xx1[ip1]) /4
  }
}

x1=xx2 *2 # *2 because smoothing reduces the magnitude

plot(x1, type="l")


# Gen ini perturbation (of the same magnitude as x1 yet)

set.seed(seed_ini_cond_Lorenz+561)
x_pert=rnorm(n, mean=assumed_mean, sd=assumed_sd) # ini condition

# smooth ini cond to reduce the Lorenz-05's initial transient period

nsweep=ceiling( 3*(60/n) * J_Lorenz)
xx2=x_pert

for(sweep in 1:nsweep){
  xx1=xx2
  for (i in 1:n){
    im1=i-1
    if(im1 < 1) im1=im1+n
    
    ip1=i+1
    if(ip1 > n) ip1=ip1-n
    
    xx2[i]=(xx1[im1] + 2* xx1[i] + xx1[ip1]) /4
  }
}

x_pert=xx2 *2 # *2 because smoothing reduces the magnitude
plot(x_pert, type="l")

# noise=0:

noise_1=rep(0, n)  # model error (system noise)
noise_ntime=matrix(0, nrow=n, ncol=ntime)  # model error (system noise)

#---------------------------------------------
# Test the 1-step-ahead lorenz05lin mdl

y1 = lorenz05_step(x1, n, dt_Lorenz, F_Lorenz, J_Lorenz, noise_1)

plot(x1, type="l")
lines(y1, col="green")
sqrt(mean( (y1-x1)^2) )

#perturb

eps=1e-6
dx=eps* x_pert  #prtbn
x2=x1 + dx
sqrt(mean(dx^2))

plot(x1)
lines(x2)

y2 = lorenz05_step(x2, n, dt_Lorenz, F_Lorenz, J_Lorenz, noise_1)
sqrt(mean( (y2-x2)^2) )

plot(x2, type="l")
lines(y2, col="green")

plot(y1)
lines(y2)

#fcst perturb (Delta y)

Delta_y=y2-y1
sqrt(mean(Delta_y^2))
rms_Dy=sqrt(mean(Delta_y^2))

# Differential

n_pert=1
U=matrix(dx, nrow=n, ncol=1)
noise_spatim=array(noise_ntime, dim=c(n,n_pert, ntime))

LIN=lorenz05lin_step(x1, n, U, n_pert, dt_Lorenz, F_Lorenz, J_Lorenz, noise_spatim[,1,1])
dy=LIN$U_fcst
y1_=LIN$Xref_fcst

norm(as.matrix(y1 - y1_),"f") # =0 OK
norm(as.matrix(y1),"f")

plot(Delta_y, type="l", main="Delta y , dy (green)")
lines(dy[,1], col="green")
sqrt(mean(dy^2))
rms_dy=sqrt(mean(dy^2))
rms_Dy
rms_dy

rel_err=norm(as.matrix(dy - Delta_y),"f") / norm(as.matrix(Delta_y),"f")
rel_err  
# --> decreases almost linearly with dt. OK.
#---------------------------------------------
# Test the multi-step-ahead lorenz05 and lorenz05lin mdls

xx1=lorenz05(x1, n, ntime, dt_Lorenz, F_Lorenz, J_Lorenz, noise_ntime)
#xx_old=lorenz05_(x1, n, ntime, dt_Lorenz, F_Lorenz, J_Lorenz, noise_ntime) 
#max(abs(xx1 -xx_old)) # =0 OK

#rel_err_ntime_w_prev=norm(as.matrix(xx1_prev_ntime - xx1[,ntime]),"f") / norm(as.matrix(xx1[,ntime]),"f")
#rel_err_ntime_w_prev  
#xx1_prev_ntime=xx1[,ntime]

## --> convergence of the nlin mdl as dt -> 0, T=const OK.
## With 667  time steps at the 20h time intvl, the rel numerical integration error is 0.01
## With 6667 time steps at the 20h time intvl, the rel numerical integration error is 0.001
## --> The discretization of the nonlinear model is OK.

xx2=lorenz05(x2, n, ntime, dt_Lorenz, F_Lorenz, J_Lorenz, noise_ntime)

image2D(xx1-xx2)

#---------------------------------------------
# Perturbations

LIN=lorenz05lin(x1, n, U, n_pert, ntime, dt_Lorenz, F_Lorenz, J_Lorenz, noise_spatim)
XX1=LIN$XX
#max(abs(XX1-xx1)) # =0 OK

UUU=LIN$UUU

xx2_lin=xx1 + UUU[,1,]

image2D(xx2)
image2D(xx2_lin)
image2D(xx2_lin - xx2)

rel_err_lin_ntime=norm(as.matrix(xx2_lin[,ntime] - xx2[,ntime]),"f") / norm(as.matrix(xx2[,ntime]),"f")
rel_err_lin_ntime
## --> convergence of the lin-mdl solution to the nlin-mdl solution as eps -> 0, T=const OK.
## With eps=0.1 , the rel linearization error is 0.2
## With eps=0.01 , the rel linearization error is 0.02
## With eps=0.001 , the rel linearization error is 0.002
## ## With eps=1e-6 , the rel linearization error is 2e-6
## --> The tangent Linear Model is OK.
#---------------------------------------------

