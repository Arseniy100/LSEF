
lowpassFlt_S1 = function(f, nx, n_cutoff){
  
  #-------------------------------------------------------------------------------------
  # Lowpass flt f(x) with rectngular transfer fu, n_cutoff
  # 
  #    Args
  # 
  # f - fu to be filtered (assumed to be REAL!)   
  # domain - 'S1' or 'S2'
  # nx - grid size on S1 (assumed to be even).
  # n_cutoff - cut-off wvn
  # 
  # return: f_lpf (lowpass filtered)
  # 
  # M Tsy May 2020
  #-------------------------------------------------------------------------------------
  
  
  nmax = nx/2
  f_lpf = f # init
  
  # forward DFFT
  
  f_fft = fft(f, inverse=FALSE) /nx # cplx
  
  # define the filter's tranfu 
  
  tranfu_lowpass  = tranfuSmoo_lowpass_dscrWvns("S1", nmax, n_cutoff)
  
  # Filtering 
  
  f_fft_lpf = f_fft * tranfu_lowpass
  
  f_lpf= Re( fft(f_fft_lpf, inverse=TRUE) ) 
  
  return(f_lpf)
  
}

  
