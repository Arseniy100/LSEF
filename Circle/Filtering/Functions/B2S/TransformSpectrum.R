  
TransformSpectrum = function(x, type="none", pow=1, inverse = F){

  # Transform (fwd and bckw) spectrum and spectral band variances
  # 
  #  Args:
  #  
  # x - spectrum, 0:nmax,  can be [1:n_sample, 1:nmaxp1]
  # type - transformation type. 
  #        List of allowed types:
  #       "none", "log", "sqrt", "pow", "RELU"
  # pow - power-law transform exponent, (abs(x))^pow
  #  
  # M Tsy 2022 Dec
  
    if(!{inverse}){  # frw
        
      if(type == "none") {
        TransformSpectrum = x
      }else if(type == "log") {
        TransformSpectrum = log(x)
      }else if(type == "sqrt") {
        TransformSpectrum = sqrt(x)
      }else if(type == "pow") {
        TransformSpectrum = (abs(x))^pow
      }else if(type == "RELU") {
        TransformSpectrum = x
        TransformSpectrum[TransformSpectrum < 0] =0
      }else{
        stop("TransformSpectrum: Invalid type")
      } 
      
    }else{ # bckw
      
      if(type == "none") {
        TransformSpectrum = x
      }else if(type == "log") {
        TransformSpectrum = exp(x)
      }else if(type == "sqrt") {
        TransformSpectrum = x^2
      }else if(type == "pow") {
        TransformSpectrum = (abs(x))^(1/pow)
      }else if(type == "RELU") {
        TransformSpectrum = x
        TransformSpectrum[TransformSpectrum < 0] =0
      }else{
        stop("TransformSpectrum: Invalid type")
      } 
      
    }
    return(TransformSpectrum)
  }