  
NmonotSpec = function(f){
  # detect significantly non-monotone behaviour in f 
  # (which should, normally, decrease)
  
  if(max(f) > 1.2*f[1]) {
    out=F
  }else{
    out = T
  }
  return(out)  
}