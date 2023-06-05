D2norm = function(f){
  d2 = diff(diff(f))
  d2n = sum(abs(d2))
}

D4norm = function(f){
  d4 = diff(diff(diff(diff(f))))
  d4n = sum(abs(d4))
}  