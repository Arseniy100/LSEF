
logistic = function(x, b=1)( (1+exp(b))/(1+exp(b-x)) )

# x \to  -\infty  ===>  g(x,b) \to  0
# x \to  +\infty  ===>  g(x,b) \to  1 + exp(b)