function iters = perceptron(X, y, w, max_iter)
  retval = 0;
  wt = w;
  
  while(iters < max_iter)
    m = find(sign(X*wt) != y);
    
    if(length(m) == 0)
      break;
    end
    
    n = fix(unifrnd(1, length(m)));
    
    if(isnan(n))
      n = 1;
    end
    
    wt = wt + X(:, m) * y(m);
    
    iters = iters + 1;
  end
end