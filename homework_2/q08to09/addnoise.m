function ny = addnoise(y, prob)
  ny = y;
  n = length(y);  
  count = fix(n * prob);
  
  for i = 1:count
    random = fix(unifrnd(1, n));
    ny(random) = ny(random) * -1;
  end
end