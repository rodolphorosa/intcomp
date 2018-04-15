n = 1000;
ein = zeros(1000, 1);

for i = 1:1000
  
  X = unifrnd(-1, 1, n, 2);
  
  X = [ones(n, 1) X];
  
  y = arrayfun(@target, X(:,2), X(:,3));
  
  y = addnoise(y, 0.1);
  
  w = linear_regression(X, y);
  
  ein(i) = error_in(X, y, w);
  
end

fprintf("Average of in-sample errors: %f\n", mean(ein));