function nX = nonlinear_transform(X)
  nX = X;
  nX = [nX arrayfun(@ (x1, x2) (x1 * x2), X(:,2), X(:,3))];
  nX = [nX arrayfun(@ (x1) (x1 * x1), X(:,2))];
  nX = [nX arrayfun(@ (x2) (x2 * x2), X(:,3))];
end