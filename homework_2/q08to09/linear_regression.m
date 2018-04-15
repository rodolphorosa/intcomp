function w = linear_regression(X, y)
  w = pinv(X) * y;
end
  