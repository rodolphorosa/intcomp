function ein = error_in (X, y, w)
  ein = mean(sign(X*w) != y);
end