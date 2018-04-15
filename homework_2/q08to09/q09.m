% Number of samples
n = 1000;

% Dataset 
X = unifrnd(-1, 1, n, 2);

% Adds x0 = 1 to each sample of the dataset
X = [ones(n, 1) X];

% Calculates target function for each sample in the dataset
y = arrayfun(@target, X(:,2), X(:,3));

% Generates simulated noise
y = addnoise(y, 0.1);

% Nonlinear transformation of X
X = nonlinear_transform(X);

% Runs linear regression algorithm 
w = linear_regression(X, y);

% Classifies samples according to hipothesis w
h = sign(X * w);

g1 = @ (x1, x2) sign(-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
  + 1.5 * x1 * x1 + 1.5 * x2 * x2);

g2 = @ (x1, x2) sign(-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ... 
  + 1.5 * x1 * x1 + 15 * x2 * x2);

g3 = @ (x1, x2) sign(-1 - 0.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
  + 15 * x1 * x1 + 1.5 * x2 * x2);

g4 = @ (x1, x2) sign(-1 - 1.05 * x1 + 0.08 * x2 + 0.13 * x1 * x2 ...
  + 0.05 * x1 * x1 + 0.05 * x2 * x2);

g5 = @ (x1, x2) sign(-1 - 0.05 * x1 + 0.08 * x2 + 1.5 * x1 * x2 ...
  + 0.15 * x1 * x1 + 0.15 * x2 * x2);

agrees = zeros(5, 1);

g = arrayfun(@(x1, x2) g1(x1, x2), X(:,2), X(:, 3));
agrees(1) = mean(g != h);

g = arrayfun(@(x1, x2) g2(x1, x2), X(:,2), X(:, 3));
agrees(2) = mean(g != h);

g = arrayfun(@(x1, x2) g2(x1, x2), X(:,2), X(:, 3));
agrees(3) = mean(g != h);

g = arrayfun(@(x1, x2) g2(x1, x2), X(:,2), X(:, 3));
agrees(4) = mean(g != h);

g = arrayfun(@(x1, x2) g2(x1, x2), X(:,2), X(:, 3));
agrees(5) = mean(g != h);

fprintf("Agreements for g1 ... g5 to hipothesis w: [%f %f %f %f %f]\n", agrees);