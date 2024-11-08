% Define neural network architecture
layers = [
    featureInputLayer(2, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(50, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(50, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'output') % Output u(x,t)
    ];

% Create layer graph and dlnetwork object
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

% Training parameters
numEpochs = 5000;
miniBatchSize = 15;
learningRate = 0.001;
trailingAvg = [];
trailingAvgSq = [];

% Generate training data (X: space, T: time, U: initial condition)
[X, T] = meshgrid(linspace(-1, 1, 500), linspace(0, 1, 500));
X = X(:);
T = T(:);
U = exp(-X.^2);  % Gaussian initial condition
inputData = [X, T]';

% Physics parameters
alpha = 1.75; % Fractional order
sigma = 0.1;  % Diffusion coefficient
d = 2;        % Dimension

% Training loop
for epoch = 1:numEpochs
    % Random mini-batch sampling
    idx = randperm(length(X), miniBatchSize);
    XBatch = X(idx);
    TBatch = T(idx);
    UBatch = U(idx);

    % Prepare the input data
    inputBatch = [XBatch, TBatch]';
    dlInputBatch = dlarray(inputBatch, 'CB');
    dlUBatch = dlarray(UBatch', 'CB');

    % Compute loss and gradients
    [loss, gradients] = dlfeval(@modelLoss, dlnet, dlInputBatch, dlUBatch, alpha, sigma, d);

    % Update the network using Adam optimizer
    [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, trailingAvg, trailingAvgSq, epoch, learningRate);
    
    % Display progress
    if mod(epoch, 1) == 0
        disp(['Epoch ' num2str(epoch) ', Loss: ' num2str(extractdata(loss))]);
    end
end

% Evaluate the trained network
XTest = linspace(-1, 1, 500)';
TTest = linspace(0, 1, 500)';
inputTest = [XTest, TTest]';
dlInputTest = dlarray(inputTest, 'CB');
dlYPred = predict(dlnet, dlInputTest);

% Plot the results
plot(XTest, extractdata(dlYPred));
xlabel('x');
ylabel('Predicted u(x,t)');
title('FPL Equation Solution using PINNs');
grid on;

% Function to compute the loss
function [loss, gradients] = modelLoss(dlnet, dlInputBatch, dlUBatch, alpha, sigma, d)
    % Predict u(x,t)
    dlUPred = forward(dlnet, dlInputBatch);

    % Ensure predictions are real
    dlUPred = real(dlUPred);

    % Convert dlarray to regular array for finite difference operations
    uPredArray = extractdata(dlUPred);

    % Compute the data loss (MSE with initial condition)
    dataLoss = mse(dlUPred, dlUBatch);

    % Compute gradients of u(x,t) with respect to x and t
    dlx = dlInputBatch(1, :);
    dlt = dlInputBatch(2, :);

    % Compute the first-order derivatives
    dlUx = dlgradient(sum(dlUPred, 'all'), dlx, 'EnableHigherDerivatives', true);
    dlUt = dlgradient(sum(dlUPred, 'all'), dlt);

    % Ensure that derivatives are real
    dlUx = real(dlUx);
    dlUt = real(dlUt);

    % Compute the second-order derivatives
    dlUxx = dlgradient(sum(dlUx, 'all'), dlx);

    % Ensure that second-order derivatives are real
    dlUxx = real(dlUxx);

    % Compute the fractional Laplacian using Finite Difference method
    fractionalLaplacian = computeFractionalLaplacian(uPredArray, alpha);

    % Convert fractional Laplacian back to dlarray
    fractionalLaplacian = dlarray(fractionalLaplacian, 'CB');

    % Drift term
    fVal = dlx .* tanh(abs(dlx)/sqrt(d)) .* dlUx;

    % Physics loss (residual of the FPL equation)
    residual = dlUt + fVal - sigma * fractionalLaplacian - dlUxx;
    residual = real(residual); % Ensure residual is real
    physicsLoss = sum(residual.^2, 'all');

    % Total loss (combination of data loss and physics loss)
    loss = dataLoss + physicsLoss;

    % Compute gradients with respect to the network parameters
    gradients = dlgradient(loss, dlnet.Learnables);
end

% Function to compute the fractional Laplacian using Finite Difference method
function fractionalLaplacian = computeFractionalLaplacian(uPredArray, alpha)
    % Parameters
    N = size(uPredArray, 1); % Number of grid points
    L = 2; % Length of the domain
    dx = L / (N - 1); % Grid spacing

    % Construct the finite difference matrix for fractional Laplacian
    D = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if i ~= j
                distance = (i - j) * dx; % Difference between grid points
                D(i, j) = (distance / dx)^(alpha) / dx;
            end
        end
    end
    D = D - diag(sum(D, 2)); % Ensure the matrix is a valid operator

    % Apply the fractional Laplacian operator
    fractionalLaplacian = D * uPredArray;

    % Ensure the result is real
    fractionalLaplacian = real(fractionalLaplacian);
end
