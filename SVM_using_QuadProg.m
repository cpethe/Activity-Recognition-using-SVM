% Quadprog implementation

run('./vlfeat-0.9.21/toolbox/vl_setup')

load('./SVM_data.mat');

d = size(trD, 1);
n = size(trD, 2);

X = trD;
y = trLb;

%C = 0.1;
C = 10;

H = zeros(n, n);

for i = 1:n
    for j = 1:n
        xi = X(:, i);
        xj = X(:, j);
        H(i, j) = dot(xi, xj) * y(i) * y(j);
    end
end

f = ones(n, 1);
f = -f;

A = [];
b = [];

Aeq = transpose(y);
beq = 0;

l = zeros(n, 1);
u = C * ones(n, 1);

[alphas, fval, exitflag, output, lambda] = quadprog(H, f, A, b, Aeq, beq, l, u);

% Computing weights and bias from alphas
mat2 = alphas .* y;
w = X * mat2;
tempmat = (C / 2) * ones(n, 1);
tempmat2 = abs(tempmat - alphas);
[minval, minindex] = min(tempmat2);
bias = y(minindex) - (transpose(w) * X(:, minindex));
weights = w;

% Objective value returned by quadprog
objective_value = -fval;

supportVectorCount = 0;
% values which are greater than epsilon (a very small value) are support vectors
for i = 1:size(alphas, 1)
    if alphas(i) > C / 1000
        supportVectorCount = supportVectorCount + 1;
    end
end

% Computing the confusion matrix and accuracy on validation data
n_val = size(valD, 2);
true_pos = 0;
true_neg = 0;
false_pos = 0;
false_neg = 0;
for i = 1:n_val
    xi = valD(:, i);
    yi = valLb(i);
    pred_value = (transpose(w) * xi) + bias;
    if pred_value > 0
        y_pred = 1;
    else
        y_pred = -1;
    end
    if yi == 1 && y_pred == 1
        true_pos = true_pos + 1;
    elseif yi == -1 && y_pred == -1
        true_neg = true_neg + 1;
    elseif yi == 1 && y_pred == -1
        false_neg = false_neg + 1;
    elseif yi == -1 && y_pred == 1
        false_pos = false_pos + 1;
    end
end

confusion_matrix = [true_pos, false_pos; false_neg, true_neg];
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg);

accuracy
objective_value
supportVectorCount
confusion_matrix
