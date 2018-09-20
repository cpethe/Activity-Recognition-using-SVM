% SGD Kaggle submission

load('./q3_2_data.mat');

X = cat(2, trD, valD);

X_squared_features = X .* X;

X = cat(1, X, X_squared_features);

y = cat(1, trLb, valLb);

d = size(X, 1);
n = size(X, 2);

max_epoch = 20;
eta0 = 1;
eta1 = 500;
C = 66;

labels = unique(y);
k = length(labels);

W = zeros(d, k);

for index = 1:length(y)
    y(index) = find(labels==y(index));
end

plot_objective_values = zeros(max_epoch, 1);
plot_epochs = zeros(max_epoch, 1);

for epoch = 1:max_epoch
    eta = eta0 / (eta1 + epoch);
    random_is = randperm(n);
    objective_function = 0;
    for iter = 1:n
        i = random_is(iter);
        xi = X(:, i);
        yi = y(i);
        %compute yi_hat
        yi_hat = 0;
        maxval = log(0);
        for m = 1:k
            if m ~= yi
                if transpose(W(:, m)) * xi > maxval
                    yi_hat = m;
                    maxval = transpose(W(:, m)) * xi;
                end
            end
        end
        flag = 0;
        second_loss_term = 0;
        term1 = transpose(W(:, yi_hat)) * xi;
        term2 = transpose(W(:, yi)) * xi;
        calc_val = term1 - term2 + 1;
        if calc_val > 0
            second_loss_term = calc_val;
            flag = 1;
        end
        sum_wj_norm = 0;
        for j = 1:k
            derivative = (1 / n) * W(:, j);
            if flag == 1
                if j == yi
                    derivative = derivative - (C * xi);
                elseif j == yi_hat
                    derivative = derivative + (C * xi);
                end
            end
            W(:, j) = W(:, j) - (eta * derivative);
            sum_wj_norm = sum_wj_norm + (norm(W(:, j)) * norm(W(:, j)));
        end
        first_loss_term = sum_wj_norm / (2 * n);
        second_loss_term = C * second_loss_term;
        objective_function = objective_function + first_loss_term + second_loss_term;
    end
    plot_objective_values(epoch) = objective_function;
    plot_epochs(epoch) = epoch;
end

plot(plot_epochs, plot_objective_values);
title('Loss value in each epoch');
xlabel('Epoch');
ylabel('Loss');

X_val = tstD;
X_val_squared = X_val .* X_val;
X_val = cat(1, X_val, X_val_squared);

k = 10;

n = size(tstD, 2);

y_pred = zeros(n, 1);
for i = 1:n
    max_val = log(0);
    xi = X_val(:, i);
    for iter = 1:k
        weight_vec = W(:, iter);
        op = transpose(weight_vec) * xi;
        if op > max_val
            max_val = op;
            prediction = iter;
        end
    end
    y_pred(i) = labels(prediction);
end
