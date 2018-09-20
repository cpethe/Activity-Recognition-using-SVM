% SGD implementation

load('./SVM_data.mat');

d = size(trD, 1);
n = size(trD, 2);

max_epoch = 2000;
eta0 = 1;
eta1 = 100;
%C = 0.1;
C = 10;

labels = unique(trLb);
k = length(labels);

W = zeros(d, k);

X = trD;
y = trLb;

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
title('Loss after each epoch');
xlabel('Epoch');
ylabel('Loss');

loss_value = plot_objective_values(size(plot_objective_values, 1))


X_val = valD;
y_val = valLb;

k = length(unique(valLb));
n = size(valD, 2);
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
correct_count = 0;
wrong_count = 0;
for i = 1:n
    if y_val(i) == y_pred(i)
        correct_count = correct_count + 1;
    else
        wrong_count = wrong_count + 1;
    end
end

validation_error = wrong_count / (correct_count + wrong_count)


X_val = trD;
y_val = trLb;

k = length(unique(trLb));
n = size(trD, 2);
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
correct_count = 0;
wrong_count = 0;
for i = 1:n
    if y_val(i) == y_pred(i)
        correct_count = correct_count + 1;
    else
        wrong_count = wrong_count + 1;
    end
end
training_error = wrong_count / (correct_count + wrong_count)


norm_value = norm(W) * norm(W)
