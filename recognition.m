function classId = recognition(img_path,train_mean,U,rank,xi)
N = 5;
% Defining thresholds, these values were defined by trial and error
epsilon_0 = 50; % Maximum allowable distance from any known face in the training set S
epsilon_1 = 15; % Maximum allowable distance from face space

epsilons = zeros(N, 1);
test_image = readImage(img_path);
test_image = test_image(:) - train_mean; % Normalizing test image
x = U(:, 1:rank)' * test_image; % Calculating coordinate vector x of test image
epsilon_f = ((test_image - U(:, 1:rank) * x)' * (test_image - U(:, 1:rank) * x)) ^ 0.5;
% Checks if it is in face space
if epsilon_f < epsilon_1
    % Computing distance epsilon_i to the face space
    for i = 1:N
        epsilons(i, 1) = (xi(:, i) - x)' * (xi(:, i) - x);
    end
    [val classId] = min(epsilons(:, 1));
    if val < epsilon_0
        disp(sprintf('The face belongs to %d', classId));
    else
        disp('Unknown face');
    end
else
    disp('Input image is not a face');
end

