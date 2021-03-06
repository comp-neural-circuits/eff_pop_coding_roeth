clear
load('AMT_data_fitting.mat')

outlier_idx = 122;  %outlier, 275-09

r(outlier_idx) = [];
R(outlier_idx) = [];
sigma(outlier_idx) = [];
theta(outlier_idx) = [];

norm_r = r./R;

F_norm_r = zeros(size(norm_r));
F_sigma = zeros(size(sigma));

for i = 1:length(norm_r)
        F_norm_r(i) = mean(norm_r <= norm_r(i));
        F_sigma(i) = mean(sigma <= sigma(i));
end

idx_type1 = (F_norm_r >= F_sigma);
idx_type2 = (F_norm_r < F_sigma);
%{
figure
set(gcf, 'color', [1 1 1])
scatter(F_norm_r(idx_type1), F_sigma(idx_type1), 'filled');
hold on
scatter(F_norm_r(idx_type2), F_sigma(idx_type2), 'filled');
plot([0 1], [0 1], 'color', 'k', 'LineStyle', '--');
%}
avg_sigma_1 = mean(sigma(idx_type1));
avg_sigma_2 = mean(sigma(idx_type2));
avg_spon_fire_1 = mean(norm_r(idx_type1));
avg_spon_fire_2 = mean(norm_r(idx_type2));
avg_theta_1 = mean(theta(idx_type1));
avg_theta_2 = mean(theta(idx_type2));

%transform into model
mu_s = 30;    %30dB
sigma_s = 12.5;     %12.5dB

avg_sigma_1_simul = avg_sigma_1/sigma_s;
avg_sigma_2_simul = avg_sigma_2/sigma_s;
avg_theta_1_simul = (avg_theta_1 - mu_s)/sigma_s;
avg_theta_2_simul = (avg_theta_2 - mu_s)/sigma_s;

clear i
save('AMT_data_statistics_2types.mat')

mdl1 = fitlm(norm_r, sigma);
mdl2 = fitlm(sigma, theta);
