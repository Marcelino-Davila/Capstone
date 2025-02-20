clear;
load("2024_07_31_aspire_3d_sar_img_hanning.mat", "-mat");

% Convert to magnitude for visualization
img_data = abs(img_hh); % Use HH polarization or another channel

% Select a 2D slice for CFAR (middle Z-plane)
z_index = 100;
slice_data = img_data(:,:,z_index);

% CFAR Parameters
guard_cells = 2;      % Guard cells in each direction
training_cells = 10;  % Training cells in each direction
false_alarm_rate = 1e-4; % Probability of false alarm (Pfa)

% Create CFAR Detector Object
cfar = phased.CFARDetector2D( ...
    'TrainingBandSize', [training_cells training_cells], ...
    'GuardBandSize', [guard_cells guard_cells], ...
    'ThresholdFactor', 'Auto', ...
    'ProbabilityFalseAlarm', false_alarm_rate, ...
    'Method', 'CA'); % Cell-Averaging CFAR

% Define the valid region where CFAR can be applied (avoid edges)
[row_max, col_max] = size(slice_data);
row_min = training_cells + guard_cells + 1;
col_min = training_cells + guard_cells + 1;
row_max = row_max - (training_cells + guard_cells);
col_max = col_max - (training_cells + guard_cells);

% Create indices for CFAR detection (excluding edge regions)
[row_indices, col_indices] = meshgrid(row_min:row_max, col_min:col_max);
cell_indices = [row_indices(:) col_indices(:)]';

% Apply CFAR Detection
detection_map = cfar(slice_data, cell_indices);

% Convert detection map back to image size
cfar_result = zeros(size(slice_data)); % Initialize empty map
for i = 1:length(detection_map)
    r = cell_indices(1, i);
    c = cell_indices(2, i);
    cfar_result(r, c) = detection_map(i);
end


%disp(size(cell_indices)); % Should be [2, N]
%disp(cell_indices(:, 1:10)); % Print first 10 indices

figure;
imagesc(cfar_result);
colorbar;
title('Raw CFAR Detection Map');

asiofjioa = row_min:row_max;


% Plot results
figure;
imagesc(x_img, y_img, 10*log10(slice_data)); % Log scale visualization
hold on;
[x_detect, y_detect] = find(cfar_result); % Get detected points
plot(x_img(y_detect), y_img(x_detect), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5); % Mark detections
hold off;

xlabel('X (meters)');
ylabel('Y (meters)');
title('2D CFAR Object Detection');
colorbar;
colormap(jet);


