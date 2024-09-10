%% Real data undersampling
clc
close all
clear

%% Add missing folders
addpath(genpath('pygrappa'))
addpath(genpath('dependencies'))
addpath(genpath('datasets'))

%% Load kspace data
data = load("datasets/profiles207to246_allcoils.mat");
data = data.slices; % TODO: change name to correct data name
data_msg = 'Data loading DONE \n';
fprintf(data_msg)

%% Variables to define
undersampling_y = 2; % undersampling rate in x-direction, default = 2
undersampling_z = 3; % undersampling rate in y-direction,default = 2
calib_y = 42; % calibration width in x-direction,default = 84
calib_z = 36; % calibration width in y-direction,default = 72
save_data = true; % want to save data?


%% Create undersampling mask
size_im = size(data);
n_slice = size_im(1);
size_y = size_im(2);
size_z = size_im(3);
mask = zeros(size_y,size_z);
mask_c = zeros(size_y,size_z);

% undersampling in x- and y-direction
mask_under_y = mask; mask_under_y(undersampling_y:undersampling_y:end,:) = 1;
mask_under_z = mask; mask_under_z(:,undersampling_z:undersampling_z:end) = 1;

mask_under = mask_under_y .* mask_under_z;

% calibration-part of the mask
y_high = size_y/2+(calib_y/2);
y_low = y_high-calib_y+1;
z_high = size_z/2+(calib_z/2);
z_low = z_high-calib_z+1;

calib_mask = mask_c; calib_mask(y_low:y_high,z_low:z_high) = 1;

% Combine mask
undersampling = mask_under + calib_mask;
undersampling = undersampling > 0.5; % binarize mask

% calculate and display undersampling rate
n_under = sum(undersampling(:) == 1);
under = (size_y*size_z)/n_under;
formatSpec = 'Undersampling rate: %.3f \n';
fprintf(formatSpec,under)

% Display undersampling mask
figure;
imagesc(undersampling);
title 'undersampling mask';
axis off;

mask_shape = [size_im(1), size(mask)];


%% Use data for undersampling

for slice = 1:n_slice
    data_slice = data(slice,:,:,:,:);
    data_slice = squeeze(data_slice);
    kspace = fft2c_mri(data_slice);


    kspace_under = mask_under.*kspace;

   
    coils_for_use = [1:size_im(5)]; % all coils

    calib = kspace(y_low:y_high, z_low:z_high,:,coils_for_use);
    kspace_under = kspace_under(:, :, :, coils_for_use);
    %dimensions = size(kspace_under);



    % save slice per slice
    if save_data == true
        filename_calib = sprintf('pygrappa/temp/calib_%d.mat',slice);
        filename_under = sprintf('pygrappa/temp/kspace_under_%d.mat',slice);
        delete(filename_calib);
        delete(filename_under);
        save(filename_calib, 'calib');
        save(filename_under, 'kspace_under');
    end
    data_msg = 'Handeled slice nr %d\n';
    fprintf(data_msg, slice)
end

%% Export data for use in python
if save_data == true
    save('pygrappa/mask.mat', 'undersampling')
    fprintf('saved data \n')
end
