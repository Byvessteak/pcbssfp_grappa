%% Create plots
clc
close all
clear

%% Add missing folders
addpath(genpath('python_reconstruction'))
addpath(genpath('dependencies'))
addpath(genpath('datasets'))
addpath(genpath('results'))


%% Load mat data

image = load("results/grappa2D_2_60.mat"); % accelerated image

original = load("datasets/profiles_slice.mat"); % fully sampled image




img = image.matrix_name;

img = squeeze(img); % choose correct slice
img = flip(img,1);
img = flip(img,2);


orig = original.profiles_slice;


%% plot
tiledlayout(2,2)
nexttile
imagesc(real(orig(:,:,1)))
title('phase-cycle 1')

nexttile
imagesc(real(orig(:,:,10)))
title('phase-cycle 10')

nexttile
imagesc(real(img(:,:,1)))
title('accelerated phase-cycle 1')

nexttile
imagesc(real(img(:,:,10)))
title('accelerated phase-cycle 10')


