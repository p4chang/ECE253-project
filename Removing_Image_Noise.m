%% Import Dataset
% import images
sample = imread("testSign.jpg");
sample = rgb2gray(sample);

%% Median filtering
sampleMed = medfilt2(sample);

%% MA filtering
sampleMA = uint8(filter2((1/64)*ones(8,8), double(sample)));

%% Wavelet denoising
sampleWave = wdenoise2(sample);

%% Display results
figure();
subplot(2,2,1)
imshow(sample);
title('\fontsize{18} 1/32 Scale Sample')
subplot(2,2,2)
imshow(sampleUp);
title('\fontsize{18} Spectral Interpolated Reconstruction with MSE ')
subplot(2,2,3)
imshow(sampleUpSpline);
title('\fontsize{18} B-Spline Interpolated Reconstruction with MSE ')
subplot(2,2,4)
imshow(sampleUpBSAI);
title('\fontsize{18} BSAI Interpolated Reconstruction with MSE ')
sgtitle('\fontsize{24} Comparison of Interpolation Techniques for Increasing Image Resolution')