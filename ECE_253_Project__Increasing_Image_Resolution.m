%% Import Dataset
% import images
sample = imread("testIm.png");
sample = rgb2gray(sample);
%create low res version
sampleSmol = imresize(sample,0.125);


%% Basic Spectral Interpolation
sampleUp = upsampleBy2(sampleSmol);
for i = 1:2
    sampleUp = upsampleBy2(sampleUp);
end


%% Cubic Spline Interpolation
sampleUpSpline = upsampleBy2BSpline(sampleSmol);
for i = 1:2
    sampleUpSpline = upsampleBy2BSpline(sampleUpSpline);
end
    

%% Lanczos interpolation
a = 3;
%Lanczos interpolation a=2:
ta = [-a:1/3:a];
Ha = sin(pi*ta)./(pi*ta).*sin(pi*ta/a)./(pi*ta/a);
Ha(3*a+1) = 1;
sampleUpLanczos = conv2(sampleSmol,Ha(2:(6*a))'*Ha(2:(6*a)),"same");

t3=[-3:1/3:3];H3=sin(pi*t3)./(pi*t3).*sin(pi*t3/3)./(pi*t3/3);H3(10)=1;%1-D Lanczos (a=3).
Y5=conv2(sampleSmol,H3(2:18)'*H3(2:18));%Lanczos 2-D interpolation.

%% Bilateral soft-decision interpolation (BSAI)
% "bilateral filter weights Ak to replace the least
% squares parameters and adopts the following cost function
% (modified from SAl and RSAl) for estimating a HR pixel at
% one time (instead of a block of HR pixels at one time in the
% SAl and RSAl"
A = zeros(4,1); 
% "The weight Uk is
% defmed by the bilateral filter weights (since the continuity
% property of edge depends on the edge orientation, which is
% exploited by the bilateral filter weight) added with a constant
% for stabilization"
U = zeros(4,1);
% calculate numerator from eqs 2.12 interpolation paper
newPxl = 0;
X = zeros(4,1); %fix this with correct definition from fiure 3 page 4 summary interpolation paper
for k = 1:4
    %calculate inner sum with A_i and X_ki
    Xout = zeros(4,1); %fix this with correct definition from fiure 3 page 4 summary interpolation paper
    innerSum = 0;
    for i = 1:4
        if i~=4-k % might need correction
            innerSum = innerSum + A(i)*Xout(i); 
        end
    end
    %add to outer sum
    newPxl = newPxl + A(k)*X(k) + U(k)*A(5-k)*(X(k)-innerSum);
end
newPxl = 1 + U(1)*(A(4)^2) + U(2)*(A(3)^2) + U(3)*(A(2)^2) + U(4)*(A(1)^2);



%% Plotting Results
figure();
subplot(2,2,1)
imshow(sample);
title('\fontsize{20} Orignal 512x512 Sample')
subplot(2,2,2)
imshow(sampleSmol);
title('\fontsize{20} Low-Resolution 64x64 Sample')
subplot(2,2,3)
imshow(sampleUp);
title(['\fontsize{20} Spectral Interpolated Reconstruction with MSE ' num2str(mse(sample,sampleUp))])
subplot(2,2,4)
imshow(sampleUp);
title(['\fontsize{20} B-Spline Interpolated Reconstruction with MSE ' num2str(mse(sample,sampleUpSpline))])
sgtitle('\fontsize{24} Comparison of Interpolation Techniques for Increasing Image Resolution')


%% Custom Functions
function sampleUp = upsampleBy2(sample)
% Upsamples an image by two using simple spectral interpolation. Assumes
% uint8 grayscale input

    % capture spectrum of original sample
    sampleF = fft2(sample);

    %record sample size and calculate size of new image
    [Mo, No] = size(sample);
    Mu = 2*Mo;
    Nu = 2*No;    

    % zero pad spectrum appropriately using rules in pgs 137-139 of book
    if mod(No,2) ~= 0
        sampleUpF = [sampleF(:,1:((No+1)/2)), zeros(Mo,Nu-No), sampleF(:,((No+3)/2):end)];
    else
        sampleUpF = [sampleF(:,1:No/2-1), (1/2)*sampleF(:,No/2) , zeros(Mo,Nu-No-1), (1/2)*sampleF(:,No/2), sampleF(:,No/2+1:end)];
    end
    if mod(Mo,2) ~= 0
        sampleUpF = [sampleUpF(1:((Mo+1)/2),:); zeros(Mu-Mo,Nu); sampleUpF(((Mo+3)/2):end,:)];
    else
        sampleUpF = [sampleUpF(1:Mo/2-1,:); (1/2)*sampleUpF(Mo/2,:); zeros(Mu-Mo-1,Nu); (1/2)*sampleUpF(Mo/2,:); sampleUpF(Mo/2+1:end,:)];
    end

    %scale for lost power
    sampleUpF = ((Mu*Nu) / (Mo*No)) * sampleUpF;

    % generate image with IFFT and convert to orignal variable type
    sampleUp = uint8(real(ifft2(sampleUpF)));
    
end
function sampleUp = upsampleBy2BSpline(sample)
% Upsamples an image by two using B-Splines. Code was derived from sample
% code from pg 150 of textbook

    % Cubic spline function samples.
    beta=[1/6 2/3 1/6];
    beta2=beta'*beta;
    % Cubic spline function.
    t1=[1:2]/3;
    St1=2/3-t1.^2+t1.^3/2;
    t2=[3:5]/3;
    St2=(2-t2).^3/6;
    % 2-D cubic spline.
    S=[St1 St2];
    S=[fliplr(S) 2/3 S];
    S2=S'*S;
    % Coefficients C by econvolution
    dim = 2*size(sample);
    C = real(ifft2(fft2(sample,dim(1),dim(2))./fft2(beta2,dim(1),dim(2))));
    sampleUp = uint8(conv2(C,S2,'same'));
 
end

