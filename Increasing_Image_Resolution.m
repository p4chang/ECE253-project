%% Import Dataset
% import images
sample = imread("testSign.jpg");
sample = rgb2gray(sample);
%create low res version
sampleSmol = imresize(sample,0.0625);

%% Basic Spectral Interpolation
sampleUp = upsampleBy2(sampleSmol);
for i = 1:3
    sampleUp = upsampleBy2(sampleUp);
end

%% Cubic Spline Interpolation
sampleUpSpline = bspline_resize(sampleSmol,2);
for i = 1:3
    sampleUpSpline = bspline_resize(sampleUpSpline,2);
end

%% Bilateral soft-decision interpolation (BSAI)
BSAIparams = [5, 1, 5, 1e-3];
sampleUpBSAI = BSAI(sampleSmol, 2, BSAIparams);
for i = 1:3
    sampleUpBSAI = BSAI(sampleUpBSAI, 2, BSAIparams);
end

%% Plotting Results
figure();
subplot(2,3,1)
imshow(sample);
pixels = size(sample);
title(['\fontsize{18} Orignal ' num2str(pixels(1)) ' by ' num2str(pixels(2)) ' Sample'])
subplot(2,3,2)
imshow(sampleSmol);
pixels = size(sampleSmol);
title(['\fontsize{18} Low Resolution ' num2str(pixels(1)) ' by ' num2str(pixels(2)) ' Sample'])
subplot(2,3,3)
imshow(sampleUp);
title(['\fontsize{18} Spectral Interpolated Reconstruction with MSE ' num2str(mse(sample,sampleUp))])
subplot(2,3,4)
imshow(sampleUpSpline);
title(['\fontsize{18} B-Spline Interpolated Reconstruction with MSE ' num2str(mse(sample,sampleUpSpline))])
subplot(2,3,5:6)
imshow(sampleUpBSAI);
title(['\fontsize{18} BSAI Interpolated Reconstruction with MSE ' num2str(mse(sample,sampleUpBSAI))])
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
function up = bspline_resize(img, scale)
    
    img = double(img);
    
    % cubic B-spline 1D kernel sampled at half-pixel
    x = -2:1/scale:2;
    B = cubicB(x);
    K = B' * B;   % 2-D separable
    
    % compute B-spline coefficients via recursive filtering
    C = bsplineCoef(img);
    
    % convolve with kernel
    up = conv2(C, K, 'same');
        
    up = uint8(round(min(max(up,0),255)));
    %up = histeq(up);
    end
    
    function y = cubicB(x)
    ax = abs(x);
    y = ((ax < 1) .* (4 - 6*ax.^2 + 3*ax.^3) + ...
         (1 <= ax & ax < 2) .* (2 - ax).^3) / 6;
    end
    
    function C = bsplineCoef(f)
    % Unser 1993 recursive implementation
    z = sqrt(3)-2;
    C = f;
    
    % causal pass
    for j=2:size(f,2)
        C(:,j) = C(:,j) + z*C(:,j-1);
    end
    
    % anti-causal
    C(:,end) = z/(1-z^2)*C(:,end);
    for j=size(f,2)-1:-1:1
        C(:,j) = z*(C(:,j+1)-C(:,j));
    end
    
end


% function sampleUp = upsampleBy2BSpline(sample)
% % Upsamples an image by two using B-Splines. Code was derived from sample
% % code from pg 150 of textbook
%     sample = double(sample);
%     % Cubic spline function samples.
%     beta=[1/6 2/3 1/6];
%     beta2=beta'*beta;
%     % Cubic spline function.
%     t1=[1:2]/3;
%     St1=2/3-t1.^2+t1.^3/2;
%     t2=[3:5]/3;
%     St2=(2-t2).^3/6;
%     % 2-D cubic spline.
%     S=[St1 St2];
%     S=[fliplr(S) 2/3 S];
%     S2=S'*S;
%     % Coefficients C by econvolution
%     dim = 2*size(sample);
%     C = real(ifft2(fft2(sample,dim(1),dim(2))./fft2(beta2,dim(1),dim(2))));
%     spec = conv2(C,S2,'same');
%     sampleUp = uint8(round(min(max(spec,0),255)));
%     
% end

function HR = BSAI(LR, scale, params)
    % BSAI  Bilateral Soft-decision Interpolation
    %   HR = BSAI(LR, scale) upsamples LR by integer factor 'scale' using the
    %   Bilateral Soft-decision Interpolation approach (practical implementation).
    %
    %   HR = BSAI(LR, scale, [A, B, C, D]) allows options:
    %     A   = spatial sigma for bilateral (default: 1.0)
    %     B   = range (intensity) sigma for bilateral (default: 15)
    %     C   = weight t in the paper (default: 0.8)
    %     D   = stabilization constant added to A_k (default: 1e-3)

    % convert to double for processing
    I = double(LR);
    [Lh, Lw, ~] = size(I);
    % output HR size (simple nearest-integer upscale)
    Hh = Lh * scale;
    Hw = Lw * scale;
    % Prepare HR result
    HR = zeros(Hh, Hw);
    % Use MATLAB's imresize to get initial guess (keeps edge structure)
    initHR = imresize(I, [Hh Hw], 'bilinear');
    % For each HR pixel: find four surrounding LR samples, compute A_k, U_k, Xk_neigh, then final value
    lambda = params(3);
    sigma_s = params(1);
    sigma_r = params(2);
    epsU = params(4);
    % small helper to fetch LR pixel safely
    get_LR = @(r,c) I(max(1,min(Lh,r)), max(1,min(Lw,c)));

    % iterate over every pixel
    cnt = 0;
    for iH = 1:Hh
        for jH = 1:Hw
            cnt = cnt + 1;
            % Compute continuous LR coordinates (1-based)
            y = (iH-1)/scale + 1;
            x = (jH-1)/scale + 1;
            % If this HR pixel maps exactly to an LR pixel (integer coords), copy directly
            if abs(y - round(y)) < 1e-12 && abs(x - round(x)) < 1e-12
                yi = round(y); xi = round(x);
                HR(iH,jH) = get_LR(yi, xi);
                continue;
            end
            % corner indices (clamped)
            y0 = floor(y); y1 = ceil(y);
            x0 = floor(x); x1 = ceil(x);
            y0 = max(1, min(Lh, y0));
            y1 = max(1, min(Lh, y1));
            x0 = max(1, min(Lw, x0));
            x1 = max(1, min(Lw, x1));
            % four corner LR samples X_k
            Xc = [ get_LR(y0,x0);  % top-left
                get_LR(y0,x1);  % top-right
                get_LR(y1,x0);  % bottom-left
                get_LR(y1,x1)]; % bottom-right
            % spatial distances from HR pos to LR sample positions (in LR pixel units)
            pts_lr = [y0, x0; y0, x1; y1, x0; y1, x1];
            dy = pts_lr(:,1) - y;
            dx = pts_lr(:,2) - x;
            dist2 = dx.^2 + dy.^2;
            % initial center estimate: use bilinear initialization
            centerInit = initHR(iH,jH);
            % compute photometric difference
            diff_r = (Xc - centerInit).^2;
            % compute bilateral weights A_k
            Ak = exp(-dist2/(2*sigma_s^2)) .* exp(-diff_r/(2*sigma_r^2));
            % normalize A_k
            sumAk = sum(Ak);
            if sumAk > 0
                Ak = Ak / sumAk;
            else
                Ak = ones(4,1)/4;
            end
            % U_k stabilization: paper says "added with a constant for stabilization"
            Uk = Ak + epsU;
            % X_{k,i} approx: mean of 4-neighbors of each LR corner sample (in LR grid)
            Xk_neigh = zeros(4,1);
            for k = 1:4
                yy = pts_lr(k,1);
                xx = pts_lr(k,2);
                % get 4-neighbors in LR (N,S,E,W) clipped at image boundaries
                neighbors = [ get_LR(yy-1, xx), get_LR(yy+1, xx), ...
                    get_LR(yy, xx-1), get_LR(yy, xx+1) ];
                % if neighbor indices outside bounds, get_LR clamps them (replicate)
                Xk_neigh(k) = mean(neighbors);
            end

            % Now implement the closed-form-like expression (interpreting eqn 2.12)
            % numerator = t * sum_k A_k X_k + sum_k [ U_k * A_k * Xk_neigh ]
            % denominator = t * sum_k A_k + sum_k [ U_k * A_k ]
            numer = lambda * sum(Ak .* Xc) + sum( (Uk .* Ak) .* Xk_neigh );
            denom = lambda * sum(Ak) + sum( (Uk .* Ak) );
            if denom <= 0
                outv = centerInit; % fallback
            else
                outv = numer / denom;
            end
            HR(iH,jH) = outv;

        end
    end
    
    %convert back to proper format
    HR = uint8(round(min(max(HR,0),255)));

end

