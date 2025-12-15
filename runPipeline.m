%% Run Pipeline

% Define names of single enhancements/pipelines for looping through all of them 
enhancementCategories = ['small','spectralUpsample','bsplineUpsample','bsaiUpsample','histeq',...
    'gamma','unsharp','deblur','median','MA','wavelet','chain1','chain2','chain3','chain4','chain5', 'gray'];

% Loop through every category
for j = 1:length(enhancementCategories) 
    % Loop through every data point
    for n = 1:346 
        
        % Establish filename and import data
        inputFileName = ['Inputs\finalDataset\finalDataset\', num2str(n),  '.jpg']; %  --> online dataset
        %inputFileName = 'Inputs\IMG_5660.jpg'; % --> temp for reaching around
        enhanced = im2double(imread(inputFileName));
        enhanced = rgb2gray(enhanced);
    
        %  Category for filename/posterity
        %enhancementCategory = '__';k
        dowsampling = 4;
        
        if j==1
            enhancementCategory = 'small';
            enhanced = enhanced (1:end-16,:); 
            enhanced = imresize(enhanced,1/dowsampling);
        elseif j==2
            %spectral 
            enhancementCategory = 'spectralUpsampling';
            enhanced = imresize(enhanced,1/dowsampling);
            enhanced = spectralUpsample(enhanced,dowsampling);
        elseif j==3
            %bspline 
            enhancementCategory = 'bsplineUpsampling';
            enhanced = imresize(enhanced,1/dowsampling);
            enhanced = bsplineUpsample(enhanced,dowsampling);
        elseif j==4
            %bsai 
            enhancementCategory = 'bsaiUpsampling';
            enhanced = imresize(enhanced,1/dowsampling);
            enhanced = bsaiUpsample(enhanced,dowsampling,zeros(1,5));
        elseif j==5
            %histeq 
            enhancementCategory = 'histeq';
            enhanced = histeq(enhanced);
        elseif j==6
            %gamma 
            enhancementCategory = 'gamma';
            enhanced = gammaCorrect(enhanced);
        elseif j==7
            %unsharp
            enhancementCategory = 'unsharp';
            enhanced = imagesharp(enhanced);
        elseif j==8
            %deblur
            enhancementCategory = 'deblur';
            %best_psf = deblur_psf(enhanced);
            %enhanced = deconvwnr(enhanced, best_psf, 0.005);
        elseif j==9
            %median
            enhancementCategory = 'median';
            enhanced = medfilt2(enhanced,[5 5]);
        elseif j==10
            %MA
            enhancementCategory = 'MA';
            enhanced = filter2((1/25)*ones(5,5), enhanced);
        elseif j==11
            %wavelet
            enhancementCategory = 'wavelet';
            enhanced = wdenoise2(enhanced);
        elseif j==12
            %chain1
            enhancementCategory = 'chain1';
            enhanced = chain1(enhanced);
        elseif j==13
            %chain2
            enhancementCategory = 'chain2';
            enhanced = chain2(enhanced);
        elseif j==14
            %chain3
            enhancementCategory = 'chain3';
            enhanced = chain3(enhanced);
        elseif j==15
            %chain4
            enhancementCategory = 'chain4';
            enhanced = chain4(enhanced);
        elseif j==16
            %chain5
            enhancementCategory = 'chain5';
            enhanced = chain5(enhanced);
        elseif j==17
            %chain5
            enhancementCategory = 'grey';
        end
    
        % RUN DESIRED ENHANCEMENTS, EDIT THIS FOR DIFFERENT PIPELINES:
    
            %autorun chains
    %     outputFileName1 = ['Outputs\chain1_IMG_', num2str(suffixes(n)), '.jpg'];
    %     enhanced1 = chain1(enhanced);
    %     imwrite(enhanced1,outputFileName1);
    %     outputFileName1 = ['Outputs\chain2_IMG_', num2str(suffixes(n)), '.jpg'];
    %     enhanced1 = chain3(enhanced);
    %     imwrite(enhanced1,outputFileName1);
    %     outputFileName1 = ['Outputs\chain3_IMG_', num2str(suffixes(n)), '.jpg'];
    %     enhanced1 = chain3(enhanced);
    %     imwrite(enhanced1,outputFileName1);
    
    
            % Run Chains
        %enhanced = chain1(enhanced);
        %enhanced = chain2(enhanced);
        %enhanced = chain3(enhanced);
    
            % Convert to grayscale and trim 
        %enhanced = rgb2gray(enhanced);
        %enhanced = enhanced (1:end-16,:); 
    
            % Decrease resolution
        %enhanced = imresize(enhanced,1/32);
    
            % Increase resolution - Spectral, B-spline, and BSAI interpolation
        %enhanced = spectralUpsample(enhanced,32);
        %enhanced = bsplineUpsample(enhanced,32);
        %enhanced = bsaiUpsample(enhanced,32,zeros(1,5));
            
            % Increase brightness - Histogram eq, gamma correction
        %enhanced = histeq(enhanced);
        %enhanced = gammaCorrect(enhanced);
    
            % Sharpen image - Unsharp mask and deblurring
        %enhanced = imagesharp(enhanced);
        %best_psf = deblur_psf(enhanced);
        %enhanced = deconvwnr(enhanced, best_psf, 0.005);
    
            % Denoising image - Median filter, MA filter, and Wavelet denoising
        %enhanced = medfilt2(enhanced,[5 5]);
        %enhanced = filter2((1/25)*ones(5,5), double(enhanced));
        %enhanced = wdenoise2(double(enhanced));
    
        % Establish filepath/name and output to file
        outputFileName = ['Outputs\finalPictureSet\' enhancementCategory, '_', num2str(n), '.jpg']; % edit this for output filepath
        %outputFileName = ['Outputs\' enhancementCategory,'_IMG_5668.jpg'];%manual output 
        imwrite(enhanced,outputFileName);
    
    end

end
%% Testing execution speed
%import data and convert to gray
enhanced = im2double(imread('Inputs\finalDataset\finalDataset\340.jpg'));
enhanced = rgb2gray(enhanced);

f = @() chain1(enhanced);
resChain1 = timeit(f);

f = @() chain2(enhanced);
resChain2 = timeit(f);

f = @() chain3(enhanced);
resChain3 = timeit(f);

f = @() chain4(enhanced);
resChain4 = timeit(f);

f = @() chain5(enhanced);
resChain5 = timeit(f);

f = @() gammaCorrect(enhanced);
resGamma= timeit(f);

f = @() deconvwnr(enhanced, deblur_psf(enhanced), 0.005);
resDeblur= timeit(f);

f = @() imagesharp(enhanced);
resUnsharp= timeit(f);

f = @() histeq(enhanced);
resHist = timeit(f);

f = @() medfilt2(enhanced,[2 2]);
resMedian = timeit(f);

f = @() filter2((1/25)*ones(2,2), enhanced);
resMA = timeit(f);

f = @() wdenoise2(enhanced);
resWavelet = timeit(f);

f = @() imresize(enhanced,1/4);
resDown = timeit(f);

enhanced = imresize(enhanced,1/4); %downsample

f = @() bsaiUpsample(enhanced,4,zeros(1,5));
resBSAI = timeit(f);

f = @() bsplineUpsample(enhanced,4);
resBspline = timeit(f);

f = @() spectralUpsample(enhanced,4);
resSpectral = timeit(f);



%% Enhancement Functions for Pipeline 

function enhanced = chain1(data)
    %downres to represent poor camera
    enhanced = imresize(data,1/4);
    %upres with bspline
    enhanced = bsplineUpsample(enhanced,4);
    %increase sharpness
    enhanced = imagesharp(enhanced);
    %contrast adjustment with histeq
    enhanced = histeq(enhanced);
    %denoise with median
    enhanced = medfilt2(enhanced,[5 5]);
end

function enhanced = chain2(data)
    %downres to represent poor camera
    enhanced = imresize(data,1/4);
    %contrast adjustment with histeq
    enhanced = histeq(enhanced);
    %upres with bspline
    enhanced = bsplineUpsample(enhanced,4);
    %increase sharpness
    enhanced = imagesharp(enhanced);
    %denoise with median
    enhanced = medfilt2(enhanced,[5 5]);
end

function enhanced = chain3(data)
    %downres to represent poor camera
    enhanced = imresize(data,1/4);
    %increase sharpness
    enhanced = imagesharp(enhanced);
    %contrast adjustment with histeq
    enhanced = histeq(enhanced);
    %denoise with median
    enhanced = medfilt2(enhanced,[2 2]);
    %upres with bspline
    enhanced = bsplineUpsample(enhanced,4);    
end

function enhanced = chain4(data)
    %downres to represent poor camera
    enhanced = imresize(data,1/4);
    %upres with bspline
    enhanced = bsplineUpsample(enhanced,4);
    %contrast adjustment with histeq
    enhanced = histeq(enhanced);
    %denoise with median
    enhanced = medfilt2(enhanced,[5 5]);
end

function enhanced = chain5(data)
    %downres to represent poor camera
    enhanced = imresize(data,1/4);
    %contrast adjustment with histeq
    enhanced = histeq(enhanced);
    %denoise with median
    enhanced = medfilt2(enhanced,[2, 2]);  
end


function sampleUp = spectralUpsample(sample, scale)
% Upsamples an image by two using simple spectral interpolation. Assumes
% uint8 grayscale input
    currScale = 1;
    while currScale ~= scale
        %convert to double and capture spectrum of original signal
        sample = double(sample);
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
        sampleUp = real(ifft2(sampleUpF));

        %increment j and set up next upscale
        currScale = currScale * 2;
        sample = sampleUp;
    end

end


function sampleUp = bsplineUpsample(img, s)
% Upsamples an image by two using cubic b-spline interpolation. Assumes
% uint8 grayscale input
    
    % capture size
    [N,M] = size(img);
    %zero-insert
    sampleUp = zeros(s*N, s*M);
    sampleUp(1:s:end, 1:s:end) = img;  
    %apply cubic b-spline kernel
    x = -2:1/s:2;
    ax = abs(x);
    B = ((ax < 1) .* (4 - 6*ax.^2 + 3*ax.^3) + ...
         (1 <= ax & ax < 2) .* (2 - ax).^3) / 6;
    K = B' * B;
    sampleUp = conv2(sampleUp, K, 'same');

end


function sampleUp = bsaiUpsample(I, scale, params)
% BSAI  Bilateral Soft-decision Interpolation
%   HR = BSAI(LR, scale) upsamples LR by integer factor 'scale' using the
%   Bilateral Soft-decision Interpolation approach (practical implementation).
%
%   HR = BSAI(LR, scale, [A, B, C, D]) allows options:
%     A   = spatial sigma for bilateral (default: 1.0)
%     B   = range (intensity) sigma for bilateral (default: 15)
%     C   = weight t in the paper (default: 0.8)
%     D   = stabilization constant added to A_k (default: 1e-3)

    %if input parameters are all zeros, default parameters are used
    if params == zeros(1,5)
        params = [1, 15, 0.8, 1e-3];
    end
    [Lh, Lw, ~] = size(I);
    % output HR size (simple nearest-integer upscale)
    Hh = Lh * scale;
    Hw = Lw * scale;
    % Prepare HR result
    sampleUp = zeros(Hh, Hw);
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
                sampleUp(iH,jH) = get_LR(yi, xi);
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
            sampleUp(iH,jH) = outv;
    
        end
    end

end


function res = gammaCorrect(I)
    gamma = 0.4:0.1:2.5;
    gamma = [gamma, 3:1:20];
    bestG = 1;
    bestS = inf;
    for g = gamma
        test = (I.^g);
        score = abs(220-mean(test));
        if bestS > score 
            bestS  = score;
            bestG = g;
        end
    end
    res = (I.^bestG);
end


function score = sharpness_metric(I)
    Gx = imfilter(I, fspecial('sobel')');
    Gy = imfilter(I, fspecial('sobel'));
    score = sum(Gx(:).^2 + Gy(:).^2);
end




function res = deblur_psf(img)
    %deblur
    psf_len = [5 9 13 17 21 25 29 33 41 51];
    psf_ang = -90:5:90; 
    bestS = -inf;
    res = [];
    small = imresize(img, 1/3);
    for L = psf_len
        for A = psf_ang
            psf = fspecial('motion', L, A);
            psf = psf / sum(psf(:));  
            if length(psf) > min(size(img)/3)
                psf = psf(end-floor(min(size(img)/3)):end);
            end     
            if psf == zeros(size(psf))
                psf(1,1) = 1;
            end
            J = deconvwnr(small, psf, 0.005);
            score = sharpness_metric(J);
            if score > bestS
                bestS = score;
                res = psf;
            end
        end
    end
end


function res = imagesharp(I)
    %upsharp masking 
    amount = 0.5:0.5:5;
    radius = 0.5:0.5:4;
    bestScore = -Inf;
    res = [];
    denoised = imgaussfilt(I, 0.5);
    for a = amount
        for r = radius
            mask = denoised - imgaussfilt(denoised, r);
            candidate = denoised + a * mask;
            candidate = max(min(candidate, 1), 0);
            score = sharpness_metric(candidate);
            if score > bestScore
                bestScore = score;
                res = candidate;
            end
        end
    end
end
