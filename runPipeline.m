%% Run Pipeline

% Suffix for every file must be in this array
suffixes = [5618, 5619, 5620];

% Loop through every data point
for n = 1:length(suffixes)

    % Import data
    inputFileName = ['Inputs\IMG_', num2str(suffixes(n)),  '.jpg'];
    data = imread(fileName);

    % Perform image enhancements
    %   category for filename/posterity
    enhancementCategory = 'gray';
    %   run desired en
    enhanced = rgb2gray(data);

    % Output to file
    outputFileName = ['Outputs\' enhancementCategory, '_IMG_', num2str(suffixes(n)), '.jpg']; 
    imwrite(enhanced,outputFileName);
end