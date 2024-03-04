
% Specify the directory containing image files
directory = 'C:\Users\love\Pictures\mixed_rotation';

% Retrieve the list of PNG files in the directory
files = dir(fullfile(directory, '*.png'));

% Initialize variables to store calculated features
m2 = double.empty(0, 0);
m3ent1 = double.empty(0, 0);

% Iterate over the first 50 image files
for k = 1:min(50, numel(files))
    % Construct the full path to the image file
    filename = fullfile(directory, files(k).name);
    
    % Read the image and convert it to grayscale
    input_im = imread(filename);
    input_im = rgb2gray(input_im);
    
    % Resize the image to a fixed size of 50x50 pixels
    input_im = imresize(input_im, [50, 50]);
    
    % Convert the colormap to grayscale
    colormap(gray);
    
    % Convert the image to double precision
    input_im = double(input_im);
    
    % Calculate the third central moment and entropy of the image
    m = moment(input_im, 3);
    m3ent = entropy(m);
    
    % Store the calculated features
    m3ent1 = [m3ent1; m3ent];
    m2 = [m2; m];
end

% Save the calculated features to a MAT file
save('train_dataset.mat', 'm2', 'm3ent1');

% Load the saved features
load train_dataset.mat;

% Initialize cell arrays to store group labels
group = cell(50, 1);
strArray = java_array('java.lang.String', 50);

% Assign labels based on calculated entropy values
for i = 1:50
    if m3ent1(i) == 0
        strArray(i) = "Car";
    else
        strArray(i) = "Not Car";
    end
end

% Save the group labels to a MAT file
group = cellstr(strArray);
save('group.mat', 'group');

% Load the saved group labels
load group.mat;

% Load the fisheriris dataset (not used in this code)
load fisheriris;

% Extract features from the data matrix
xdata = m2(:, 20:21);
group1 = group;

% Train a support vector machine (SVM) classifier
svmStruct = svmtrain(xdata, group1, 'showplot', true);
