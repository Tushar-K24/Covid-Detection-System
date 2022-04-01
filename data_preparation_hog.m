clc;
close all;
input_dir1 = 'D:\MATLAB\Covid-Detection\Dataset\Covid-19\';    %input directory for covid-19
input_dir2 = 'D:\MATLAB\Covid-Detection\Dataset\Healthy\'; %input directory for healthy
input_dir3 = 'D:\MATLAB\Covid-Detection\Dataset\Pneumonia\';   %input directory for pneumonia 

%importing data
img_covid = imageDatastore(input_dir1); %covid image data
img_healthy = imageDatastore(input_dir2); %healthy image data
img_pneumonia = imageDatastore(input_dir3); %pneumonia image data
    
%number of files of said classes
numImagesCovid = numel(img_covid.Files); %number of covid images
numImagesHealthy = numel(img_healthy.Files); %number of healthy images
numImagesPneumonia = numel(img_pneumonia.Files); %number of pneumonia images

%adding labels 
%healthy    : 1
%covid      : 2
%pneumonia  : 3
img_healthy.Labels = repelem(1,numImagesHealthy); 
img_covid.Labels = repelem(2,numImagesCovid);
img_pneumonia.Labels = repelem(3,numImagesPneumonia);

n = numImagesCovid;

%converting images into feature vectors
K = 4; %hyperparameter (will change to get better results)
cell_size = [K,K];
img = imread(img_covid.Files{1});
[hogfv,hogvis] = extractHOGFeatures(img,'CellSize',cell_size);
hogfeaturesize = length(hogfv);
h_X = zeros(n,hogfeaturesize,'single');
h_y = img_healthy.Labels;
cov_X = zeros(n,hogfeaturesize,'single');
cov_y = img_covid.Labels;
p_X = zeros(n,hogfeaturesize,'single');
p_y = img_pneumonia.Labels;
for i=1:n
    h_dir = img_healthy.Files{i};
    cov_dir = img_covid.Files{i};
    p_dir = img_pneumonia.Files{i};
    h_img = imread(h_dir);
    h_img = im2uint8(h_img);
    cov_img = imread(cov_dir);
    cov_img = im2uint8(cov_img);
    p_img = imread(p_dir);
    p_img = im2uint8(p_img);
    h_X(i,:) = extractHOGFeatures(h_img,'CellSize',cell_size);
    cov_X(i,:) = extractHOGFeatures(cov_img,'CellSize',cell_size);
    p_X(i,:) = extractHOGFeatures(p_img,'CellSize',cell_size);
end

%creating training and testing set
train_size = 0.8*n;
X_train = [h_X(1:train_size,:);cov_X(1:train_size,:);p_X(1:train_size,:)];
y_train = [h_y(1:train_size,:);cov_y(1:train_size,:);p_y(1:train_size,:)];

X_test = [h_X(train_size+1:end,:);cov_X(train_size+1:end,:);p_X(train_size+1:end,:)];
y_test = [h_y(train_size+1:end,:);cov_y(train_size+1:end,:);p_y(train_size+1:end,:)];

train_size = size(X_train);
train_size = train_size(1);
test_size = size(X_test);
test_size = test_size(1);

%shuffling the data of training and testing images
idx = randperm(train_size);
X_train = X_train(idx(:),:);
y_train = y_train(idx(:),:);

idx = randperm(test_size);
X_test = X_test(idx(:),:);
y_test = y_test(idx(:),:);
