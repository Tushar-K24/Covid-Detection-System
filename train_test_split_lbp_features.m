clc;
close all;
input_dir1 = 'D:\MATLAB\Covid-Detection\Dataset\Covid-19\';    %input directory for covid-19
input_dir2 = 'D:\MATLAB\Covid-Detection\Dataset\Healthy\'; %input directory for healthy
input_dir3 = 'D:\MATLAB\Covid-Detection\Dataset\Pneumonia\';   %input directory for pneumonia 

%importing data
img_covid = imageDatastore(input_dir1,'LabelSource','foldernames'); %covid image data
img_healthy = imageDatastore(input_dir2,'LabelSource','foldernames'); %healthy image data
img_pneumonia = imageDatastore(input_dir3,'LabelSource','foldernames'); %pneumonia image data

%number of files of said classes
numImagesCovid = numel(img_covid.Files); %number of covid images
numImagesHealthy = numel(img_healthy.Files); %number of healthy images
numImagesPneumonia = numel(img_pneumonia.Files); %number of pneumonia images

n = numImagesCovid;

%converting images into feature vectors
K = 32; %hyperparameter (will change to get better results)
cell_size = [K,K];
img = imread(img_covid.Files{1});
lbpfv = extractLBPFeatures(img,'Cellsize',cell_size);
lbpfeaturesize = length(lbpfv);
h_X = zeros(n,lbpfeaturesize,'single');
h_y = img_healthy.Labels;
cov_X = zeros(n,lbpfeaturesize,'single');
cov_y = img_covid.Labels;
p_X = zeros(n,lbpfeaturesize,'single');
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
    h_X(i,:) = extractLBPFeatures(h_img,'Cellsize',cell_size);
    cov_X(i,:) = extractLBPFeatures(cov_img,'Cellsize',cell_size);
    p_X(i,:) = extractLBPFeatures(p_img,'Cellsize',cell_size);
end

%combining data
img_dat = [h_X;cov_X;p_X];
img_labels = [h_y;cov_y;p_y];
n = length(img_labels);

%creating training and testing set
rng(1);
train_size = 0.8*n;
test_size = 0.2*n;
idx = randperm(n);
X_train = img_dat(idx(1:train_size),:);
y_train = img_labels(idx(1:train_size),:);

X_test = img_dat(idx(train_size+1:end),:);
y_test = img_labels(idx(train_size+1:end),:);

%save training and testing set
save('X_train_lbp.mat','X_train');
save('y_train_lbp.mat','y_train');

save('X_test_lbp.mat','X_test');
save('y_test_lbp.mat','y_test');