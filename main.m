clc;
close all;
load('clf.mat');
[filename,path] = uigetfile('*.*', 'Select Input Image');
file = strcat(path,filename);
img = imread(file);
img = im2gray(img);
img = imresize(img,[224 224]);
img = im2uint8(img);
cell_size = [32,32];
lbpfv = extractLBPFeatures(img,'CellSize',cell_size);
pred = predict(clf,lbpfv);

figure
imshow(img)
title([path ': ' pred])

