clc;
close all;
%input directory
input_dir1 = 'D:\MATLAB\Covid-Detection\Dataset\Covid-19\';
input_dir2 = 'D:\MATLAB\Covid-Detection\Dataset\Healthy\';
input_dir3 = 'D:\MATLAB\Covid-Detection\Dataset\Pneumonia\';
 
%output directory
output_dir1 = 'D:\MATLAB\Covid-Detection\B_and_W\Covid-19\';
output_dir2 = 'D:\MATLAB\Covid-Detection\B_and_W\Healthy\';
output_dir3 = 'D:\MATLAB\Covid-Detection\B_and_W\Pneumonia\'; 

input_images_covid = fullfile(input_dir1, '*');
input_images_healthy = fullfile(input_dir2, '*');
input_images_pneumonia = fullfile(input_dir3, '*');

%list of images present in the directory
src_files_covid = dir(input_images_covid); 
src_files_healthy = dir(input_images_healthy);
src_files_pneumonia= dir(input_images_pneumonia); 

%no. of images present in the directory
cov = length(src_files_covid);

for i=3:cov
    %compiled the filename for image i
   img_dir_cov = [input_dir1 src_files_covid(i).name];
   img_dir_healthy = [input_dir2 src_files_healthy(i).name];
   img_dir_pneumonia = [input_dir3 src_files_pneumonia(i).name];
   %extracted name, path and file extension
   [path1,name_cov,ext1] = fileparts(img_dir_cov);
   [path2,name_healthy,ext2] = fileparts(img_dir_healthy);
   [path3,name_pneumonia,ext3] = fileparts(img_dir_pneumonia);
  
   outfile_cov = [output_dir1 name_cov '.png'];
   outfile_healthy = [output_dir2 name_healthy '.png'];
   outfile_pneumonia = [output_dir3 name_pneumonia '.png'];
   
   %reading the input image
   img_cov = imread(img_dir_cov); 
   img_healthy = imread(img_dir_healthy);
   img_pneumonia = imread(img_dir_pneumonia);
   
   %rgb image to gray
   img_gray_cov = im2gray(img_cov); 
   img_gray_healthy = im2gray(img_healthy);
   img_gray_pneumonia = im2gray(img_pneumonia);
   
   %resize image to 224*224 pixels
   method = 'bicubic';
   img_resized_cov = imresize(img_gray_cov,[224 224],'method',method); 
   img_resized_healthy = imresize(img_gray_healthy,[224 224],'method',method);
   img_resized_pneumonia = imresize(img_gray_pneumonia,[224 224],'method',method);
   
   %saving the new image
   imwrite(img_resized_cov, outfile_cov);
   imwrite(img_resized_healthy, outfile_healthy);
   imwrite(img_resized_pneumonia, outfile_pneumonia);
end
