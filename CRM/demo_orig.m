%demo
clc;close all;clear;

addpath('./low');

%% image set 
para.img_set_name= 'LOW_IMAGE';
para.img_path=['./', para.img_set_name, '/'];
para.result_path = ['./img_output/low/', para.img_set_name, '/'];

if (~exist(para.result_path, 'dir')) 
    mkdir(para.result_path);
end
para.files_list=dir([para.img_path '*.PNG']);
para.img_num=length(para.files_list);

for img_idx = 1:para.img_num
    I = imread([para.img_path, para.files_list(img_idx).name]);
    J = Ying_2017_CAIP(I); 
    imwrite(J,[para.result_path para.files_list(img_idx,1).name(1:end-4) '_CAIP.png'],'png');
    J2 = Ying_2017_ICCV(I); 
    imwrite(J2,[para.result_path para.files_list(img_idx,1).name(1:end-4) '_ICCV.png'],'png');
end

%I = imread('2.bmp');
%J = Ying_2017_ICCV(I); 
%imwrite(J,'2_output.png');
%subplot 121; imshow(I); title('Original Image');
%subplot 122; imshow(J); title('Enhanced Result');
