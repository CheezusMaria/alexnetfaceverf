clear,clc,close all
load('Classifier2.mat');
% Select image to do verification
[filename, pathname] = uigetfile('*.jpg','select image');
newImage = fullfile(pathname, filename);
img = imread(newImage);
img1 = img(:,1:113,:);
img1 = imresize(img1, [227 227]);
img2 = img(:,115:end,:);
img2 = imresize(img2, [227 227]);

% Use the trained net to classify faces
result1 = classify(classifier,img1);
result2 = classify(classifier,img2);

% Make a decision of matching or not using the classifier
if result1 == result2
    T = 'Matching images';
else
    T = 'Not matching images';
end

% Show final result
imshow(img),title(T);
