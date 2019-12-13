clear,clc,close all
%Get list of persons
Curr = cd;
cd([Curr '\Train']);
categories = ls; categories(1:2,:) = [];
cd(Curr)
rootFolder = fullfile(Curr, 'Train');

%Loading training dataset
imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Split images randomely into train and test sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% Modify Alexnet for new classification
net = alexnet; % Load the pretrained Alexnet
net = net.Layers(1:end-3); 
numClasses = length(categories);
net = [net;fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
       softmaxLayer;classificationLayer];

% Setting training parameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',16, ...
    'InitialLearnRate',1e-4, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
% Training new network
classifier = trainNetwork(imdsTrain,net,options);

% Evaluate Classifier accuracy
[YPred,~] = classify(classifier,imdsValidation);
YValidation = imdsValidation.Labels;
Accuracy = mean(YPred == YValidation);
disp('=====================================')
disp(['Mean accuracy = ' num2str(Accuracy)])
disp('=====================================')

% Saving trained model
save('Classifier2');