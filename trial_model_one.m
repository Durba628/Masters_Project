clc;
clear;
close all;

dataDir = 'C:\Users\durba\Downloads\Durba_PM\Durba_PM';

imds = imageDatastore( dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');


classes = categories(imdsTrain.Labels);
numClasses = numel(classes);

tbl = countEachLabel(imdsTrain);
total = sum(tbl.Count);
classWeights = total ./ tbl.Count;

inputSize = [128 128 3];

trainAugmenter = imageDataAugmenter( 'RandRotation', [-30 30], 'RandXReflection', true, 'RandYReflection', true);

augimdsTrain = augmentedImageDatastore( inputSize, imdsTrain, 'DataAugmentation', trainAugmenter);

augimdsVal = augmentedImageDatastore( inputSize, imdsVal);   % NO augmentation for validation

layers = [
    imageInputLayer(inputSize, 'Name','input')

    convolution2dLayer(3, 16, 'Padding','same', 'Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool1')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool2')

    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')

    classificationLayer( 'Classes', classes, 'ClassWeights', classWeights, 'Name','classOutput')];


options = trainingOptions('adam', 'InitialLearnRate', 1e-3, 'MaxEpochs', 15, 'MiniBatchSize', 20, 'Shuffle','every-epoch', 'ValidationData', augimdsVal,'ValidationFrequency', 30, 'Plots','training-progress', 'Verbose', false);


net = trainNetwork(augimdsTrain, layers, options);


YPred = classify(net, augimdsVal);
YVal  = imdsVal.Labels;

valAccuracy = mean(YPred == YVal) * 100;
fprintf('Validation Accuracy: %.2f %%\n', valAccuracy);


figure;
confusionchart(YVal, YPred);
title('Validation Confusion Matrix');


classNames = net.Layers(end).Classes;
idx = randperm(numel(imdsVal.Files), min(10, numel(imdsVal.Files)));

for i = idx
    img = imread(imdsVal.Files{i});
    [label, scores] = classify(net, imresize(img, inputSize(1:2)));

    figure;
    imshow(img);
    title(sprintf('Predicted: %s (%.2f%%)', string(label), max(scores)*100));
end



save('trialModelOne.mat', 'net');
