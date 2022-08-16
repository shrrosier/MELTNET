clearvars;

nClasses =10; % number of categories/labels from the segmentation network

% load training dataset in the form:
% XTrain: categorised ice shelf melt rate for training (SSCB 64x64x1xNtrain)
% XValidation: categorised ice shelf melt rate for validation  (SSCB 64x64x1xNval)
% YTrain: true melt rates from NEMO for training (SSCB 64x64x1xNtrain)
% YValidation: true melt rates from NEMO for validation (SSCB 64x64x1xNval)
fname = "ClassN_"+num2str(nClasses);
load(fname+"_AE_dataset.mat");

% define DAE network architecture
layers = [
    imageInputLayer([64 64 1],"Name","InputLayer","Normalization","none")
    convolution2dLayer([3 3],16,"Name","Conv1","Padding",[1 1 1 1],"WeightsInitializer","he")
    swishLayer("Name","ReLU1")
    convolution2dLayer([3 3],16,"Name","Conv2","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"WeightsInitializer","he")
    batchNormalizationLayer("Name","BNorm2","OffsetL2Factor",0,"ScaleL2Factor",0)
    swishLayer("Name","ReLU2")
    convolution2dLayer([3 3],16,"Name","Conv3","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"WeightsInitializer","he")
    batchNormalizationLayer("Name","BNorm3","OffsetL2Factor",0,"ScaleL2Factor",0)
    swishLayer("Name","ReLU3")
    convolution2dLayer([3 3],16,"Name","Conv4","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"WeightsInitializer","he")
    batchNormalizationLayer("Name","BNorm4","OffsetL2Factor",0,"ScaleL2Factor",0)
    swishLayer("Name","ReLU4")
    convolution2dLayer([3 3],16,"Name","Conv5","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"WeightsInitializer","he")
    batchNormalizationLayer("Name","BNorm5","OffsetL2Factor",0,"ScaleL2Factor",0)
    swishLayer("Name","ReLU5")
    convolution2dLayer([3 3],1,"Name","Conv6","Padding",[1 1 1 1],"WeightsInitializer","he")
    regressionLayer("Name","FinalRegressionLayer")];

% define training options
opts = trainingOptions('sgdm', ...
    'MaxEpochs',500, ...
    'InitialLearnRate',2e-8, ...
    'Momentum',0.9, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ValidationData',{XValidation,YValidation}, ...
    'MiniBatchSize',16, ...
    'L2Regularization',5e-2, ...
    'Verbose',true);

% train network
[net,info] = trainNetwork(XTrain,YTrain,layers,opts);

% save trained network
save(fname+"inverse_classifier_net.mat",'net' ,'info')
