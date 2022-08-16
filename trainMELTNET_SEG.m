clearvars
nClasses = 10; % number of classes to be predicted by segmentation network

% path to folder containing all training and validation images, should have
% the following subfolders:
% training_inputs/ : containing .mat files each with a 64x64x4 input array (for network training only)
% validation_inputs/ : containing .mat files each with a 64x64x4 input array (for network validation only)
% training_targets/ : containing single channel images that serve as the segmentation network targets (corresponding to the training_inputs/ directory)
% validation_targets/ : containing single channel images that serve as the segmentation network targets (corresponding to the validation_inputs/ directory)

imageStoreFolder = "../NClass_"+num2str(nClasses);

load_params = false; % true if you want to load model from a previous state
load_epoch = 1; % if above true, from what epoch do you want to load?

% segmentation network options
nfilt = 3; % size of convolutional filters 
numEpochs = 600; % number of epochs to train for
LR = 0.0000005; % initial learning rate for SGDM
momentum = 0.85; % momentum for SGDM
miniBatchSize = 16; % number of images seen in each training iteration
imageSize = [64 64 4]; % input image size (SSC)

% create minibatchqueues for training and validation
labelIDs = floor(linspace(255,0,nClasses));
for ii = 1:nClasses
    classNames(ii) = "C"+num2str(labelIDs(ii));
end

imds = imageDatastore(imageStoreFolder+"/training_inputs",'FileExtensions','.mat','ReadFcn',@matReader);
labelDir = imageStoreFolder+"/training_targets";
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
ds = pixelLabelImageDatastore(imds,pxds);

valimds = imageDatastore(imageStoreFolder+"/validation_inputs",'FileExtensions','.mat','ReadFcn',@matReader);
vallabelDir = imageStoreFolder+"/validation_targets";
valds = pixelLabelDatastore(vallabelDir,classNames,labelIDs);
valds1 = pixelLabelImageDatastore(valimds,valds);

tbl = countEachLabel(pxds); % number of each label in the training set
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount; 
classWeights = 0.01./imageFreq; % to be used for weighted loss function

mbq = minibatchqueue(ds,...
    'MiniBatchSize',miniBatchSize,...
    'OutputAsDlarray',[1,0],...
    'MiniBatchFormat',{'SSCB',''},...
    'OutputEnvironment',{'gpu','cpu'});

valq = minibatchqueue(valds1,...
    'MiniBatchSize',64,...
    'OutputAsDlarray',[1,0],...
    'MiniBatchFormat',{'SSCB',''},...
    'OutputEnvironment',{'gpu','cpu'});

% initialise model parameters
if load_params
    load("checkpoints/NC" +sprintf('%02d',nClasses) + "_Epoch_"+ sprintf('%04d',load_epoch)+".mat", 'params','state','val_loss_sav','accuracy_sav','train_loss_sav');
    total_epochs = numEpochs + load_epoch;
    start_epoch = load_epoch;
else
    params = get_weights(nClasses,nfilt);
    state = initialise_state();
    total_epochs = numEpochs;
    start_epoch = 1;
    val_loss_sav = [];
    accuracy_sav = [];
    train_loss_sav = [];
end

% initialise training progress figures
figure
subplot(2,1,1);
lineLossTrain = animatedline(Color=[0.85 0.325 0.098]);
lineLossVal = animatedline(Color=[0 0.4470 0.7410]);
xlabel("Iteration")
ylabel("Loss")
ylim([0 inf])
grid on
subplot(2,1,2);
lineAccShelf = animatedline(Color=[0.85 0.325 0.098]);
lineAccTotal = animatedline(Color=[0 0.4470 0.7410]);
xlabel("Iteration")
ylabel("Accuracy")
ylim([0 100])
grid on;

vel = [];
iteration = 0;
start = tic;

for epoch = start_epoch:total_epochs
    % Shuffle data.
    shuffle (mbq);
    
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq);
        
        % Evaluate the model loss and gradients and updated state
        [loss, gradients,state] = dlfeval(@lossMELTNET_SEG, params, X, T, nClasses,state,true,classWeights);
        
        % Update the network parameters using the SGDM optimizer.
        [params,vel] = sgdmupdate(params,gradients,vel,LR,momentum);
        
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        Y = extractdata(loss);
        train_loss_sav = [train_loss_sav; double(Y)];
        addpoints(lineLossTrain,iteration,double(Y));
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        
        % every 50 iterations, make a prediction on the validation set
        if mod(iteration,50)==0
            shuffle(valq);
            [X1,T1] = next(valq);
            [loss,~,~,~, accuracy, total_accuracy] = dlfeval(@lossMELTNET_SEG, params, X1, T1, nClasses,state,true,classWeights);
            Y = extractdata(loss);
            addpoints(lineLossVal,iteration,double(Y));
            addpoints(lineAccShelf,iteration,accuracy);
            addpoints(lineAccTotal,iteration,total_accuracy);
            accuracy_sav = [accuracy_sav; accuracy];
            val_loss_sav = [val_loss_sav; double(Y)];
        end
        drawnow
    end
    
    if mod(epoch,100)==0 % save model param and state every 100 epochs
        
        save("checkpoints/NC" +sprintf('%02d',nClasses) + "_Epoch_"+ sprintf('%04d',epoch)+".mat", 'params','state','val_loss_sav','accuracy_sav','train_loss_sav');
        
    end
        
end


function out = init_weights(sfilt,channels,nfilt)

filterSize = [sfilt sfilt];
numChannels = channels;
numFilters = nfilt;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

out = initializeGlorot(sz,numOut,numIn);

end

function params = get_weights(nClasses,nfilt)

nf = 2.^(5:8);
nfb = [32,32,32,64,64,64,128,128,128,256,256,256,256,256,256,256,128,128,128,64,64,64,32,32,32,32,32,32];
se_rat = 8;

params.enWeights1{1} = init_weights(nfilt,4,32);
params.enWeights2{1} = init_weights(nfilt,32,32);
params.enWeights3{1} = init_weights(1,4,32);

params.Offset{1} = initializeZeros([32 1]);
params.Scale{1} = initializeOnes([32 1]);
params.Offset{2} = initializeZeros([32 1]);
params.Scale{2} = initializeOnes([32 1]);
for ii = 3:numel(nfb)
    params.Offset{ii} = initializeZeros([nfb(ii) 1]);
    params.Scale{ii} = initializeOnes([nfb(ii) 1]);
end
for ii = 1:4 
    params.enBias1{ii} = initializeZeros([1,1,nf(ii)]);
    params.enBias2{ii} = initializeZeros([1,1,nf(ii)]);
    params.enBias3{ii} = initializeZeros([1,1,nf(ii)]);
end
for ii = 1:3 
    
    params.deBias1{ii} = initializeZeros([1,1,nf(4-ii)]);
    params.deBias2{ii} = initializeZeros([1,1,nf(4-ii)]);
    params.deBias3{ii} = initializeZeros([1,1,nf(4-ii)]);
    
    params.enWeights1{ii+1} = init_weights(nfilt,nf(ii),nf(ii+1));
    params.enWeights2{ii+1} = init_weights(nfilt,nf(ii+1),nf(ii+1));
    params.enWeights3{ii+1} = init_weights(1,nf(ii),nf(ii+1));
    
    params.deWeights1{ii} = init_weights(2,nf(4-ii),nf(5-ii));
    params.deWeights2{ii} = init_weights(nfilt,nf(5-ii),nf(4-ii));
    params.deWeights3{ii} = init_weights(nfilt,nf(4-ii),nf(4-ii));
    params.deWeights4{ii} = init_weights(1,nf(5-ii),nf(4-ii));
    
    params.seWeights1{ii} = dlarray(ones(nf(ii)./se_rat,1,1,nf(ii))); 
    params.seWeights2{ii} = dlarray(ones(nf(ii),1,1,nf(ii)./se_rat)); 
end
params.BiasOut = initializeZeros([1 1 nClasses]);
params.WeightsOut = init_weights(1,32,nClasses);

params.ASPPWeights1 = init_weights(nfilt,256,256);
params.ASPPWeights2 = init_weights(nfilt,256,256);
params.ASPPWeights3 = init_weights(nfilt,256,256);
params.ASPPWeights4 = init_weights(nfilt,256,256);
params.ASPPWeights5 = init_weights(1,256,256);

params.ASPPWeights6 = init_weights(nfilt,32,32);
params.ASPPWeights7 = init_weights(nfilt,32,32);
params.ASPPWeights8 = init_weights(nfilt,32,32);
params.ASPPWeights9 = init_weights(nfilt,32,32);
params.ASPPWeights10 = init_weights(1,32,32);

params.ASPPBias1 = initializeZeros([1 1 256]);
params.ASPPBias2 = initializeZeros([1 1 256]);
params.ASPPBias3 = initializeZeros([1 1 256]);
params.ASPPBias4 = initializeZeros([1 1 256]);
params.ASPPBias5 = initializeZeros([1 1 256]);

params.ASPPBias6 = initializeZeros([1 1 32]);
params.ASPPBias7 = initializeZeros([1 1 32]);
params.ASPPBias8 = initializeZeros([1 1 32]);
params.ASPPBias9 = initializeZeros([1 1 32]);
params.ASPPBias10 = initializeZeros([1 1 32]);
end

function state = initialise_state()

nf = [32,32,32,64,64,64,128,128,128,256,256,256,256,256,256,256,128,128,128,64,64,64,32,32,32,32,32,32];

for ii = 1:numel(nf)
    state.batchnorm{ii}.TrainedMean = initializeZeros([nf(ii) 1]);
    state.batchnorm{ii}.TrainedVariance = initializeOnes([nf(ii) 1]);
end

end

function parameter = initializeOnes(sz)

parameter = ones(sz,'single');
parameter = dlarray(parameter);

end

function parameter = initializeZeros(sz)

parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end