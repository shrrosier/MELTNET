function [loss, gradients, state, yPred, accuracy, total_accuracy] = lossMELTNET_SEG(params,X,T,nC,state,training,Cw)


%% first layer

c1a = dlconv(X, params.enWeights1{1}, params.enBias1{1}, 'Stride',1,'Padding','same');
[bn1a,state] = batchnorm_state(c1a,params,1,state,training);
r1b = bn1a.*sigmoid(bn1a);
c1b = dlconv(r1b, params.enWeights2{1}, params.enBias2{1}, 'Stride',1,'Padding','same');

c1c = dlconv(X, params.enWeights3{1}, params.enBias3{1}, 'Stride',1,'Padding','same');
[bn1c,state] = batchnorm_state(c1c,params,2,state,training);
add1 = bn1c + c1b;

%% encoder blocks

[add2,state] = SE_encoder(add1,1,state);

[add3,state] = SE_encoder(add2,2,state);

[add4,state] = SE_encoder(add3,3,state);

%% ASPP bridge block

x1a = dlconv(add4,params.ASPPWeights1,params.ASPPBias1,'Padding','same','DilationFactor',[6 6]);
[x1b,state] = batchnorm_state(x1a,params,12,state,training);

x2a = dlconv(add4,params.ASPPWeights2,params.ASPPBias2,'Padding','same','DilationFactor',[12 12]);
[x2b,state] = batchnorm_state(x2a,params,13,state,training);

x3a = dlconv(add4,params.ASPPWeights3,params.ASPPBias3,'Padding','same','DilationFactor',[18 18]);
[x3b,state] = batchnorm_state(x3a,params,14,state,training);

x4a = dlconv(add4,params.ASPPWeights4,params.ASPPBias4,'Padding','same');
[x4b,state] = batchnorm_state(x4a,params,15,state,training);

aspp1 = x1b + x2b + x3b + x4b;
aspp2 = dlconv(aspp1,params.ASPPWeights5,params.ASPPBias5,'Padding','same');

%% decoder blocks

[add5,state] = SE_decoder(aspp2,add3,1,state);

[add6,state] = SE_decoder(add5,add2,2,state);

[add7,state] = SE_decoder(add6,add1,3,state);

%% ASPP output block

y1a = dlconv(add7,params.ASPPWeights6,params.ASPPBias6,'Padding','same','DilationFactor',[6 6]);
[y1b,state] = batchnorm_state(y1a,params,25,state,training);

y2a = dlconv(add7,params.ASPPWeights7,params.ASPPBias7,'Padding','same','DilationFactor',[12 12]);
[y2b,state] = batchnorm_state(y2a,params,26,state,training);

y3a = dlconv(add7,params.ASPPWeights8,params.ASPPBias8,'Padding','same','DilationFactor',[18 18]);
[y3b,state] = batchnorm_state(y3a,params,27,state,training);

y4a = dlconv(add7,params.ASPPWeights9,params.ASPPBias9,'Padding','same');
[y4b,state] = batchnorm_state(y4a,params,28,state,training);

aspp3 = y1b + y2b + y3b + y4b;
aspp4 = dlconv(aspp3,params.ASPPWeights10,params.ASPPBias10,'Padding','same');

%% output layers

cfin = dlconv(aspp4, params.WeightsOut, params.BiasOut, 'Stride',1,'Padding','same');
sm = softmax(cfin);

%% Prediction

% predicted melt rate is highest activation channel not including pixels
% outside of the ice shelf domain, all other pixels forced to be zero melt
% since this is known a-priori. This avoids very occasional mis-
% classification of pixels within an ice shelf as 'no melt'
[~,yPred1] = max(sm(:,:,1:nC-1,:),[],3);
outside = squeeze(X(:,:,2,:)==0);
yPred1(outside)= nC;
yPred =squeeze(extractdata(yPred1));

if training
    
    T1 = categorical(T);
    cl = string(1:nC); % this is needed in case a prediction misses a label
    T2 = onehotencode(T1,4,"ClassNames",cl); % one hot encode format
    T3 = permute(T2,[1 2 4 3]); % rearrange from SSBC to SSCB

    weights = dlarray(ones(size(T3))); 
    for ii = 1:nC-1
        weights(:,:,ii,:) = weights(:,:,ii,:).*Cw(ii);
    end
    weights(:,:,nC,:) = 0; % set weight of 'no melt' class to zero
    
    % calculate weighted cross entropy loss
    loss = dlarray(0);
    loss = loss + crossentropy(sm,T3,weights,'TargetCategories','exclusive');

    % calculate gradient of loss with respect to model parameters
    [gradients] = dlgradient(loss, params);
    
    
    nobs = size(yPred,3);
    idx = T~=nC; % mask for ignoring 'no melt' class

    % classification accuracy for melting pixels only
    accuracy = 100.*sum(T(idx)==yPred(idx),'all')./sum(idx,'all'); 
    % classification accuracy for all pixels
    total_accuracy = 100.*sum(T==yPred,'all')./(64*64*nobs);
    
else % no need to calculate loss or gradients when making a prediction
    
    loss = [];
    gradients = [];
    accuracy = [];
    total_accuracy = [];
    
end

    function [out,state] = SE_encoder(in,idx,state)
        % Squeeze and excited encoder block

        
        nf = 2.^(5:8); 
        sidx1 = (idx-1)*3 + [3:5]; % batch norm state index
        
        % squeeze and excite block
        se1a = avgpool(in,'global');
        se1b = fullyconnect(se1a,params.seWeights1{idx},zeros(nf(idx)/8,1));
        se1c = se1b.*sigmoid(se1b);
        se1d = fullyconnect(se1c,params.seWeights2{idx},zeros(nf(idx),1));
        se1e = se1d.*sigmoid(se1d);
        se1out = se1e.*in;
        
        % main convolutional block
        [bn2a,state] = batchnorm_state(se1out,params,sidx1(1),state,training);
        r2a = bn2a.*sigmoid(bn2a);
        c2a = dlconv(r2a, params.enWeights1{idx+1},  params.enBias1{idx+1}, 'Stride',2,'Padding','same');
        [bn2b,state] = batchnorm_state(c2a,params,sidx1(2),state,training);
        r2b = bn2b.*sigmoid(bn2b);
        c2b = dlconv(r2b, params.enWeights2{idx+1},  params.enBias2{idx+1}, 'Stride',1,'Padding','same');
        
        % shortcut
        c2c = dlconv(se1out, params.enWeights3{idx+1}, params.enBias3{idx+1}, 'Stride',2,'Padding','same');
        [bn2c,state] = batchnorm_state(c2c,params,sidx1(3),state,training);
        
        % add
        out = bn2c + c2b;
        
    end

    function [out,state] = SE_decoder(in,skp,idx,state)
        % Squeeze and excited decoder block

        
        sidx1 = (idx-1)*3 + [16:18];
        
        % upsampling/concatenation layer (no attention gates)
        c5u = dltranspconv(in, params.deWeights1{idx}, params.deBias1{idx}, 'Stride',2,'Cropping',0);
        q5 = cat(3,c5u,skp);
        
        % main convolutional block
        [bn5a,state] = batchnorm_state(q5,params,sidx1(1),state,training);
        r5a = bn5a.*sigmoid(bn5a);
        c5a = dlconv(r5a, params.deWeights2{idx},  params.deBias1{idx}, 'Stride',1,'Padding','same');
        [bn5b,state] = batchnorm_state(c5a,params,sidx1(2),state,training);
        r5b = bn5b.*sigmoid(bn5b);
        c5b = dlconv(r5b, params.deWeights3{idx},  params.deBias2{idx}, 'Stride',1,'Padding','same');
        
        %shortcut
        c5c = dlconv(q5, params.deWeights4{idx}, params.deBias3{idx}, 'Stride',1,'Padding','same');
        [bn5c,state] = batchnorm_state(c5c,params,sidx1(3),state,training);
        
        % add
        out = bn5c + c5b;
        
    end

    function [out,state] = batchnorm_state(in,params,idx,state,training)
        % batch normalization layer with optional state update
        
        if training % if training, update the batch norm mean and variance
            [out,trainedMean,trainedVariance] = batchnorm(in,params.Offset{idx},params.Scale{idx},state.batchnorm{idx}.TrainedMean,state.batchnorm{idx}.TrainedVariance);
            
            state.batchnorm{idx}.TrainedMean = trainedMean;
            state.batchnorm{idx}.TrainedVariance = trainedVariance;
        else
            out = batchnorm(in,params.Offset{idx},params.Scale{idx},state.batchnorm{idx}.TrainedMean,state.batchnorm{idx}.TrainedVariance);
        end
        
        
    end




end