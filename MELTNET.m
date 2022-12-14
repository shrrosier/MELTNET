function Y = MELTNET(X,DAEnet,SEGparams,SEGstate,options)

arguments
    X (64,64,4,:) dlarray % input image consisting of four bands representing bathymetry, draft, temperature and salinity
    DAEnet SeriesNetwork % trained DAE network
    SEGparams struct % weights and biases from trained segmentation net
    SEGstate struct % normalization states from trained segmentation net
    options.nClasses (1,1) = 10 % number of classes for segmentation net
    options.clip_min (1,1) = -150 % minimum melt rate cutoff used for DAE network training
    options.clip_max (1,1) = 10 % maximum melt rate cutoff used for DAE network training
end 

% call the segmentation network to classify melt rates
[~, ~, ~,segnetY] = dlfeval(@lossMELTNET_SEG, SEGparams, X, [], options.nClasses,SEGstate,false,[]);
labelIDs = floor(linspace(255,0,options.nClasses));
img3d = labelIDs(segnetY);

% predict requires input images with a channel dimension, our output is
% grayscale so this is added as a dummy dimension
img4d(1,:,:,:) = img3d;
img_SSCB = permute(img4d,[2 3 1 4]);

% call the DAE network to convert from categorised to continuous melt rates
DAEnetY = predict(DAEnet,img_SSCB);
outside = extractdata(X(:,:,2,:)==0);
% final melt rate recovered by reversing the normalisation done to melt
% rates in the DAE network training set
Y = (DAEnetY./(255/(options.clip_min-options.clip_max)))+options.clip_max;
Y(outside) = nan;

end