clearvars;

load('MELTNET_DAE_C10_E500.mat','DAEnet','clip_min','clip_max');

load('MELTNET_SEG_C10_E600.mat','params','state');

input_files = dir("NClass_10/validation_inputs/*.mat");
target_files = dir("NClass_10/validation_targets/*.png");
nfiles = numel(input_files);

X = zeros(64,64,4,nfiles);
T = zeros(64,64,1,nfiles);

for ii = 1:nfiles
    load("NClass_10/validation_inputs/"+input_files(ii).name,'input_out');
    img = imread("NClass_10/validation_targets/"+target_files(ii).name);
    X(:,:,:,ii) = input_out;
    T(:,:,:,ii) = img;
end

X = dlarray(X,'SSCB');

Y = MELTNET(X,DAEnet,params,state,"nClasses",10,"clip_min",clip_min,"clip_max",clip_max);


