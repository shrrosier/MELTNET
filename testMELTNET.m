clearvars;

% load the DAE network, along with associated clip properties
load('MELTNET_DAE_C10_E500.mat','DAEnet','clip_min','clip_max');

% load the SEG network param and state
load('MELTNET_SEG_C10_E600.mat','params','state');

% directories to the validation inputs and corresponding melt rates
input_files = dir("NClass_10/validation_inputs/*.mat");
target_files = dir("NClass_10/validation_meltrates/*.mat");
nfiles = numel(input_files);

% initialise input and output 4D matrices
X = zeros(64,64,4,nfiles);
T = zeros(64,64,1,nfiles);

% loop through files, load matrices and store in the 4D matrices
for ii = 1:nfiles
    load("NClass_10/validation_inputs/"+input_files(ii).name,'input_out');
    load("NClass_10/validation_meltrates/"+target_files(ii).name,'ab');

    X(:,:,:,ii) = input_out;
    T(:,:,:,ii) = ab;
end

% run MELTNET for 10 classes using inputs in X (which must be in dlarray format)
X = dlarray(X,'SSCB');
Y = MELTNET(X,DAEnet,params,state,"nClasses",10,"clip_min",clip_min,"clip_max",clip_max);

% for easy comparison, mask out all not-floating pixels in target
shelf = ~isnan(T);
Yvec = double(Y(shelf)); Tvec = T(shelf); % create vectors of melt rates 

nrmse = sqrt(mean((Yvec-Tvec).^2))./range(Tvec); % normalized root mean square error

cc = corr(Yvec,Tvec); % correlation coefficient