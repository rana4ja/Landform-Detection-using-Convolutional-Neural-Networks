clear all,close all

addpath ./CNN
addpath ./util
%addpath ./data/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Inputs                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntrain=500; % The number of training set

wdx=8; 
wdz=8;
image_x=wdx*2+1; % Original Image Patch Size X
image_z=wdz*2+1; % Original Image Patch Size Z
% Intepolate image to make sure size before pooling is even
image_x_new=32;  % Intepolated Image Patch Size X
image_z_new=32;  % Intepolated Image Patch Size Z

opts.alpha = 1;   % Learning Rate
opts.batchsize = 20;  % Minibatch Size
opts.numepochs = 20;  % Iteration Number

%%%%  CNN structure %%%%%%%

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 7, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
    struct('type', 'c', 'outputmaps', 14, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Step 1 Prepare test and training set            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load image1.mat 

%%%%%%%%%%%% Load Prepared training set%%%%%%%%%%%%% 
load trainzt.mat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Step 2 Convert Image into Image patches             % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nz,nx]=size(image_input);
image_pad=padimage(image_input,wdz,wdx);
%%% Obtain image batches from full image
image_slice=zeros(image_z,image_x,nz*nx);
image_slice_new=zeros(image_z_new,image_x_new,nz*nx);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
ii=1;
for ix=1:nx
    for iz=1:nz
        image_slice(1:image_z,1:image_x,ii)=image_pad(iz:iz+image_z-1,ix:ix+image_x-1);
        image_slice_new(:,:,ii)=interp2(image_slice(:,:,ii),Xq,Zq);
        ii=ii+1;
    end
end

%%% Obtain image batches from train set
train_slice=zeros(image_z,image_x,2*ntrain);
train_slice_new=zeros(image_z_new,image_x_new,2*ntrain);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
for in=1:2*ntrain
        train_slice(1:image_z,1:image_x,in)=...
            image_pad(train.trainz(in):train.trainz(in)+image_z-1,train.trainx(in):train.trainx(in)+image_x-1);
        train_slice_new(:,:,in)=interp2(train_slice(:,:,in),Xq,Zq);
end

clear image_slice train_slice

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Step 3 Train CNN structure                    % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Input Training and Test set
train_x =train_slice_new;  % Training set
train_y =train.label;      % Training label
test_x =image_slice_new;   % Test set


rand('state',0)
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Step 4 Input full image for Classification             % 
%   (Seperate The Image into 10 parts In case of out of memory)  %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii=1:(nx/100)
%for ii=1:1
disp(['Part ',num2str(ii),' Start']);
tic
uu = cnnff(cnn, test_x(:,:,nz*100*(ii-1)+1:nz*100*ii));
toc
[vv,ll]=max(uu.o);
[zz,ee]=min(uu.o);
tran=uu.fv(end,:);
ui=reshape(-ll+2,nz,100);
ai=reshape(ee,nz,100);
bi=reshape(tran,nz,100);
un(:,100*(ii-1)+1:100*ii)=ui(:,1:100);
an(:,100*(ii-1)+1:100*ii)=ai(:,1:100);
bn(:,100*(ii-1)+1:100*ii)=bi(:,1:100);
end


%%%%%%%%%% Image the result %%%%%%%%%%%%
figure(1)
subplot(1,2,1)
imagesc(image_input)
title({['Input Image']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
subplot(1,2,2)
imagesc(un)
title({['CNN Labels']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
set(gcf,'unit','centimeters','position',[12 5 30 13])
colormap(parula)


figure;plot(uu.rL)
