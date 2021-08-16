clear all; close all;
f=imread('yx1.jpg'); %%"yx1.jpg" is the file name. You can change it to your own photos. 

tn=500;
figure(1);imshow(f); %% To test the read matrix.
ff=f(:,:,1);

ff=ff(1:743,1:1200); %% To uniform the data matrix into certain scale (benefit future learning)
ff=double(ff);
ff=ff/255;
figure(2);imshow(ff); %% To test the current figure matrix. 
ff=ff-0.5; 
 [aa,bb]=find(abs(ff+0.5)<0.05);
 [cc,dd]=find(abs(ff+0.5)>0.05); 
[m,t]=size(aa);
[n,t]=size(cc);
rand_index1=randperm(m);
rand_index2=randperm(n);
draw_rand_index1=rand_index1(1:tn);
draw_rand_index2=rand_index2(1:tn);
aa1=aa(draw_rand_index1);bb1=bb(draw_rand_index1);
cc1=cc(draw_rand_index2);dd1=dd(draw_rand_index2);
trainx(1,1:tn)=aa1';trainx(1,tn+1:2*tn)=cc1';
trainz(1,1:tn)=bb1';trainz(1,tn+1:2*tn)=dd1';
label(1,1:tn)=1;label(1,tn+1:2*tn)=0;
label(2,:)=1-label(1,:); %% To label the featues in 500 pixels. 
train.trainx=trainz;train.trainz=trainx;train.label=label; 
save trainzt train; %% To save the labeled matrix. 
image_input=ff;
save image1 image_input;

