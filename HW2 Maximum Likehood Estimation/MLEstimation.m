load('.../trainingSamplesDCT_8_new.mat');

numCheetahSamples = size(TrainsampleDCT_FG, 1);
numGrassSamples = size(TrainsampleDCT_BG, 1);


totalSamples = numCheetahSamples + numGrassSamples;


PY_cheetah = numCheetahSamples / totalSamples;
PY_grass = numGrassSamples / totalSamples;

disp(['PY(cheetah) = ' num2str(PY_cheetah)]);
disp(['PY(grass) = ' num2str(PY_grass)]);

prior=[PY_cheetah,PY_grass];
figure;
bar(["cheetah";"grass"],prior);
title('Prior Probability Estimate');
ylabel('Probability');
ax = gca;
exportgraphics(ax,".../priors.jpg")

%As the features follow a Gaussian distribution, the sample mean and sample
%variance are the best ML estimators
mean_cheetah=mean(TrainsampleDCT_FG,1);
mean_grass=mean(TrainsampleDCT_BG,1);

std_cheetah=std(TrainsampleDCT_FG,0,1);
std_grass=std(TrainsampleDCT_BG,0,1);

for i=1:64
    % Taking range of x between 5 times the std deviation from the mean
    x_cheetah(i,:)=(mean_cheetah(i)-5*std_cheetah(i)):std_cheetah(i)/100:(mean_cheetah(i)+5*std_cheetah(i));
    Px_cheetah(i,:)= normpdf(x_cheetah(i,:),mean_cheetah(i),std_cheetah(i));

    x_grass(i,:)=mean_grass(i)-5*std_grass(i):std_grass(i)/100:mean_grass(i)+5*std_grass(i);
    Px_grass(i,:) = normpdf(x_grass(i,:),mean_grass(i),std_grass(i));
end


% Plotting P(xi|cheetah) and P(xi|grass) desnsities of for i=[1,64]
for k=0:3
    fig=figure;
    for i = 1:16
        subplot(4,4,i);
        plot(x_cheetah(i+16*k, :),Px_cheetah(i+16*k, :),'-b',x_grass(i+16*k, :),Px_grass(i+16*k, :),'-r');
        title(['Features ',num2str(i+16*k)]);
    end
    print(fig,'-djpeg',sprintf(".../All_features %d.jpg",k+1));
end

%Choosing best and worst features by visualizing the distributions
best=[1,7,8,9,12,14,18,27];
worst=[3,4,5,59,60,62,63,64];

fig=figure;
for i=1:8
    subplot(2,4,i);
    ix=best(i);
    plot(x_cheetah(ix, :),Px_cheetah(ix, :),'-b',x_grass(ix, :),Px_grass(ix, :),'-r');
    title(['Best Features ',num2str(ix)]);
end
print(fig,'-djpeg',".../Best_features.jpg");

fig=figure;
for i=1:8
    subplot(2,4,i);
    ix=worst(i);
    plot(x_cheetah(ix, :),Px_cheetah(ix, :),'-b',x_grass(ix, :),Px_grass(ix, :),'-r');
    title(['Worst Features ',num2str(ix)]);
end
print(fig,'-djpeg',".../Worst_features.jpg");

% calculating covariance matrix and alpha values for the training samples
cov_cheetah_64=cov(TrainsampleDCT_FG);
cov_grass_64=cov(TrainsampleDCT_BG);
alpha_cheetah_64=log(((2*pi)^64)*det(cov_cheetah_64))-2*log(PY_cheetah);
alpha_grass_64=log(((2*pi)^64)*det(cov_grass_64))-2*log(PY_grass);


dct_fg_8=TrainsampleDCT_FG(:,best);
dct_bg_8=TrainsampleDCT_BG(:,best);

mean_cheetah_best8=mean_cheetah(best);
mean_grass_best8=mean_grass(best);

cov_cheetah_best8=cov(dct_fg_8);
cov_grass_best8=cov(dct_bg_8);
alpha_cheetah_best8=log(((2*pi)^8)*det(cov_cheetah_best8))-2*log(PY_cheetah);
alpha_grass_best8=log(((2*pi)^8)*det(cov_grass_best8))-2*log(PY_grass);



I = imread('.../cheetah.bmp');
I=im2double(I);



A = zeros(size(I, 1) - 7,size(I, 2) - 7);
A_best8 = zeros(size(I, 1) - 7,size(I, 2) - 7);

for i = 1:size(I, 1) - 7
    for j = 1:size(I, 2) - 7
        block = I(i:i+7, j:j+7);
        block_dct = dct2(block);
        dct_flat=zigzaged(block_dct);
        dct_flat_best8=dct_flat(best);
        g_cheetah=1/(1+exp(dxy(dct_flat,mean_cheetah,cov_cheetah_64)-dxy(dct_flat,mean_grass,cov_grass_64)+alpha_cheetah_64-alpha_grass_64));
        if g_cheetah>0.5
            A(i+3,j+3)=1
        end
        g_cheetah_best8=1/(1+exp(dxy(dct_flat_best8,mean_cheetah_best8,cov_cheetah_best8)-dxy(dct_flat_best8,mean_grass_best8,cov_grass_best8)+alpha_cheetah_best8-alpha_grass_best8));
        if g_cheetah_best8>0.5
            A_best8(i+3,j+3)=1
        end
    end
end
% Padding the image with zeros
A_resized=zeros(255,270);
A_best8_resized=zeros(255,270);
for i=4:251
    for j=4:266
        A_resized(i,j)=A(i-3,j-3);
        A_best8_resized(i,j)=A_best8(i-3,j-3);
    end
end

figure;
subplot(1,2,1);
imshow(A_resized);
title("Mask with all 64 features");
ax = gca;
exportgraphics(ax,".../mask_64.jpg");
subplot(1,2,2);
imshow(A_best8_resized);
title("Mask with best 8 features");
ax = gca;
exportgraphics(ax,".../mask_best8.jpg");


im_test = imread('.../cheetah_mask.bmp');
im_test=im2double(im_test);
err=abs(im_test-A_resized);
prob_err=sum(err,"all")/(255*270);
disp(['Probability error with 64 features:' num2str(prob_err)]);
err2=abs(im_test-A_best8_resized);
prob_err2=sum(err2,"all")/(255*270);
disp(['Probability error with 8 best features:' num2str(prob_err2)]);

imwrite(A_resized,".../mask_64.bmp");
imwrite(A_best8_resized,".../mask_best8.bmp");

function output= zigzaged(input)
    zigzag=importdata('.../Zig-Zag Pattern.txt') ;
    zigzag=zigzag+1;
    output=zeros(1,64);
    for i=1:8
        for j=1:8
            output(zigzag(i,j))=input(i,j);
        end
    end
end

function output=dxy(x,y,cov)
    output=(x-y)*inv(cov)*transpose(x-y);
end



