load('.../TrainingSamplesDCT_8_new.mat');
num_FG = size(TrainsampleDCT_FG, 1);
num_BG = size(TrainsampleDCT_BG, 1);
p_FG=num_FG/(num_FG+num_BG);
p_BG=num_FG/(num_FG+num_BG);

I = imread('.../cheetah.bmp');
I=im2double(I);
[row,col]=size(I);
A=zeros((row-7)*(col-7),64);
index=1;

for i = 1:row - 7
    for j = 1:col - 7
        block = I(i:i+7, j:j+7);
        block_dct = dct2(block);
        dct_flat=zigzag(block_dct);
        A(index,:)=dct_flat;
        index=index+1;
    end
end


% For each class, learn 5 mixtures of C = 8 component
C=8;
dim= [1,2,4,8,16,24,32,40,48,56,64];
error_mat=zeros(25,11);
for i=1:1
    [pi_FG,mu_FG,cov_FG]=EM(8,TrainsampleDCT_FG,1000);
    for j=1:1
        [pi_BG,mu_BG,cov_BG]=EM(8,TrainsampleDCT_BG,1000);
        for d=1:1
            mask=zeros(size(A,1),1);
            for k=1:size(A,1)
                prob_bg=0;
                prob_fg=0;
                for l=1:C
                    prob_bg=prob_bg+mvnpdf(A(k,1:dim(d)),mu_BG(l,1:dim(d)),cov_BG(1:dim(d),1:dim(d),l))*pi_BG(l);
                    prob_fg=prob_fg+mvnpdf(A(k,1:dim(d)),mu_FG(l,1:dim(d)),cov_FG(1:dim(d),1:dim(d),l))*pi_FG(l);
                end
                if(prob_fg*p_FG>prob_bg*p_BG)
                    mask(k)=1;
                end
            end
            mask=reshape(mask,263,248)';
            mask_resized=zeros(255,270);
            mask_resized(1:248,1:263)=mask;
            error_mat(5*(i-1)+j,d)=error(mask_resized);
        end
    end
end

figure;
for j=1:5
    subplot(3,2,j);
    for i = 5*(j-1)+1:5*(j-1)+5
        plot(dim,error_mat(i, :), 'LineWidth', 1); 
        hold on; 
    end
    xlabel('Number of Dimensions');
    ylabel('Probability of Error');
    title(sprintf('FG%d Mixture',j));
    legend(sprintf('FG%dBG1',j),sprintf('FG%dBG2',j),sprintf('FG%dBG3',j),sprintf('FG%dBG4',j),sprintf('FG%dBG5',j));
    grid on;
    hold off;
end


% For each class, learn mixtures with C ∈ {1, 2, 4, 8, 16, 32}
mixtures=[1,2,4,8,16,32];
error_mat_b=zeros(length(mixtures),length(dim));
for c=1:length(mixtures)
    [pi_FG,mu_FG,cov_FG]=EM(mixtures(c),TrainsampleDCT_FG,1000);
    [pi_BG,mu_BG,cov_BG]=EM(mixtures(c),TrainsampleDCT_BG,1000); 
    for d=1:length(dim)
        disp(c);
        disp(d);
        mask=zeros(size(A,1),1);
        for i=1:size(A,1)
            prob_bg=0;
            prob_fg=0;
            for j=1:mixtures(c)
                prob_bg=prob_bg+mvnpdf(A(i,1:dim(d)),mu_BG(j,1:dim(d)),cov_BG(1:dim(d),1:dim(d),j))*pi_BG(j);
                prob_fg=prob_fg+mvnpdf(A(i,1:dim(d)),mu_FG(j,1:dim(d)),cov_FG(1:dim(d),1:dim(d),j))*pi_FG(j);
            end
            if(prob_fg*p_FG>prob_bg*p_BG)
                mask(i)=1;
            end
        end
        mask=reshape(mask,263,248)';
        mask_resized=zeros(255,270);
        mask_resized(1:248,1:263)=mask;
        error_mat_b(c,d)=error(mask_resized);
    end
end

figure;
for i = 1:6
    plot(dim,error_mat_b(i, :), 'LineWidth', 1); 
    hold on; 
end

xlabel('Number of Dimensions');
ylabel('Probability of Error');
title('POE for different number of components');
legend('C=1', 'C=2', 'C=4', 'C=8', 'C=16', 'C=32');
grid on;
hold off;


function [pi_c,mu,cov]=EM(C,sample,numstep)
    % Initialization
    n=size(sample,1);
    pi_c= randi([1,10], C,1);           
    pi_c= pi_c / sum(pi_c);
    mu=sample(randi([1 200],1,C),:);
    cov=zeros(64,64,C);
    for i=1:C
        cov_temp = normrnd(5, 0.3, 1, 64);
        cov(:,:,i) = diag(cov_temp);
    end


    hij=zeros(n,C);
    for step=1:numstep
        %E step
        for i=1:C
            hij(:,i)=mvnpdf(sample,mu(i,:),cov(:,:,i))*pi_c(i);
        end
        hij=hij./sum(hij,2);

        %M step
        pi_c=sum(hij)/n;
        mu=hij'*sample./sum(hij)';

        for c=1:C
            cov(:,:,c)=diag(diag(((sample-mu(c,:))'.*hij(:,c)'*(sample-mu(c,:))./sum(hij(:,c),1))+1e-6));
        end

    end
end


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

function poe=error(image)
    im_test = imread('.../cheetah_mask.bmp');
    im_test=im2double(im_test);
    err=abs(im_test-image);
    poe=sum(err,"all")/(255*270);
end

