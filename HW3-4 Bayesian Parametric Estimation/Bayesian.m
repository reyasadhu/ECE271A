load('.../TrainingSamplesDCT_subsets_8.mat');
load('.../Alpha.mat');

train_BG = D1_BG;
train_FG = D1_FG;
helper(1,train_BG,train_FG,alpha,1);
helper(1,train_BG,train_FG,alpha,2);

train_BG = D2_BG;
train_FG = D2_FG;
helper(2,train_BG,train_FG,alpha,1);
helper(2,train_BG,train_FG,alpha,2);

train_BG = D3_BG;
train_FG = D3_FG;
helper(3,train_BG,train_FG,alpha,1);
helper(3,train_BG,train_FG,alpha,2);

train_BG = D4_BG;
train_FG = D4_FG;
helper(4,train_BG,train_FG,alpha,1);
helper(4,train_BG,train_FG,alpha,2);

function []= helper(Data, BG, FG, alpha, strategy)

    p=load(['.../Prior_',num2str(strategy),'.mat']);
    w0=p.W0;
    mu0_FG=p.mu0_FG;
    mu0_BG=p.mu0_BG;

    % ML Priors
    Prior_FG=size(FG,1)/(size(FG,1)+size(BG,1));
    Prior_BG=size(BG,1)/(size(FG,1)+size(BG,1));

    mu_FG=mean(FG,1);
    mu_BG=mean(BG,1);

    FG_cov = cov(FG) * (size(FG, 1) - 1) / size(FG, 1);
    BG_cov = cov(BG) * (size(BG, 1) - 1) / size(BG, 1);

    A_ML= zeros((size(I, 1) - 7),(size(I, 2) - 7));

    Error_bayesian = zeros(1,9);
    Error_ML = zeros(1,9);
    Error_MAP = zeros(1,9);
    

    for k=1:length(alpha)

        cov_0=diag(w0)*alpha(k);

        N_FG=size(FG,1);
        sample_cov_FG=FG_cov/N_FG;
        mu_1_FG=cov_0/(cov_0+sample_cov_FG)*mu_FG'+sample_cov_FG/(cov_0+sample_cov_FG)*mu0_FG';
        cov_1_FG=cov_0/(cov_0+sample_cov_FG)*sample_cov_FG;

        N_BG=size(BG,1);
        sample_cov_BG=BG_cov/N_BG;
        mu_1_BG=cov_0/(cov_0+sample_cov_BG)*mu_BG'+sample_cov_BG/(cov_0+sample_cov_BG)*mu0_BG';
        cov_1_BG=cov_0/(cov_0+sample_cov_BG)*sample_cov_BG;
        
       
       
        %Predictive Distribution
        mu_pred_FG=mu_1_FG;
        mu_pred_BG=mu_1_BG;
        cov_pred_FG=cov_1_FG+FG_cov;
        cov_pred_BG=cov_1_BG+BG_cov;

        % Bayesian

        I = imread('.../cheetah.bmp');
        I=im2double(I);
        A_Bayesian = zeros((size(I, 1) - 7),(size(I, 2) - 7));
        A_MAP = zeros((size(I, 1) - 7),(size(I, 2) - 7));

        for i = 1:size(I, 1) - 7
            for j = 1:size(I, 2) - 7

                block = I(i:i+7, j:j+7);
                block_dct = dct2(block);
                dct_flat=zigzaged(block_dct);

                %Bayesian
                alp_FG=log(((2*pi)^64)*det(cov_pred_FG))-2*log(Prior_FG);
                alp_BG=log(((2*pi)^64)*det(cov_pred_BG))-2*log(Prior_BG);
                g_cheetah=1/(1+exp(dxy(dct_flat',mu_pred_FG,cov_pred_FG)-dxy(dct_flat',mu_pred_BG,cov_pred_BG)+alp_FG-alp_BG));
                if g_cheetah>0.5
                    A_Bayesian(i+3,j+3)=1;
                end

                %ML
                if k==1
                    alp_FG=log(((2*pi)^64)*det(FG_cov))-2*log(Prior_FG);
                    alp_BG=log(((2*pi)^64)*det(BG_cov))-2*log(Prior_BG);
                    g_cheetah=1/(1+exp(dxy(dct_flat',mu_FG',FG_cov)-dxy(dct_flat',mu_BG',BG_cov)+alp_FG-alp_BG));
                    if g_cheetah>0.5
                        A_ML(i+3,j+3)=1;
                    end
                end

                %MAP
                alp_FG=log(((2*pi)^64)*det(FG_cov))-2*log(Prior_FG);
                alp_BG=log(((2*pi)^64)*det(BG_cov))-2*log(Prior_BG);
                g_cheetah=1/(1+exp(dxy(dct_flat',mu_pred_FG,FG_cov)-dxy(dct_flat',mu_pred_BG,BG_cov)+alp_FG-alp_BG));
                if g_cheetah>0.5
                    A_MAP(i+3,j+3)=1;
                end

            end
        end

        %Padding
        A_Bayesian_resized=zeros(255,270);
        if k==1
            A_ML_resized=zeros(255,270);
        end
        A_MAP_resized=zeros(255,270);
        for i=4:251
            for j=4:266
                A_Bayesian_resized(i,j)=A_Bayesian(i-3,j-3);
                if k==1
                    A_ML_resized(i,j)=A_ML(i-3,j-3);
                end
                A_MAP_resized(i,j)=A_MAP(i-3,j-3);
            end
        end

        Error_bayesian(k)=error(A_Bayesian_resized);
        Error_ML(k)=error(A_ML_resized);
        Error_MAP(k)=error(A_MAP_resized);
        


    end

    fig=figure;
    semilogx(alpha, Error_bayesian, '-r',alpha, Error_ML, '-g',alpha, Error_MAP, '-b');
    grid
    %ylim([0.1460 0.1500])
    legend('Bayesian','ML','MAP')
    xlabel('alpha');
    ylabel('probability of error');
    title(['Dataset ',num2str(Data),' and Strategy ',num2str(strategy)]);
    print(fig,'-djpeg',sprintf(".../Dataset %d strategy %d.jpg",Data,strategy));
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

function output=dxy(x,y,cov)
    output=transpose(x-y)/cov*(x-y);
end

function prob_err=error(input)
    im_test = imread('.../cheetah_mask.bmp');
    im_test=im2double(im_test);
    err=abs(im_test-input);
    prob_err=sum(err,"all")/(255*270);
end

