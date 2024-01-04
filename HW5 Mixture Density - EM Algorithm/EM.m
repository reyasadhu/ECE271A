function [pi_c,mu,cov]=EM(C,sample,numstep)
    % Initialization
    n=size(sample,1);
    pi_c= randi([1,10], C,1);           
    pi_c= pi_c / sum(pi_c);% mixture weights must add up to one
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
      
        for i=1:C
            cov(:,:,i)=diag(diag(((sample-mu(i,:))'.*hij(:,i)'*(sample-mu(i,:))./sum(hij(:,i),1))));
            cov(:,:,i)=checkAndFixCovarianceMatrix(cov(:,:,i),10^(-10)); % Covariance matrix should be zero
        end
    
    end
end