% Florian Meyer, 2012

function [ covarianceOut ] = checkAndFixCovarianceMatrix( covarianceIn, minStepSize )
covarianceOut = covarianceIn;

% make sure covarianceOut is real
if(~isreal(covarianceOut))
    covarianceOut = real(covarianceOut);
end

% make sure covarianceOut has no NaNs
if(any(isnan(covarianceOut(:))))
    covarianceOut(isnan(covarianceOut)) = 0;
end

% make sure covarianceOut is symmetric, positive semidefinite, and well conditioned
if(~issymmetric(covarianceOut) || any(eig(covarianceOut) < 0) || rcond(covarianceOut) < 10^(-15))
    covarianceOut = covarianceOut*covarianceOut';
    while(any(eig(covarianceOut) < 0) || rcond(covarianceOut) < 10^(-15))
        stepSize = abs(max(eig(covarianceOut))/10^(15));
        if(stepSize==0)
            stepSize = minStepSize;       
        end
        covarianceOut = covarianceOut+stepSize*eye(size(covarianceIn,1));
    end
    covarianceOut = sqrtm(covarianceOut);
end

end