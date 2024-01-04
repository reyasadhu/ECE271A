% Read the training data
load('.../TrainingSamplesDCT_8.mat');
numCheetahSamples = size(TrainsampleDCT_FG, 1);
numGrassSamples = size(TrainsampleDCT_BG, 1);


totalSamples = numCheetahSamples + numGrassSamples;

% ML estimate of class priors
PY_cheetah = numCheetahSamples / totalSamples;
PY_grass = numGrassSamples / totalSamples;


disp(['PY(cheetah) = ' num2str(PY_cheetah)]);
disp(['PY(grass) = ' num2str(PY_grass)]);

cheetahSecondLargestIndices = zeros(size(TrainsampleDCT_FG, 1), 1);
grassSecondLargestIndices = zeros(size(TrainsampleDCT_BG, 1), 1);

% For each sample in training data, get the index of second maximum energy
for row = 1:size(TrainsampleDCT_FG, 1)
    absValuesf = abs(TrainsampleDCT_FG(row, :));
    [~, sortedIndicesf] = maxk(absValuesf,2);
    cheetahSecondLargestIndices (row) = sortedIndicesf(2);
end
for row = 1:size(TrainsampleDCT_BG, 1)
    absValuesb = abs(TrainsampleDCT_BG(row, :));
    [~, sortedIndicesb] = maxk(absValuesb, 2);
    grassSecondLargestIndices (row) = sortedIndicesb(2);
end


cheetahHistogram = hist(cheetahSecondLargestIndices, 1:64);
grassHistogram = hist(grassSecondLargestIndices, 1:64);

%ML estimate of class conditional probabilities
cheetahPDF = cheetahHistogram / sum(cheetahHistogram);
grassPDF = grassHistogram / sum(grassHistogram);

% Plot the histograms
figure;
subplot(2, 1, 1);
bar(1:64, cheetahPDF);
title('PX|Y(x|cheetah)');
xlabel('Index');
ylabel('Probability Density');

subplot(2, 1, 2);
bar(1:64, grassPDF);
title('PX|Y(x|grass)');
xlabel('Index');
ylabel('Probability Density');

zigzag=importdata('.../Zig-Zag Pattern.txt') ;
zigzag_flat=reshape(zigzag.',1,[])
disp(zigzag_flat);

I = imread('.../cheetah.bmp');
I=im2double(I);

A = zeros(size(I, 1) - 7,size(I, 2) - 7);

% Calculate DCT for each block and use BDR to assign class to the central pixel
% of the 8*8 block
for i = 1:size(I, 1) - 7
    for j = 1:size(I, 2) - 7
        row_start = i;
        row_end = i + 7;
        col_start = j;
        col_end = j + 7;
        block = I(row_start:row_end, col_start:col_end);
        
        block_dct = dct2(block);
        block_dct_flat = reshape(block_dct.',1,[]);
        [~, index] = maxk(abs(block_dct_flat), 2);
        
        if PY_cheetah * cheetahPDF(zigzag_flat(index(2))+1) >= PY_grass * grassPDF(zigzag_flat(index(2))+1)
            % Assigning it to the central pixel of the block
            A(i+3,j+3)=1 ;
        else
           A(i+3,j+3)= 0;
        end
    end
end

% Padding the image with zeros
A_resized=zeros(255,270);
for i=5:252
    for j=5:267
        A_resized(i,j)=A(i-4,j-4);
    end
end



figure;
imshow(A_resized);

% import the given mask to compare
im_test = imread('.../cheetah_mask.bmp');
im_test=im2double(im_test);
err=abs(im_test-A_resized);
prob_err=sum(err,"all")/(255*270);
disp(prob_err);