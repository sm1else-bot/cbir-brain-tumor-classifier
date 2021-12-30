close all
clc
clear all
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);
figure, imshow(I); title('Brain MRI Image');
I = imresize(I,[200,200]);

% Convert to grayscale
gray = rgb2gray(I);

% Otsu Binarization for segmentation
level = graythresh(I);
img = im2bw(I,level);
figure, imshow(img);title('Otsu Thresholded Image');

cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',1);
pixel_labels = reshape(cluster_idx,nrows,ncols);

segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

%
figure, imshow(segmented_images{1});title('Objects in Cluster 1');

seg_img = im2bw(segmented_images{1});
figure, imshow(seg_img);title('Segmented Tumor');
x = double(seg_img);
m = size(seg_img,1);
n = size(seg_img,2);
signal1 = seg_img(:,:);

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
whos DWT_feat
whos G
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

load Trainset.mat
 xdata = meas;
 group = label;
 svmStruct1 = svmtrain(xdata,group,'kernel_function', 'linear');
 species = svmclassify(svmStruct1,feat,'showplot',false)

% To plot classification graphs, SVM can take only two dimensional data
data1   = [meas(:,1), meas(:,2)];
newfeat = [feat(:,1),feat(:,2)];

pause
%close all

svmStruct1_new = svmtrain(data1,group,'kernel_function', 'linear','showplot',false);
species_Linear_new = svmclassify(svmStruct1_new,newfeat,'showplot',false);

%%
% Multiple runs for accuracy highest is 90%
load Trainset.mat
%data   = [meas(:,1), meas(:,2)];
data = meas;
groups = ismember(label,'BENIGN   ');
groups = ismember(label,'MALIGNANT');
[train,test] = crossvalind('HoldOut',groups);
cp = classperf(groups);
%svmStruct = svmtrain(data(train,:),groups(train),'boxconstraint',Inf,'showplot',false,'kernel_function','rbf');
svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
classes = svmclassify(svmStruct,data(test,:),'showplot',false);
classperf(cp,classes,test);
Accuracy_Classification = cp.CorrectRate.*100;
sprintf('Accuracy of Linear kernel is: %g%%',Accuracy_Classification)

%% Accuracy with RBF
svmStruct_RBF = svmtrain(data(train,:),groups(train),'boxconstraint',Inf,'showplot',false,'kernel_function','rbf');
classes2 = svmclassify(svmStruct_RBF,data(test,:),'showplot',false);
classperf(cp,classes2,test);
Accuracy_Classification_RBF = cp.CorrectRate.*100;
sprintf('Accuracy of RBF kernel is: %g%%',Accuracy_Classification_RBF)

%% Accuracy with Polynomial
svmStruct_Poly = svmtrain(data(train,:),groups(train),'Polyorder',2,'Kernel_Function','polynomial');
classes3 = svmclassify(svmStruct_Poly,data(test,:),'showplot',false);
classperf(cp,classes3,test);
Accuracy_Classification_Poly = cp.CorrectRate.*100;
sprintf('Accuracy of Polynomial kernel is: %g%%',Accuracy_Classification_Poly)

%%

% 5 fold cross validation
% 5 fold cross validation
load Normalized_Features.mat
 xdata = norm_feat;
 group = norm_label;
indicies = crossvalind('Kfold',label,5);
cp = classperf(label);
for i = 1:length(label)
    test = (indicies==i);train = ~ test;
    svmStruct = svmtrain(xdata(train,:),group(train),'boxconstraint',Inf,'showplot',false,'kernel_function','rbf');
    classes = svmclassify(svmStruct,xdata(test,:),'showplot',false);
    %class = svmclassify(meas(test,:),meas(train,:),label(train,:));
    classperf(cp,classes,test);
end
Accuracy = cp.CorrectRate;