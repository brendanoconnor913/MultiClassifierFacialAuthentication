function[LDAdists] = LDA()

ddir = dir('./orl_faces');
trainingdata = []; % Matrix to store all of our images
inc = 1;
% Read through all of the image directories and get the first half of each
% directory (5 of the 10 images)
for k = 3:length(ddir) % discard '.' and '..'
    if (ddir(k).isdir)
        fname = strcat('orl_faces/',ddir(k).name);
        disp(fname);
        imds = imageDatastore(fname);
        % load all images into matrix
        for i = 1:(length(imds.Files)/2)
            % Scale the image and add to our matrix of all images
            m = imresize(double(readimage(imds,i)), .1);
            l = reshape(m, [120,1]);
            trainingdata(:,inc) = l;
            inc = inc + 1;
        end
    end
end

testdata = [];
inc = 1;
for k = 3:length(ddir) % discard '.' and '..'
    if (ddir(k).isdir)
        fname = strcat('orl_faces/',ddir(k).name);
        disp(fname);
        imds = imageDatastore(fname);
        % load all images into 
        for i = (length(imds.Files)/2)+1:(length(imds.Files))
            m = imresize(double(readimage(imds,i)), .1);
            l = reshape(m, [120,1]);
            testdata(:,inc) = l;
            inc = inc + 1;
        end
    end
end

% Make 3d array 1d: each image 2d: each feature 3d: each class
classes = [];
i = 1;
for k = (1:5:size(trainingdata,2))
    classes(:,:,i) = trainingdata(:,k:(k+4))';
    i = i + 1;
end

% Number of observations of each class
num = [];
for i = 1:(size(classes,3))
    num(end+1) = size(classes, 1);
end

%Mean of each class
mu_n = [];
for i = 1:(size(classes,3))
    mu_n(1,:,i) = mean(classes(:,:,i));
end

% Average of the mean of all classes (mean of entire data)
mu_a = mu_n(1,:,1);
for i = 2:(size(classes,3))
    mu_a(1,:) = mu_a(1,:) + mu_n(1,:,i);
end
mu_a = mu_a / (size(classes,3));
    

% Center the data on respective class mean (data-mean)
d_n = [];
for i = 1:(size(classes,3))
    d_n(:,:,i) = classes(:,:,i)-repmat(mu_n(:,:,i),size(classes(:,:,i),1),1);
end

% Calculate the within class scatter for each class
sw_n = [];
for i = 1:(size(classes,3))
    sw_n(:,:,i) = d_n(:,:,i)'*d_n(:,:,i);
end

% Calculate within class scatter for entire data set (SW)
sw_a = sw_n(:,:,1);
for i = 2:(size(classes,3))
    sw_a = sw_a(:,:) + sw_n(:,:,i);
end
invswa = inv(sw_a); % get inverse for use in calculating eigen vector

% Calculate distance from each class mean to all data mean
sb_n = [];
for i = 1:(size(classes,3))
   sb_n(:,:,i) = num(i)*((mu_n(:,:,i)-mu_a(:,:))'*(mu_n(:,:,i)-mu_a(:,:)));
end

% Combine all of these to get between class scatter (SB)
sb_a = sb_n(:,:,1);
for i = 2:(size(classes,3))
    sb_a(:,:) = sb_a(:,:) + sb_n(:,:,i);
end

v = invswa*sb_a;

% find eigen values and eigen vectors of the (v)
[eigvector,eval]=eig(v);
% Sort eigenvalues and vectors descending
eigvalue = diag(eval);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Get eigenvectors with non-zero eigenvalues
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% Calculate proportion of variance from non zero eigenvalues
% and get input from user on how many vectors to use
nonzeig = eigvalue(1:count1);
if (length(nonzeig) > 1)
    results = [];
    tempvecs = [];
    sumnz = sum(nonzeig);
    for i = 1:(length(nonzeig))
        tempvecs(end+1) = nonzeig(i);
        sumt = sum(tempvecs);
        r = sumt/sumnz;
        results(end+1) = r;
    end
    numvec = (1:length(results));
    plot(numvec, results);
    out = 'Enter number of vectors to use: ';
    innum = input(out);
end

vec = eigvector(:,1:innum);
% combine each mean norm. class data into one matrix, normalize on data
% mean
testdata = [];
for i=1:size(d_n, 3)
    for r=1:size(d_n, 1)
        testdata(end+1,:) = classes(r,:,i)-mu_a;
    end
end
% transpose data for projection into fisher space
testdata = testdata';
fdata = vec'*testdata;

% Get mean of transformed data and mean normalize test data with it
% Mean of transformed data
fmean = mean(testdata,2);
testdata = testdata-repmat(fmean,1, size(testdata,2));
% Project test data into fisher space
ftestdata = vec'*testdata;

%  construct matrix to indicate class of each test image
% going to be 1 & 0's

% actuallabels = zeros([40 200]);
% for j=1:200
%     i = ceil(j/5);
%     actuallabels(i,j) = 1;
% end

% Calculate distance of test image to each training image
gendists = zeros([1 1000]);
impostdists = zeros([1 39000]);
for i=1:size(ftestdata,2) % Use each test image
    alldata(1,:) = ftestdata(:,i)'; % Insert testing image
    alldata(2:size(fdata,2)+1,:) = fdata';
    dist = pdist(alldata); 
    for genimg=1:5
        gendists(1,((i-1)*5)+genimg) = dist(((ceil(i/5)-1)*5)+genimg);
    end
    for impimg=6:size(fdata,2)
       impostdists(1,((i-1)*(39*5)+(impimg-5))) = dist(((ceil(i/5)-1)*5)+impimg); 
    end
end
normc = max(max(gendists), max(impostdists)); % normilization constant
LDAdists = [(1-(gendists/normc)) (1-(impostdists/normc)); (gendists/normc) (impostdists/normc)];

