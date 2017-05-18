
gens = [1; 0];
imposts = [0; 1];
targs = [repmat(gens, 1, 1000), repmat(imposts, 1, 39000)];
ldists = LDA();
pdists = PCA();

% average the distance between pca and lda
ave = zeros([2 40000]);
for i=1:size(ave,2)
   ave(1,i) = (ldists(1,i)+pdists(1,i))/2;
   ave(2,i) = 1-ave(1,i);
end

% take min distance between pca and lda
mins = zeros([2 40000]);
for i=1:size(mins,2)
   mins(1,i) = min(ldists(1,i), pdists(1,i));
   mins(2,i) = 1-mins(1,i);
end

% take max distance between pca and lda
maxs = zeros([2 40000]);
for i=1:size(mins,2)
   maxs(1,i) = max(ldists(1,i), pdists(1,i));
   maxs(2,i) = 1-maxs(1,i);
end

plotroc(targs, maxs)
