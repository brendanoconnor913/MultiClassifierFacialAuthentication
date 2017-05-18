
gens = [1; 0];
imps = [0; 1];
targs = [repmat(gens, 2, 200) repmat(imps, 2, 39000)];
ldists = LDA();
pdists = PCA();

% LDA Fusion
ldafus = zeros([2 200]);
% Genuine averages
finalindex = 1;
for i=1:5:1000
   ave = 0;
   % average the results for each class distance comparison
   for fi=0:4
       ave = ave + ldists(1, i+fi);
   end
   ave = ave/5;
   ldafus(1,finalindex) = ave;
   ldafus(2,finalindex) = 1-ave;
   finalindex = finalindex + 1;
end

% PCA Fusion
pcafus = zeros([2 200]);
% Genuine averages
finalindex = 1;
for i=1:5:1000
   ave = 0;
   % average the results for each class distance comparison
   for fi=0:4
       ave = ave + pdists(1, i+fi);
   end
   ave = ave/5;
   pcafus(1,finalindex) = ave;
   pcafus(2,finalindex) = 1-ave;
   finalindex = finalindex + 1;
end
% get imposter distance comparisons for plotting
limpos = ldists(:,1001:40000);
pimpos = pdists(:,1001:40000);

plotroc(targs, [ldafus limpos; pcafus pimpos], 'fused')
