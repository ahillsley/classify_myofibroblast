clear all
close all
clc

%% Normalize Cell Stats
shape_feats = readtable("../Final_things/c_shape_props_pca.csv");
shape_feats = shape_feats(:,[2,3,4,5]);
area = table2array(shape_feats(:,"CellArea"));
norm_area = area / mean(area);
peri = table2array(shape_feats(:,"CellPerimeter"));
norm_peri = peri / mean(peri);
minor = table2array(shape_feats(:,"CellMinorAxisLength"));
norm_minor = minor / mean(minor);
circ = table2array(shape_feats(:,"CellCircularity"));
norm_circ = circ / mean(circ);

feats = cat(2,norm_area, norm_peri,norm_minor, norm_circ);

%% stats PCA
X = table2array(shape_feats);
[coeff,score,latent, tsquared, explained] = pca(feats);

big_table = readtable("../Final_things/Cell_properties_um.csv");
labels = table2cell(big_table(:,"Acti"));
p = gscatter(score(:,1), score(:,2), labels)

%% self super PCA
embed = table2array(readtable("../Final_things/37_byol_emb.csv"));
[scoeff,sscore,slatent, stsquared, sexplained] = pca(embed);
sp = gscatter(sscore(:,1), sscore(:,2))
