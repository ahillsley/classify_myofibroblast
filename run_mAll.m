%to fill in the matlab predictor methods
fileName = "CellProperties13.csv";
oCSV = readtable(fileName);
load('G:\Shared drives\000_College\Rosales Lab\Code\Complete Codes\tPredAll.mat');
s = size(oCSV);
i = 1;
cNew = cell(s(1,1), 5);
while i < s(1,1)+1
%     disp(oCSV.Name(i,1));
    CellArea = oCSV.CellArea(i);
    CellPeri = oCSV.CellPerimeter(i);
    CellMinorLength = oCSV.CellMinorAxisLength(i);
    CellCirc = oCSV.CellCircularity(i);

    cNew(i,1) = {mKNN3pred(CellArea, CellPeri, CellMinorLength)};   
    cNew(i,2) = {mSVM3pred(CellArea, CellPeri, CellMinorLength)};
    cNew(i,3) = {mSVM4pred(CellArea, CellPeri, CellMinorLength, CellCirc)};
    cNew(i,4) = {mLR4pred(CellArea, CellPeri, CellMinorLength, CellCirc)};
%     cNew(i,5) = {mEns10pred(CellArea, CellMinorLength, CellCirc, oCSV.PredDTree{i,1},...
%                 cNew(i,1), cNew(i,4), cNew(i,2), cNew(i,3),...
%                 oCSV.PredPKNN{i,1}, oCSV.PredPSVM{i,1})};
    i = i+1;
end
% oCSV.PredEns10 = cNew(:,5);
oCSV.PredMLR4 = cNew(:,4);
oCSV.PredMSVM4 = cNew(:,3);
oCSV.PredMSVM3 = cNew(:,2);
oCSV.PredMKNN3 = cNew(:,1);

writetable(oCSV, fileName);
%----------------------------------
% confusionchart(oCSV.Acti, cNew(:,1))
% confusionchart(oCSV.Acti, cNew(:,2))
% confusionchart(oCSV.Acti, cNew(:,3))
% confusionchart(oCSV.Acti, cNew(:,4))
%tActi = 181
%tNot = 380
% fActi = 13
%fNot = 21

%Accuracy = 0.94285
%Precision = 0.93299

%-----------------------------------
% i = 1;
% eTable = table();
% while i < s(1,1)+1
%     if strcmp(cNew{i}, oCSV.Acti{i})
%         i;
%     else
%         cRow = oCSV(i, :);
%         eTable = [eTable; cRow];
%     end
%     i = i+1;
% end
% writetable(eTable, 'CellProperties12_UMAP_v2f_errors.csv');        