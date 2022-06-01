function pred = mSVM3pred(CellArea, CellPeri, CellMinorLength)
    tIn = [CellArea, 0, 0, 0, 0, CellMinorLength, CellPeri];
    tZ = zeros(1,17);
    tIn = horzcat(tIn, tZ);
    load('tPredAll.mat');
    tIn = array2table(tIn, 'VariableNames', trainedSVM3.RequiredVariables);
    pred = char(trainedSVM3.predictFcn(tIn));
end