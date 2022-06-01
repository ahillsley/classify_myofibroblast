function pred = mKNN3pred(CellArea, CellPeri, CellMinorLength)
    tIn = [CellArea, 0, 0, 0, 0, CellMinorLength, CellPeri];
    tZ = zeros(1,17);
    tIn = horzcat(tIn, tZ);
    load('tPredAll.mat');
    tIn = array2table(tIn, 'VariableNames', trainedKNN3.RequiredVariables);
    pred = char(trainedKNN3.predictFcn(tIn));
end