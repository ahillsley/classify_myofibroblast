function pred = mLR4pred(CellArea, CellPeri, CellMinorLength, CellCirc)
    tIn = [CellArea, CellCirc, 0, 0, 0, CellMinorLength, CellPeri];
    tZ = zeros(1,17);
    tIn = horzcat(tIn, tZ);
    load('tPredAll.mat');
    tIn = array2table(tIn, 'VariableNames', trainedLR4.RequiredVariables);
    pred = char(trainedLR4.predictFcn(tIn));
end