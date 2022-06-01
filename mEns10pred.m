function pred = mEns10pred(CellArea, CellMinorLength, CellCirc,PredDTree,PredMKNN3, PredMLR4, PredMSVM3, PredMSVM4, PredPKNN, PredPSVM)
    load('tPredAll.mat');    
    tIn = table('VariableNames', trainedEnsemble10.RequiredVariables, ...
          'VariableTypes', {'double', 'double', 'double','double','double',...
          'double','double','double','double','double','double','double',...
          'double','double','double','double','double','double','double',...
          'double','double', 'string','string','string','string','string',...
          'string','string','double', 'double', 'double'}, 'Size', [1 31]);
    tIn.CellArea(1) = CellArea;
    tIn.CellCircularity(1) = CellCirc;
    tIn.CellMinorAxisLength(1) = CellMinorLength;
    tIn.PredDTree(1) = PredDTree;
    tIn.PredMKNN3(1) = PredMKNN3;
    tIn.PredMLR4(1) = PredMLR4;
    tIn.PredMSVM3(1) = PredMSVM3;
    tIn.PredMSVM4(1) = PredMSVM4;
    tIn.PredPKNN(1) = PredPKNN;
    tIn.PredPSVM(1) = PredPSVM;

    pred = char(trainedEnsemble10.predictFcn(tIn));
end