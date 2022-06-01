'''Comparing Prediction Methods'''
import pandas as pd
import csv

oCSV = pd.read_csv("CellProperties13.csv")


cols = ['PredDTree', 'PredPKNN', 'PredMKNN3',
                 'PredMSVM3', 'PredMSVM4', 'PredMLR4', 'PredEns10', 'Acti']
pred_res = oCSV[cols].values

pred_res = pd.DataFrame(pred_res, columns = ['PredDTree', 'PredPKNN', 'PredMKNN3',
                                             'PredMSVM3', 'PredMSVM4',
                                             'PredMLR4', 'PredEns10', 
                                             'zActi'])
dMetrics = dict()
dMetrics["Method"] = ["tActi", "tNot", "fActi", "fNot",
                      "Accuracy", "Precision Activated", "Precision Not Activated"]
for col in pred_res:
    tActi, tNot, fActi, fNot, i = 0,0,0,0,0
    while i < len(oCSV):
        if pred_res[col][i] == pred_res["zActi"][i]:
            if pred_res["zActi"][i] == "Activated":
                tActi+=1
            else:
                tNot+=1
        else:
            if pred_res[col][i] == "Activated":
                fActi+=1
            else:
                fNot+=1
        i+=1
    cAcc = (tActi+tNot)/len(oCSV)
    cPreA = tActi/(tActi+fActi)
    cPreN = tNot/(tNot+fNot)
    dMetrics[col] = [tActi, tNot, fActi, fNot, cAcc, cPreA, cPreN]

for i in dMetrics:
    print(i, dMetrics[i])
    
with open('predictor_Metrics.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in dMetrics.items():
       writer.writerow([key, value[0], value[1], value[2], value[3], value[4],
                        value[5], value[6]])