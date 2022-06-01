import pandas as pd
import numpy as np


def run_dTree(CellArea, CellPeri, CellMinorLength, CellCirc):
    # Decision Tree implemented from JMP design.
    if CellArea >= 22407:
        return "Activated"
    else:
        if CellPeri >= 1104.117928:
            if CellMinorLength >= 181.413255:
                return "Activated"
            else:
                return "Not"
        else:
            if CellPeri < 757.15:
                return "Not"
            else:
                if CellCirc < 0.254725636:
                    return "Not"
                else:
                    return "Unknown"


def run_kNN(knn, CellArea, CellPeri, CellMinorLength, CellCirc):
    # kNN trained by sklearn.
    t = np.asarray([CellArea, CellPeri, CellMinorLength, CellCirc])
    if int(knn.predict(t.reshape(1, -1))) == 0:
        return "Not"
    else:
        return "Activated"
