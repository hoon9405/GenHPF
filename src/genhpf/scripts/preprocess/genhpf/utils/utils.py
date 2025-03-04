import numpy as np
import pandas as pd


def col_name_add(x, cate_col):
    if not (x == "nan" or x == pd.isnull(x)):
        return cate_col + "_" + str(x)
    else:
        return x  # return nan


def q_cut(x, cuts):
    unique_var = len(np.unique([i for i in x]))
    nunique = len(pd.qcut(x, min(unique_var, cuts), duplicates="drop").cat.categories)
    output = pd.qcut(x, min(unique_var, cuts), labels=range(1, min(nunique, cuts) + 1), duplicates="drop")
    return output
