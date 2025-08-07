from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_auc_roc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def calculate_preference_alignment(y_true, y_pred):
    high_ratings = y_true >= 4
    return np.corrcoef(y_pred, high_ratings)[0, 1]