
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score,  precision_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, roc_curve
import random
import numpy as np
import os
import random
import numpy as np
def set_random_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

device = torch.device('cuda')

def reset_y_hat(y_hat, y):
    with torch.no_grad():
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1).type(y.dtype)
        else:
            y_hat = (y_hat.view(-1) > 0.5).type(y.dtype)
    return y_hat

def get_pred(y_hat):
    with torch.no_grad():
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = F.softmax(y_hat, dim=-1)
        else:
            y_hat.sigmoid()
    return y_hat


def get_confusion_matrix(y_hat, y):
    with torch.no_grad():
        matrix = torch.zeros((2,2))
        y_hat = reset_y_hat(y_hat, y)
        matrix[0,0] = (y_hat | y == 0).sum()
        matrix[1,1] = (y_hat & y).sum()
        matrix[1,0] = (y_hat - y == 1).sum()
        matrix[0,1] = (y - y_hat == 1).sum()
    return matrix


def get_roc_curve(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] == 2:
        y_hat = y_hat[:,1]
    y, y_hat = y.cpu().numpy(), y_hat.cpu().numpy()
    return roc_curve(y, y_hat)

def get_performance(y_hat = None,y = None, flag = True, matrix = None):
    """
    flag: True , get_roc_auc_score
    matrix: not None, input confusion_matrix
    """
    kwargs = {}
    with torch.no_grad():
        if matrix is None:
            matrix = get_confusion_matrix(y_hat,y)
        TP ,FP ,FN ,TN = matrix[[1,1,0,0],[1,0,1,0]]
        Sn = TP / (TP + FN + 1e-06)
        Sp = TN / (FP + TN + 1e-06)
        Acc = (TP + TN) / (TP + FP + FN + TN + 1e-06)
        MCC = ((TP * TN) - (FP * FN)) / torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)
        kwargs['Sn'] = Sn.item()
        kwargs['Sp'] = Sp.item()
        kwargs['Acc'] = Acc.item()
        kwargs['Mcc'] = MCC.item()
        # 计算AUC
        if flag is True:
            if len(y_hat.shape) > 1 and y_hat.shape[1] == 2:
                y_hat = y_hat[:,1]
            y, y_hat = y.cpu().numpy(), y_hat.cpu().numpy()
            recall = recall_score(y, (y_hat > 0.5))
            precision = precision_score(y, (y_hat > 0.5))
            f1 = f1_score(y, (y_hat > 0.5))
            auc = roc_auc_score(y ,y_hat)
            aupr = average_precision_score(y, y_hat)
            kwargs['Recall'] = recall
            kwargs['Precision'] = precision
            kwargs['F1'] = f1
            kwargs['Auc'] = auc
            kwargs['Aupr'] = aupr
        return kwargs
        
