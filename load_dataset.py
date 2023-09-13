import torch
from torch.utils import data
import os
from typing import Optional, Union, Any
from torch.nn import functional as F
import pandas as pd

class iDataSet(data.Dataset):
    """ build dataset """
    def __init__(self, data, feature_encoding, label:Union[int, float] , name = '') -> None:
        super(iDataSet, self).__init__()
        self.feature_encoding = feature_encoding
        self.label = label
        self.data = data
        self.name = name
    
    def __getitem__(self, index) -> Any:
        return self.feature_encoding(self.data[index]), self.label
    def __len__(self):
        return len(self.data)


class GetDataSet:
    """
    return datasets: train_pos, train_neg, test_data
    """
    def __init__(self, data_path, handle, dataset_name = '') -> None:
        assert data_path is not None
        files_name = ["train_pos.txt","train_neg.txt","test_pos.txt","test_neg.txt"]
        self.name = dataset_name
        self.data = []
        for name in files_name:
            path = os.path.join(data_path, name)
            with open(path, "r") as f:
                self.data.append(handle(f))
        
    def __call__(self, feature_encoding, type:Optional[int] = None) -> Any:
        """
        type: 
        0 : train_data (pos , neg),
        1 : test_data,
        default :  train_data(pos + neg)
        """

        if type is None:
            return iDataSet(self.data[0],feature_encoding,1, self.name) + iDataSet(self.data[1], feature_encoding, 0 , self.name)
        elif type == 0:
            return iDataSet(self.data[0],feature_encoding,1, self.name), iDataSet(self.data[1], feature_encoding, 0, self.name)
        elif type == 1:
            return iDataSet(self.data[2], feature_encoding, 1, self.name) + iDataSet(self.data[3], feature_encoding,0, self.name)
        
    
class iFunction:
    
    def __init__(self) -> None:
        """"""

    def to_number(seq):
        base_dict = {
            'A':0, 'C':1, 'G':2, 'T':3
        }
        return torch.LongTensor([base_dict[c] for c in seq]).float().unsqueeze(-1)

    def to_eiip(seq):
        base_dict = {
            'A':0.1260,'T':0.1335,'C':0.1340,'G':0.0806
        }
        return torch.tensor([base_dict[c] for c in seq]).float().unsqueeze(-1)
    
    
    def to_ncp(seq):
        base_dict = {
            'A':[1,1,1],'T':[0,0,1],'C':[0,1,0],'G':[1,0,0]
        }
        return torch.tensor([base_dict[c] for c in seq]).float()
    
    def to_nd(seq):
        count_dict = {
            'A':0,'T':0,'C':0,'G':0
        }
        res = []
        for i, (c,) in enumerate(seq):
            count_dict[c] += 1
            res.append(count_dict[c] / (i + 1))
        return torch.tensor(res).float().unsqueeze(-1)
    

    def to_one_hot(seq):
        '''独热编码'''
        base_dict = {
            'A':0, 'C':3, 'G':2, 'T':1
        }
        return F.one_hot(torch.tensor([base_dict.get(c,4) for c in seq]),4).float()

     
    def fe1(seq):
        x1 = iFunction.to_one_hot(seq)
        x2 = iFunction.to_eiip(seq)
        x3 = iFunction.to_ncp(seq)
        x4 = iFunction.to_nd(seq)
        return torch.cat([x1,x2,x3,x4], dim=-1)

    def fe_one_eiip_ncp(seq):
        x1 = iFunction.to_one_hot(seq)
        x2 = iFunction.to_eiip(seq)
        x3 = iFunction.to_ncp(seq)
        return torch.cat([x1,x2,x3], dim=-1)
    
    def fe_one_ncp_nd(seq):
        x1 = iFunction.to_one_hot(seq)
        x2 = iFunction.to_ncp(seq)
        x3 = iFunction.to_nd(seq)
        return torch.cat([x1,x2,x3], dim=-1)

    def fe_one_eiip_nd(seq):
        x1 = iFunction.to_one_hot(seq)
        x2 = iFunction.to_nd(seq)
        x3 = iFunction.to_eiip(seq)
        return torch.cat([x1,x2,x3], dim=-1)
    
    def fe_eiip_ncp_nd(seq):
        x1 = iFunction.to_ncp(seq)
        x2 = iFunction.to_nd(seq)
        x3 = iFunction.to_eiip(seq)
        return torch.cat([x1,x2,x3], dim=-1)


    
    def read_txt(f):
        return [line.strip() for line in f.readlines()]
    
    def read_txt_to_pd2(f):
        return pd.read_csv(f)['data']

    def read_txt_to_pd(f):
        return pd.read_table(f, header=None)[0]