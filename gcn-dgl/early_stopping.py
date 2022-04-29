import numpy as np

class EarlyStopping(object):
    """This class implements early stopping strategy based on params passed.
    patience is the minimum interval without any change in average delta before stop
    min_delta is the average metric threshold for early stop
    """
    def __init__(self, patience=10, min_delta=0.0005):
        self.min_delta = min_delta
        self.patience = patience
        self.metric_list = []
    
    def step(self, metric):
        self.metric_list.append(metric)
        
        if(len(self.metric_list) < self.patience):
            return False
        
        avg_metric_in_patience_interval = self.metric_list[-1] - np.mean(self.metric_list[-(self.patience+1):-1])
        if np.abs(avg_metric_in_patience_interval) < self.min_delta:
            return True