import torch
from torch import nn

class DigitalNN(nn.Module):
    """ Partial model for digital transaction domain
    Args:
        input_size (int): number of features in digital transaction domain
        digital_transaction_intput (tensor): input size of digital transaction domain
    
    """
    def __init__(
            self, 
            input_size: int,
            hidden_size: int = 32,
            output_size: int = 16,
        ):
        super().__init__()

        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, digital_input):
        
        for layer in self.layers_stack:
            digital_input = layer(digital_input)
        
        return digital_input
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()

class RetailNN(nn.Module):
    """ Partial model for retail transaction domain

    Args:
        input_size (int): number of features in retail transaction domain
        retail_transaction_intput (tensor): input size of retail transaction domain
    
    """
    def __init__(
            self, 
            input_size: int,
            hidden_size: int = 32,
            output_size: int = 16,
        ):
        super().__init__()

        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, retail_input):
        
        for layer in self.layers_stack:
            retail_input = layer(retail_input)
        
        return retail_input
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()
    
class FraudPrevNN(nn.Module):
    """ Partial model for fraud prevention domain

    Args:
        input_size (int): number of features in fraud prevention domain
        fraud_prev_intput (tensor): input size of fraud prevention domain
    
    """
    def __init__(
            self, 
            input_size: int,
            hidden_size: int = 32,
            output_size: int = 16,
        ):
        super().__init__()

        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, fraud_prev_input):
        
        for layer in self.layers_stack:
            fraud_prev_input = layer(fraud_prev_input)
        
        return fraud_prev_input
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()