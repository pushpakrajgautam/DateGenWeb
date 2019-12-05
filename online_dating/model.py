import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as func
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super(baselineLSTM, self).__init__()
        if config['cuda'] and torch.cuda.is_available():
            self.torch_device = torch.device('cuda')
        else:
            self.torch_device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size=config['input_dim'], 
                            hidden_size=config['hidden_dim'], 
                            num_layers=config['layers'], 
                            dropout=config['dropout']).to(self.torch_device)
        self.output = nn.Linear(config['hidden_dim'], config['output_dim']) .to(self.torch_device)
        self.temperature = config['gen_temp']
        self.batch_size = config['batch_size'] 
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.layers = config['layers']
        self.config = config
        
    def zero_hidden(self):
        return (torch.zeros(self.layers, self.batch_size, self.hidden_dim).to(self.torch_device),
                torch.zeros(self.layers, self.batch_size, self.hidden_dim).to(self.torch_device))

    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        sequence, self.hidden_state = self.lstm(sequence, self.hidden_state)
        sequence = self.output(sequence)
        return sequence
