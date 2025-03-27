import torch
import torch.nn as nn

class RNNSequenceModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, rnn_type="gru"):
        super(RNNSequenceModel, self).__init__()
        
        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type. Choose from 'gru' or 'lstm'")
        
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x):
        output, _ = self.rnn(x)
        return output[:, -1, :]  # Return last time-step output
