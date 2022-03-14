import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
  def __init__(self, in_features, out_features, dropout_p, batch_norm):
      super(FeedForwardBlock, self).__init__()
      self.batch_norm = batch_norm
      
      if batch_norm:
          self.b = nn.BatchNorm1d(in_features)
      self.f = nn.Linear(in_features, out_features)
      self.d = nn.Dropout(dropout_p)
      self.a = nn.ReLU()
      
  def forward(self, x): # the forward pass of info through the net
      if self.batch_norm:
          return self.a(self.d(self.f(self.b(x))))
      else:
          return self.a(self.d(self.f(x)))


class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """
    def __init__(self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=256, dropout_p=0.2):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers_in = [input_dim] + (num_layers-1)*[hidden_dim]
        layers_out = (num_layers-1)*[hidden_dim] + [hidden_dim]
        
        # loop through layers_in and layers_out lists
        self.f = nn.Sequential(*[FeedForwardBlock(in_feats, out_feats, dropout_p, batch_norm) for in_feats, out_feats in zip(layers_in, layers_out)])
        self.clf = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        y = self.f(x)
        return self.clf(y)


