import torch.nn as nn
import torch
class RegressionModel(nn.Module):

    def __init__(self, input_size, num_h, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        assert len(hidden_size) == num_h
        # Add input_size and output_size to the list
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)
        modules = []
        # Append model elements sequentially
        for l in range(len(hidden_size)-1):
            modules.append(nn.Linear(hidden_size[l], hidden_size[l+1]))
            if l == len(hidden_size)-2: break
            modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules)
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, X):

        out = self.model(X)

        return out
