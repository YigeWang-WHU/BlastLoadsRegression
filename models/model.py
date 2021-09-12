import torch.nn as nn
import torch
class RegressionModel(nn.Module):

    def __init__(self, input_size, num_h, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        # the length of hidden_size and num_h should match
        # for eg. length of hidden size [5, 10] and num_h should both be 2
        assert len(hidden_size) == num_h
        # Add input_size and output_size to the list, so that we can construct model accordingly
        #for eg, if there are 3 inputs and 4 outputs, the resulting list will be [3 ,5, 10, 4]
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)
        # the list for storing each layer
        modules = []
        # Append model layers sequentially
        for l in range(len(hidden_size)-1):
            # add fully connected layer
            modules.append(nn.Linear(hidden_size[l], hidden_size[l+1]))
            # skip adding activation function for the output
            if l == len(hidden_size)-2: break # whether activate at the last layer
            # activation function
            modules.append(nn.ReLU())
        # make the the list of layer to a pytorch model
        self.model = nn.Sequential(*modules)
        # apply model weights initialization function
        self.model.apply(self.init_weights)
    
    @staticmethod
    # function for initializing the weights of the model
    def init_weights(m):
        if type(m) == nn.Linear:
            # initilize the weights
            torch.nn.init.xavier_uniform(m.weight)
            # initialize the bias
            m.bias.data.fill_(0.01)
    # the forward pass
    def forward(self, X):

        out = self.model(X)

        return out
