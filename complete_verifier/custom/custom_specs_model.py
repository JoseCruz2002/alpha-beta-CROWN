#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""An simple example to run custom model and specs."""

import torch
import torch.nn as nn

class SimpleFeedForward(nn.Module):
    """A very simple model, just for demonstration."""
    def __init__(self, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            # Input dimension is 2, should match vnnlib file.
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output dimension is 1, should match vnnlib file.
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs):
        return self.model(inputs)


## -------------------------------------------------------------------------------------------
# ********************* Elsa-Cybersecurity competion model definitions *********************
## -------------------------------------------------------------------------------------------
class FeedForwardNN(nn.Module):

    def __init__(self, n_classes, n_features, hidden_size, layers, **kwargs):
        '''
        n_classes: two, one for malware and other for goodware
        n_features: size of input
        hidden_size: number of neurons in the hidden layers
        hidden_layers: number of hidden layers on the NN 
        '''
        print(f"hidden_size = {hidden_size}; layers = {layers}")
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, hidden_size))
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, n_classes)
        self.activation = nn.functional.relu

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
    
def feedForwardNN(n_classes, n_features, hidden_size, layers):
    return FeedForwardNN(n_classes, n_features, hidden_size, layers)

## -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Save a random model for testing.
    model = SimpleFeedForward(hidden_size=32)
    torch.save(model.state_dict(), 'models/custom_specs/custom_specs.pth')

