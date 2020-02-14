# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# **********************************************
# Autoencoder class
# **********************************************
class Autoencoder(nn.Module):
    def __init__(self, inputs, hidden, fe=F.sigmoid, fd=F.sigmoid):
        super(Autoencoder, self).__init__()
        self.le = nn.Linear(inputs, hidden)
        self.ld = nn.Linear(hidden, inputs)
        
        self.fe = fe
        self.fd = fd

    # Forward
    def __call__(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

    # Encoder
    def encoder(self, x):
        return self.fe(self.le(x))

    # Decoder
    def decoder(self, h):
        return self.fd(self.ld(h))

