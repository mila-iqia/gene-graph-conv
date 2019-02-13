from models.models import Model
from models.utils import save_computations
from torch import nn

class LR(Model):
    def __init__(self, **kwargs):
        super(LR, self).__init__(**kwargs)

    def setup_layers(self):
        self.nb_nodes = len(self.X.keys())
        self.in_dim = 1
        self.out_dim = 2

        # The logistic layer.
        logistic_in_dim = self.nb_nodes * self.in_dim
        logistic_layer = nn.Linear(logistic_in_dim, self.out_dim)
        logistic_layer.register_forward_hook(save_computations)
        self.my_logistic_layers = nn.ModuleList([logistic_layer])

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(nb_examples, -1)
        x = self.my_logistic_layers[-1](x)
        return x
