import abc
import torch
import torch.nn as nn

class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """
    abstract base for all models
    provides default weight init, saving/loading, and enforces forward().
    for easy of use, though mostly overridden
    """
    def __init__(self):
        super().__init__()                # initialize nn.Module internals
        self._init_weights()              # run shared weight init

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save(self, path):
        """Save state dict to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        """Load state dict from disk."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)