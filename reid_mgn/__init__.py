from .network import MGN
import torch


def load_MGN(path=None):
    if path:
        saved_model = torch.load(path)
        opt = saved_model['options']
        model = MGN(**opt)
        model.load_state_dict(saved_model['model_state_dict'])
        return model
    else:
        return MGN()
