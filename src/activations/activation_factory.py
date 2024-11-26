from src.registry import objective_function
import torch

@objective_function.register("sigmoid")
def sigmoid(*args, **kwargs):
    return torch.nn.Sigmoid()

@objective_function.register("tanh")
def tanh(*args, **kwargs):
    return torch.nn.Tanh()

@objective_function.register("relu")
def relu(*args, **kwargs):
    return torch.nn.ReLU()

@objective_function.register("sine")
def sin(*args, **kwargs):
    return torch.sin