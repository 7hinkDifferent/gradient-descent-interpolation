from src.registry import objective_function
import torch

@objective_function.register("sigmoid")
def sigmoid():
    return torch.nn.Sigmoid()

@objective_function.register("tanh")
def tanh():
    return torch.nn.Tanh()

@objective_function.register("relu")
def relu():
    return torch.nn.ReLU()