"""Gradient inspection and parameter counting utilities."""

import torch
import torch.nn as nn
from prettytable import PrettyTable


def count_parameters(model):
    """Count parameters in a network model."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def check_gradients(module, loss):
    """Check whether gradients for a given module are non-zero."""
    named_parameters = list(module.named_parameters())
    gradients = [param.grad for _, param in named_parameters if param.grad is not None]
    if not gradients:
        print(f"No grad to used. Loss is {loss}!")
        return


def print_gradient_norms(model, print=print):
    """Compute L2 norm of gradients."""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    print(f"Gradient L2 norms: total {total_norm}")


def print_gradients(model, print=print):
    """Print layer-wise gradient distribution."""
    print("Layer-wise Gradient Distribution:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name} - mean: {grad.mean().item():.6f}, std: {grad.std().item():.6f}, "
                  f"min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
        else:
            print(f"{name} - No gradient computed")


def hook_backward_function(module, input_grad, output_grad):
    print("distal_module output grad:", output_grad)
