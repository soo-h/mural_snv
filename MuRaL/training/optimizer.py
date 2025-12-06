import torch
import sys

def construct_params(model, learning_rates):
    """Helper function to construct parameter groups with learning rates."""
    if isinstance(learning_rates, list):
        if len(learning_rates) == 3:
            return [
                {'params': filter(lambda p: p.requires_grad, model.local_scale_model.parameters()), 'lr': learning_rates[0]},
                {'params': filter(lambda p: p.requires_grad, model.middle_scale_model.parameters()), 'lr': learning_rates[1]},
                {'params': filter(lambda p: p.requires_grad, model.large_scale_model.parameters()), 'lr': learning_rates[2]}
            ]
        elif len(learning_rates) == 1:
            return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': learning_rates[0]}]
        else:
            raise ValueError(f"Expected 1 or 3 learning rates, but got {len(learning_rates)}.")
    else:
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': learning_rates}]


def get_optimizer(optim_name,
                  model,
                  learning_rates,
                  weight_decay):
    
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }
    params = construct_params(model, learning_rates)
    optimizer = optimizers[optim_name](params, weight_decay=weight_decay)
    return optimizer

def get_weight_decay(batch_size, epochs, train_size, weight_decay_auto, weight_decay):
    if weight_decay_auto is not None and weight_decay_auto > 0:
        if weight_decay_auto >= 1:
            sys.exit("Error: Please set a value smaller than 1 for --weight_decay_auto.")
        weight_decay_auto = 1- weight_decay_auto **(batch_size/(epochs*train_size))
    else:
        weight_decay_auto = weight_decay
        
    return weight_decay_auto



def get_lr_scheduler(scheduler_name, optimizer, train_size, config, printer=print):
    """
    Returns the learning rate scheduler based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary with necessary parameters.
        optimizer (Optimizer): The optimizer to attach the scheduler to.
        train_size (int): Total size of the training dataset.
    
    Returns:
        scheduler: A PyTorch learning rate scheduler.
    """
    batch_size = config['batch_size']

    # Pre-calculate commonly used values
    step_size = (5000 * 128) // batch_size
    gamma_rop = (config['min_lr'] / config['restart_lr'])**(1 / (train_size // batch_size))
    
    # Dictionary mapping for scheduler initialization
    schedulers = {
        'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=config['LR_gamma']),
        'StepLR2': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma_rop),
        'ROP': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=1, threshold=0.0001, min_lr=1e-7)
    }

    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler name '{scheduler_name}'")

    # Log the scheduler configuration
    log_scheduler_info(scheduler_name, config, gamma_rop, printer)
    
    return schedulers[scheduler_name]

def log_scheduler_info(scheduler_name, config, gamma=None, printer=print):
    """
    Log the learning rate scheduler configuration.
    
    Args:
        scheduler_name (str): Name of the scheduler.
        config (dict): Configuration dictionary.
        gamma (float, optional): Gamma value for StepLR2, if applicable.
    """
    printer(f"Using learning rate scheduler: {scheduler_name}")
    if scheduler_name == 'StepLR2' and gamma is not None:
        printer(f"Learning rate gamma for StepLR2: {gamma}")
    if scheduler_name == 'ROP':
        printer("Using lr_scheduler.ReduceLROnPlateau with patience:", config['patience'])

