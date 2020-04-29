import torch
import torch.nn

def sample_wise_kl(train_samples, val_samples):
    
    n_train_samples = train_samples.size()[0]
    n_val_samples = train_samples.size()
    
    losses = []
    for i in range(n_train_samples):
        repeated_train_samples = train_samples[i].unsqueeze(0).repeat(n_val_samples, 1)
        losses.append(torch.nn.functional.kl_div(repeated_train_samples, val_samples))
    
    return torch.sum(losses) / len(losses)
        