import torch
import torch.nn.functional as F

def compute_reconstruction_error(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Compute mean squared error between input and reconstruction.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        x_recon: Reconstructed tensor of shape (batch_size, input_dim)
        
    Returns:
        Mean squared error
    """
    return F.mse_loss(x_recon, x).item()

def compute_free_energy(x: torch.Tensor, energy: torch.Tensor) -> float:
    """Compute free energy of the model.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        energy: Energy values of shape (batch_size,)
        
    Returns:
        Mean free energy
    """
    return energy.mean().item()

def compute_log_likelihood(x: torch.Tensor, log_probs: torch.Tensor) -> float:
    """Compute log likelihood of the data.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        log_probs: Log probabilities of shape (batch_size,)
        
    Returns:
        Mean log likelihood
    """
    return log_probs.mean().item()

def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute KL divergence between two distributions.
    
    Args:
        p: First distribution of shape (batch_size, num_classes)
        q: Second distribution of shape (batch_size, num_classes)
        
    Returns:
        Mean KL divergence
    """
    # Ensure distributions are normalized
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    
    # Compute KL divergence
    kl = (p * torch.log(p / q)).sum(dim=1)
    return kl.mean().item()

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.
    
    Args:
        predictions: Predicted labels of shape (batch_size,)
        targets: Target labels of shape (batch_size,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    return (predictions == targets).float().mean().item()

def compute_precision_recall(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Compute precision and recall.
    
    Args:
        predictions: Predicted labels of shape (batch_size,)
        targets: Target labels of shape (batch_size,)
        
    Returns:
        Tuple of (precision, recall)
    """
    true_positives = ((predictions == 1) & (targets == 1)).float().sum()
    false_positives = ((predictions == 1) & (targets == 0)).float().sum()
    false_negatives = ((predictions == 0) & (targets == 1)).float().sum()
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    
    return precision.item(), recall.item()

def compute_f1_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute F1 score.
    
    Args:
        predictions: Predicted labels of shape (batch_size,)
        targets: Target labels of shape (batch_size,)
        
    Returns:
        F1 score as a float between 0 and 1
    """
    precision, recall = compute_precision_recall(predictions, targets)
    return 2 * (precision * recall) / (precision + recall + 1e-10) 