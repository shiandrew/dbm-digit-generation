import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    """
    MNIST dataset wrapper with preprocessing.
    """
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

def get_dataset(name, train_val_split=0.9, normalize=True):
    """
    Get dataset with train/val split.
    
    Args:
        name (str): Dataset name ('mnist' or 'fashion_mnist')
        train_val_split (float): Ratio of training data
        normalize (bool): Whether to normalize the data
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if name.lower() == 'mnist':
        dataset_class = datasets.MNIST
    elif name.lower() == 'fashion_mnist':
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Define transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    dataset = dataset_class(
        root='data/raw',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset

def get_test_dataset(name, normalize=True):
    """
    Get test dataset.
    
    Args:
        name (str): Dataset name ('mnist' or 'fashion_mnist')
        normalize (bool): Whether to normalize the data
        
    Returns:
        Dataset: Test dataset
    """
    if name.lower() == 'mnist':
        dataset_class = datasets.MNIST
    elif name.lower() == 'fashion_mnist':
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Define transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    test_dataset = dataset_class(
        root='data/raw',
        train=False,
        download=True,
        transform=transform
    )
    
    return test_dataset 