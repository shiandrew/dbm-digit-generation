import pytest
import torch

from src.models.dbm import DBM

@pytest.fixture
def model():
    return DBM(
        visible_dim=784,
        hidden_dims=[500, 500],
        dropout_rate=0.2,
        batch_norm=True
    )

def test_model_initialization(model):
    """Test model initialization."""
    assert model.visible_dim == 784
    assert model.hidden_dims == [500, 500]
    assert model.dropout_rate == 0.2
    assert model.batch_norm is True
    assert len(model.layers) == 2
    assert len(model.bn_layers) == 2

def test_forward_pass(model):
    """Test forward pass through the model."""
    batch_size = 32
    x = torch.randn(batch_size, 784)
    output = model(x)
    
    assert output.shape == (batch_size, 500)
    assert torch.all((output >= 0) & (output <= 1))  # Check sigmoid output

def test_sample_visible(model):
    """Test sampling visible units."""
    batch_size = 32
    h = torch.randn(batch_size, 500)
    v = model.sample_visible(h)
    
    assert v.shape == (batch_size, 784)
    assert torch.all((v >= 0) & (v <= 1))  # Check sigmoid output

def test_sample_hidden(model):
    """Test sampling hidden units."""
    batch_size = 32
    v = torch.randn(batch_size, 784)
    h = model.sample_hidden(v)
    
    assert h.shape == (batch_size, 500)
    assert torch.all((h >= 0) & (h <= 1))  # Check sigmoid output

def test_free_energy(model):
    """Test free energy computation."""
    batch_size = 32
    v = torch.randn(batch_size, 784)
    energy = model.free_energy(v)
    
    assert energy.shape == (batch_size,)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()

def test_save_load(model, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "model.pt"
    model.save(save_path)
    
    # Load model
    loaded_model = DBM.load(save_path)
    
    # Check parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)

def test_model_device(model):
    """Test model device placement."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch_size = 32
    x = torch.randn(batch_size, 784, device=device)
    output = model(x)
    
    assert output.device == device 