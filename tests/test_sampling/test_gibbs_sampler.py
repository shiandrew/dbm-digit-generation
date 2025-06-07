import pytest
import torch

from src.models.dbm import DBM
from src.sampling.gibbs_sampler import GibbsSampler

@pytest.fixture
def model():
    return DBM(
        visible_dim=784,
        hidden_dims=[500, 500],
        dropout_rate=0.2,
        batch_norm=True
    )

@pytest.fixture
def sampler(model):
    return GibbsSampler(model)

def test_sampler_initialization(sampler, model):
    """Test sampler initialization."""
    assert sampler.model == model
    assert sampler.device == torch.device('cpu')

def test_sample_visible(sampler):
    """Test sampling visible units."""
    batch_size = 32
    h = torch.randn(batch_size, 500)
    v = sampler.sample_visible(h)
    
    assert v.shape == (batch_size, 784)
    assert torch.all((v >= 0) & (v <= 1))  # Check sigmoid output

def test_sample_hidden(sampler):
    """Test sampling hidden units."""
    batch_size = 32
    v = torch.randn(batch_size, 784)
    h = sampler.sample_hidden(v)
    
    assert h.shape == (batch_size, 500)
    assert torch.all((h >= 0) & (h <= 1))  # Check sigmoid output

def test_gibbs_step(sampler):
    """Test one Gibbs sampling step."""
    batch_size = 32
    v = torch.randn(batch_size, 784)
    h = torch.randn(batch_size, 500)
    
    v_new, h_new = sampler.gibbs_step(v, h)
    
    assert v_new.shape == (batch_size, 784)
    assert h_new.shape == (batch_size, 500)
    assert torch.all((v_new >= 0) & (v_new <= 1))
    assert torch.all((h_new >= 0) & (h_new <= 1))

def test_gibbs_chain(sampler):
    """Test Gibbs sampling chain."""
    batch_size = 32
    n_steps = 10
    v = torch.randn(batch_size, 784)
    
    samples = sampler.gibbs_chain(v, n_steps=n_steps)
    
    assert len(samples) == n_steps + 1  # Initial + n_steps
    for v in samples:
        assert v.shape == (batch_size, 784)
        assert torch.all((v >= 0) & (v <= 1))

def test_sample_from_model(sampler):
    """Test sampling from model."""
    batch_size = 32
    n_steps = 10
    samples = sampler.sample_from_model(batch_size, n_steps=n_steps)
    
    assert samples.shape == (batch_size, 784)
    assert torch.all((samples >= 0) & (samples <= 1))

def test_sample_with_temperature(sampler):
    """Test sampling with temperature."""
    batch_size = 32
    temperature = 0.5
    samples = sampler.sample_from_model(
        batch_size,
        n_steps=10,
        temperature=temperature
    )
    
    assert samples.shape == (batch_size, 784)
    assert torch.all((samples >= 0) & (samples <= 1))

def test_sample_with_initial_state(sampler):
    """Test sampling with initial state."""
    batch_size = 32
    initial_state = torch.randn(batch_size, 784)
    samples = sampler.sample_from_model(
        batch_size,
        n_steps=10,
        initial_state=initial_state
    )
    
    assert samples.shape == (batch_size, 784)
    assert torch.all((samples >= 0) & (samples <= 1))

def test_sample_with_callback(sampler):
    """Test sampling with callback function."""
    batch_size = 32
    n_steps = 10
    callback_called = False
    
    def callback(step, v, h):
        nonlocal callback_called
        callback_called = True
        assert step >= 0
        assert v.shape == (batch_size, 784)
        assert h.shape == (batch_size, 500)
    
    samples = sampler.sample_from_model(
        batch_size,
        n_steps=n_steps,
        callback=callback
    )
    
    assert callback_called
    assert samples.shape == (batch_size, 784) 