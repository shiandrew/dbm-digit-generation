#!/usr/bin/env python3
"""
Test suite for the fixed DBM implementation.
Tests all core functionality including energy computation, sampling, and training.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from typing import List


class TestRBM:
    """Test cases for RBM implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.visible_dim = 784
        self.hidden_dim = 500
        self.batch_size = 32
        
        # Import after adding to path
        from src.models.rbm import RBM
        self.rbm = RBM(
            visible_dim=self.visible_dim,
            hidden_dim=self.hidden_dim,
            use_cuda=False
        )
    
    def test_rbm_initialization(self):
        """Test RBM is properly initialized."""
        assert self.rbm.visible_dim == self.visible_dim
        assert self.rbm.hidden_dim == self.hidden_dim
        assert self.rbm.W.shape == (self.visible_dim, self.hidden_dim)
        assert self.rbm.visible_bias.shape == (self.visible_dim,)
        assert self.rbm.hidden_bias.shape == (self.hidden_dim,)
    
    def test_sample_hidden(self):
        """Test hidden unit sampling."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        hidden_probs, hidden_samples = self.rbm.sample_hidden(visible)
        
        assert hidden_probs.shape == (self.batch_size, self.hidden_dim)
        assert hidden_samples.shape == (self.batch_size, self.hidden_dim)
        assert torch.all(hidden_probs >= 0) and torch.all(hidden_probs <= 1)
        assert torch.all((hidden_samples == 0) | (hidden_samples == 1))
    
    def test_sample_visible(self):
        """Test visible unit sampling."""
        hidden = torch.rand(self.batch_size, self.hidden_dim)
        visible_probs, visible_samples = self.rbm.sample_visible(hidden)
        
        assert visible_probs.shape == (self.batch_size, self.visible_dim)
        assert visible_samples.shape == (self.batch_size, self.visible_dim)
        assert torch.all(visible_probs >= 0) and torch.all(visible_probs <= 1)
        assert torch.all((visible_samples == 0) | (visible_samples == 1))
    
    def test_free_energy(self):
        """Test free energy computation."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        free_energy = self.rbm.free_energy(visible)
        
        assert free_energy.shape == (self.batch_size,)
        assert torch.all(torch.isfinite(free_energy))
    
    def test_contrastive_divergence(self):
        """Test contrastive divergence computation."""
        visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        pos_grad, neg_grad, recon_error = self.rbm.contrastive_divergence(visible)
        
        assert pos_grad.shape == (self.visible_dim, self.hidden_dim)
        assert neg_grad.shape == (self.visible_dim, self.hidden_dim)
        assert isinstance(recon_error, float)
        assert recon_error >= 0
    
    def test_generation(self):
        """Test sample generation."""
        samples = self.rbm.generate_samples(num_samples=10, num_gibbs_steps=100)
        
        assert samples.shape == (10, self.visible_dim)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)


class TestDBM:
    """Test cases for DBM implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.visible_dim = 784
        self.hidden_dims = [500, 300]
        self.batch_size = 16
        
        # Import after adding to path
        from src.models.dbm import DBM
        self.dbm = DBM(
            visible_dim=self.visible_dim,
            hidden_dims=self.hidden_dims,
            use_cuda=False
        )
    
    def test_dbm_initialization(self):
        """Test DBM is properly initialized."""
        assert self.dbm.visible_dim == self.visible_dim
        assert self.dbm.hidden_dims == self.hidden_dims
        assert self.dbm.num_layers == len(self.hidden_dims)
        
        # Check weight dimensions
        assert len(self.dbm.weights) == len(self.hidden_dims)
        assert self.dbm.weights[0].shape == (self.visible_dim, self.hidden_dims[0])
        assert self.dbm.weights[1].shape == (self.hidden_dims[0], self.hidden_dims[1])
        
        # Check bias dimensions
        assert len(self.dbm.biases) == len(self.hidden_dims) + 1  # +1 for visible bias
        assert self.dbm.biases[0].shape == (self.visible_dim,)  # Visible bias
        assert self.dbm.biases[1].shape == (self.hidden_dims[0],)  # First hidden bias
        assert self.dbm.biases[2].shape == (self.hidden_dims[1],)  # Second hidden bias
    
    def test_energy_computation(self):
        """Test energy function computation."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        hiddens = [
            torch.rand(self.batch_size, self.hidden_dims[0]),
            torch.rand(self.batch_size, self.hidden_dims[1])
        ]
        
        energy = self.dbm.energy(visible, hiddens)
        
        assert energy.shape == (self.batch_size,)
        assert torch.all(torch.isfinite(energy))
    
    def test_free_energy_computation(self):
        """Test free energy computation."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        free_energy = self.dbm.free_energy(visible)
        
        assert free_energy.shape == (self.batch_size,)
        assert torch.all(torch.isfinite(free_energy))
    
    def test_forward_pass(self):
        """Test forward pass through DBM."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        hiddens = self.dbm.forward(visible)
        
        assert len(hiddens) == len(self.hidden_dims)
        for i, hidden in enumerate(hiddens):
            assert hidden.shape == (self.batch_size, self.hidden_dims[i])
            assert torch.all(hidden >= 0) and torch.all(hidden <= 1)
    
    def test_sampling_functions(self):
        """Test sampling functions."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        
        # Test sampling hidden given visible
        for layer_idx in range(self.dbm.num_layers):
            probs, samples = self.dbm.sample_hidden_given_visible(visible, layer_idx)
            expected_dim = self.hidden_dims[layer_idx]
            
            assert probs.shape == (self.batch_size, expected_dim)
            assert samples.shape == (self.batch_size, expected_dim)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
            assert torch.all((samples == 0) | (samples == 1))
        
        # Test sampling visible given hidden
        hidden = torch.rand(self.batch_size, self.hidden_dims[0])
        visible_probs, visible_samples = self.dbm.sample_visible_given_hidden(hidden, 0)
        
        assert visible_probs.shape == (self.batch_size, self.visible_dim)
        assert visible_samples.shape == (self.batch_size, self.visible_dim)
        assert torch.all(visible_probs >= 0) and torch.all(visible_probs <= 1)
        assert torch.all((visible_samples == 0) | (visible_samples == 1))
    
    def test_gibbs_step(self):
        """Test Gibbs sampling step."""
        visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        hiddens = [
            torch.bernoulli(torch.ones(self.batch_size, self.hidden_dims[0]) * 0.5),
            torch.bernoulli(torch.ones(self.batch_size, self.hidden_dims[1]) * 0.5)
        ]
        
        new_visible, new_hiddens = self.dbm.gibbs_step(visible, hiddens)
        
        assert new_visible.shape == visible.shape
        assert len(new_hiddens) == len(hiddens)
        for i, new_hidden in enumerate(new_hiddens):
            assert new_hidden.shape == hiddens[i].shape
            assert torch.all((new_hidden == 0) | (new_hidden == 1))
    
    def test_contrastive_divergence(self):
        """Test contrastive divergence for DBM."""
        visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        
        neg_visible, neg_hiddens = self.dbm.contrastive_divergence(visible, k=1)
        
        assert neg_visible.shape == visible.shape
        assert len(neg_hiddens) == len(self.hidden_dims)
        for i, neg_hidden in enumerate(neg_hiddens):
            assert neg_hidden.shape == (self.batch_size, self.hidden_dims[i])
    
    def test_generation(self):
        """Test sample generation from DBM."""
        samples = self.dbm.generate_samples(num_samples=5, num_gibbs_steps=10)
        
        assert samples.shape == (5, self.visible_dim)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)
    
    def test_save_load(self):
        """Test model saving and loading."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.pt")
            
            # Save model
            self.dbm.save(save_path)
            assert os.path.exists(save_path)
            
            # Load model
            from src.models.dbm import DBM
            loaded_dbm = DBM.load(save_path)
            
            # Check parameters are the same
            for p1, p2 in zip(self.dbm.parameters(), loaded_dbm.parameters()):
                assert torch.allclose(p1, p2)


class TestGibbsSampler:
    """Test cases for Gibbs sampler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        from src.models.dbm import DBM
        from src.sampling.gibbs_sampler import GibbsSampler
        
        self.visible_dim = 784
        self.hidden_dims = [500, 300]
        self.batch_size = 8
        
        self.dbm = DBM(
            visible_dim=self.visible_dim,
            hidden_dims=self.hidden_dims,
            use_cuda=False
        )
        self.sampler = GibbsSampler(self.dbm)
    
    def test_sampler_initialization(self):
        """Test sampler is properly initialized."""
        assert self.sampler.model == self.dbm
        assert self.sampler.device == torch.device('cpu')
    
    def test_sample_layer(self):
        """Test layer sampling."""
        visible = torch.rand(self.batch_size, self.visible_dim)
        
        # Test sampling first hidden layer
        probs, samples = self.sampler.sample_layer(
            layer_idx=1,
            bottom_layer=visible,
            top_layer=None
        )
        
        assert probs.shape == (self.batch_size, self.hidden_dims[0])
        assert samples.shape == (self.batch_size, self.hidden_dims[0])
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.all((samples == 0) | (samples == 1))
    
    def test_gibbs_step(self):
        """Test Gibbs sampling step."""
        visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        hiddens = [
            torch.bernoulli(torch.ones(self.batch_size, self.hidden_dims[0]) * 0.5),
            torch.bernoulli(torch.ones(self.batch_size, self.hidden_dims[1]) * 0.5)
        ]
        
        new_visible, new_hiddens = self.sampler.gibbs_step(visible, hiddens)
        
        assert new_visible.shape == visible.shape
        assert len(new_hiddens) == len(hiddens)
        for i, new_hidden in enumerate(new_hiddens):
            assert new_hidden.shape == hiddens[i].shape
    
    def test_sample_from_model(self):
        """Test sampling from model."""
        samples = self.sampler.sample_from_model(
            batch_size=5,
            n_steps=10
        )
        
        assert samples.shape == (5, self.visible_dim)
    
    def test_gibbs_chain(self):
        """Test Gibbs chain sampling."""
        initial_visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        
        chain_samples = self.sampler.gibbs_chain(
            initial_visible,
            n_steps=20,
            burn_in=5,
            sample_interval=2
        )
        
        # Should have (20 - 5) // 2 = 7 samples
        assert len(chain_samples) == 7
        for visible, hiddens in chain_samples:
            assert visible.shape == initial_visible.shape
            assert len(hiddens) == len(self.hidden_dims)
    
    def test_annealing(self):
        """Test simulated annealing."""
        initial_visible = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        temperature_schedule = [2.0, 1.5, 1.0, 0.5]
        
        final_sample = self.sampler.anneal_sampling(
            initial_visible,
            temperature_schedule,
            steps_per_temp=5
        )
        
        assert final_sample.shape == initial_visible.shape


class TestDBMTrainer:
    """Test cases for DBM trainer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        from src.models.dbm import DBM
        from src.training.trainer import DBMTrainer
        from torch.utils.data import TensorDataset, DataLoader
        
        self.visible_dim = 784
        self.hidden_dims = [100, 50]  # Smaller for faster testing
        self.batch_size = 16
        
        # Create mock dataset
        train_data = torch.bernoulli(torch.ones(100, self.visible_dim) * 0.5)
        train_labels = torch.randint(0, 10, (100,))
        train_dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        val_data = torch.bernoulli(torch.ones(20, self.visible_dim) * 0.5)
        val_labels = torch.randint(0, 10, (20,))
        val_dataset = TensorDataset(val_data, val_labels)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        self.dbm = DBM(
            visible_dim=self.visible_dim,
            hidden_dims=self.hidden_dims,
            use_cuda=False
        )
        
        # Mock config
        self.config = {
            'training': {
                'learning_rate': 0.01,
                'cd_steps': 1,
                'persistent_cd': False,
                'epochs': 2,
                'early_stopping_patience': 10
            },
            'logging': {
                'save_dir': os.path.join(os.getcwd(), 'test_checkpoints'),
                'log_interval': 10,
                'save_interval': 1
            }
        }
        
        # Create save directory
        os.makedirs(self.config['logging']['save_dir'], exist_ok=True)
        
        self.trainer = DBMTrainer(
            model=self.dbm,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=torch.device('cpu'),
            config=self.config
        )
    
    def test_trainer_initialization(self):
        """Test trainer is properly initialized."""
        assert self.trainer.model == self.dbm
        assert self.trainer.learning_rate == self.config['training']['learning_rate']
        assert self.trainer.cd_steps == self.config['training']['cd_steps']
    
    def test_contrastive_divergence_step(self):
        """Test CD step in trainer."""
        visible_data = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        
        recon_error, pos_hiddens, neg_data = self.trainer.contrastive_divergence_step(visible_data, k=1)
        
        assert isinstance(recon_error, torch.Tensor)
        assert recon_error.item() >= 0
        assert len(pos_hiddens) == len(self.hidden_dims)
        
        neg_visible, neg_hiddens = neg_data
        assert neg_visible.shape == visible_data.shape
        assert len(neg_hiddens) == len(self.hidden_dims)
    
    def test_compute_gradients(self):
        """Test gradient computation."""
        visible_data = torch.bernoulli(torch.ones(self.batch_size, self.visible_dim) * 0.5)
        pos_hiddens = self.dbm.forward(visible_data)
        neg_visible, neg_hiddens = self.dbm.contrastive_divergence(visible_data, k=1)
        
        gradients = self.trainer.compute_gradients(
            visible_data, pos_hiddens, (neg_visible, neg_hiddens)
        )
        
        # Check that all expected gradients are present
        assert 'visible_bias' in gradients
        for i in range(len(self.hidden_dims)):
            assert f'hidden_bias_{i}' in gradients
            assert f'weight_{i}' in gradients
    
    def test_update_parameters(self):
        """Test parameter updates."""
        # Get initial parameters
        initial_weights = [w.clone() for w in self.dbm.weights]
        initial_biases = [b.clone() for b in self.dbm.biases]
        
        # Create mock gradients
        gradients = {
            'visible_bias': torch.randn_like(self.dbm.biases[0]) * 0.01,
            'weight_0': torch.randn_like(self.dbm.weights[0]) * 0.01,
            'weight_1': torch.randn_like(self.dbm.weights[1]) * 0.01,
            'hidden_bias_0': torch.randn_like(self.dbm.biases[1]) * 0.01,
            'hidden_bias_1': torch.randn_like(self.dbm.biases[2]) * 0.01
        }
        
        # Update parameters
        self.trainer.update_parameters(gradients)
        
        # Check parameters were updated
        for i, (initial_w, current_w) in enumerate(zip(initial_weights, self.dbm.weights)):
            assert not torch.allclose(initial_w, current_w), f"Weight {i} was not updated"
        
        for i, (initial_b, current_b) in enumerate(zip(initial_biases, self.dbm.biases)):
            assert not torch.allclose(initial_b, current_b), f"Bias {i} was not updated"
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Train one epoch
        avg_loss = self.trainer.train_epoch(epoch=0)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
    
    def test_validate(self):
        """Test validation."""
        val_loss = self.trainer.validate(epoch=0)
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0
    
    def test_generate_samples(self):
        """Test sample generation."""
        samples = self.trainer.generate_samples(num_samples=5, num_steps=10)
        
        assert samples.shape == (5, self.visible_dim)


def run_integration_test():
    """
    Integration test that runs a mini training loop.
    """
    print("Running integration test...")
    
    from src.models.dbm import DBM
    from src.training.trainer import DBMTrainer
    from torch.utils.data import TensorDataset, DataLoader
    import tempfile
    
    # Create small DBM
    dbm = DBM(visible_dim=100, hidden_dims=[50, 25], use_cuda=False)
    
    # Create tiny dataset
    train_data = torch.bernoulli(torch.ones(50, 100) * 0.5)
    train_labels = torch.randint(0, 10, (50,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=10)
    
    val_data = torch.bernoulli(torch.ones(10, 100) * 0.5)
    val_labels = torch.randint(0, 10, (10,))
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=10)
    
    # Config for quick test
    config = {
        'training': {
            'learning_rate': 0.1,
            'cd_steps': 1,
            'persistent_cd': False,
            'epochs': 3,
            'early_stopping_patience': 10
        },
        'logging': {
            'save_dir': tempfile.mkdtemp(),
            'log_interval': 1,
            'save_interval': 1
        }
    }
    
    # Create trainer and train
    trainer = DBMTrainer(
        model=dbm,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    # Test training loop
    print("Testing training...")
    trainer.train()
    
    # Test generation
    print("Testing generation...")
    samples = trainer.generate_samples(num_samples=5, num_steps=50)
    assert samples.shape == (5, 100)
    
    print("Integration test passed!")


if __name__ == "__main__":
    """
    Run tests when script is executed directly.
    """
    print(f"Testing from directory: {os.getcwd()}")
    print(f"Python path includes: {sys.path[:3]}...")
    
    try:
        print("Testing RBM...")
        rbm_test = TestRBM()
        rbm_test.setup_method()
        rbm_test.test_rbm_initialization()
        rbm_test.test_sample_hidden()
        rbm_test.test_sample_visible()
        rbm_test.test_free_energy()
        rbm_test.test_contrastive_divergence()
        rbm_test.test_generation()
        print("✅ RBM tests passed!")
        
        print("\nTesting DBM...")
        dbm_test = TestDBM()
        dbm_test.setup_method()
        dbm_test.test_dbm_initialization()
        dbm_test.test_energy_computation()
        dbm_test.test_free_energy_computation()
        dbm_test.test_forward_pass()
        dbm_test.test_sampling_functions()
        dbm_test.test_gibbs_step()
        dbm_test.test_contrastive_divergence()
        dbm_test.test_generation()
        print("✅ DBM tests passed!")
        
        print("\nTesting Gibbs Sampler...")
        sampler_test = TestGibbsSampler()
        sampler_test.setup_method()
        sampler_test.test_sampler_initialization()
        sampler_test.test_sample_layer()
        sampler_test.test_gibbs_step()
        sampler_test.test_sample_from_model()
        sampler_test.test_gibbs_chain()
        sampler_test.test_annealing()
        print("✅ Gibbs Sampler tests passed!")
        
        print("\nTesting DBM Trainer...")
        trainer_test = TestDBMTrainer()
        trainer_test.setup_method()
        trainer_test.test_trainer_initialization()
        trainer_test.test_contrastive_divergence_step()
        trainer_test.test_compute_gradients()
        trainer_test.test_update_parameters()
        trainer_test.test_train_epoch()
        trainer_test.test_validate()
        trainer_test.test_generate_samples()
        print("✅ DBM Trainer tests passed!")
        
        print("\nRunning integration test...")
        run_integration_test()
        
        print("\n🎉 All tests passed! The DBM implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)