import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import os
import yaml
import torch
import matplotlib.pyplot as plt

from src.models.dbm import DBM
from src.sampling.gibbs_sampler import GibbsSampler



def parse_args():
    parser = argparse.ArgumentParser(description="Sample from trained DBM model")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    sampling_cfg = config['sampling']
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['device'] == 'cuda' else 'cpu')

    model_path = os.path.join(config['logging']['save_dir'], "best_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        return

    print(f"✅ Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    model_config = checkpoint.get('model_config', {})
    model = DBM(
        visible_dim=model_config['visible_dim'],
        hidden_dims=model_config['hidden_dims'],
        use_cuda=(device.type == 'cuda')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ DBM loaded: {model.visible_dim} → {model.hidden_dims}")

    sampler = GibbsSampler(model, device=device)
    if sampling_cfg.get('annealing', {}).get('enabled', False):
        
        print("🔥 Using annealed sampling...")

        # Define initial_visible here
        initial_visible = torch.bernoulli(
            torch.ones(sampling_cfg['num_samples'], model.visible_dim, device=device) * 0.5
        )

        samples = sampler.anneal_sampling(
            initial_visible=initial_visible,
            temperature_schedule=config['sampling']['annealing']['temperature_schedule'],
            steps_per_temp=config['sampling']['annealing']['steps_per_temp']
        )

    else:
        print("🎲 Using standard Gibbs sampling...")
        samples = sampler.sample_from_model(
            batch_size=sampling_cfg['num_samples'],
            n_steps=sampling_cfg['num_steps'],
            temperature=sampling_cfg['temperature'],
            return_chain=False
        )


    # Visualization (assumes binary MNIST-like output: 28x28)
    num_images = min(25, samples.shape[0])
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = samples[i].detach().cpu().numpy().reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout() 
    output_path = "results/generated_digits.png"
    plt.savefig(output_path)
    print(f"✅ Saved generated digits to {output_path}")
    plt.show()


if __name__ == '__main__':
    main()

