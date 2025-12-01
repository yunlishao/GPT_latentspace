
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data import generate_trajectory
from model import TinyGPTPhysics

def visualize_latent(model_path="model.pt", T=16, num_samples=200):
    model = TinyGPTPhysics(seq_len=T)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_h = []
    all_labels = []

    with torch.no_grad():
        for _ in range(num_samples):
            inp, tgt, a = generate_trajectory(T=T)
            x = torch.tensor(inp).unsqueeze(0)
            _, h = model(x, return_hidden=True)
            h = h.squeeze(0).numpy()  # (T, d_model)

            all_h.append(h)
            all_labels += [a] * T

    H = np.vstack(all_h)
    labels = np.array(all_labels)

    H_2d = PCA(n_components=2).fit_transform(H)

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(H_2d[:,0], H_2d[:,1], c=labels, cmap='coolwarm')
    plt.colorbar(scatter, label="Acceleration a")
    plt.title("PCA of Transformer Latent States")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_latent()
