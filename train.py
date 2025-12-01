
import torch
import torch.optim as optim
from data import generate_batch
from model import TinyGPTPhysics

def train_model(epochs=2000, batch_size=32, T=16, lr=1e-3, save_path="model.pt"):
    model = TinyGPTPhysics(seq_len=T)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for step in range(epochs):
        inp, tgt = generate_batch(batch_size=batch_size, T=T)
        pred = model(inp)
        loss = loss_fn(pred, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss = {loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
