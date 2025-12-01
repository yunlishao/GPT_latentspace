
import numpy as np
import torch

def generate_trajectory(T=16, dt=0.1):
    """
    Generate a single trajectory under constant acceleration.
    Returns:
        inp: (T, 3) [x_k, v_k, a]
        tgt: (T, 2) [x_{k+1}, v_{k+1}]
        a: acceleration value
    """
    a = np.random.choice([-1.0, -0.5, 0.5, 1.0])
    x = np.zeros(T+1)
    v = np.zeros(T+1)

    x[0] = np.random.uniform(-1, 1)
    v[0] = np.random.uniform(-1, 1)

    for k in range(T):
        v[k+1] = v[k] + a * dt
        x[k+1] = x[k] + v[k] * dt

    inp = np.stack([x[:-1], v[:-1], np.full(T, a)], axis=-1).astype(np.float32)
    tgt = np.stack([x[1:], v[1:]], axis=-1).astype(np.float32)
    return inp, tgt, a

def generate_batch(batch_size=32, T=16):
    inps = []
    tgts = []
    for _ in range(batch_size):
        inp, tgt, _ = generate_trajectory(T=T)
        inps.append(inp)
        tgts.append(tgt)
    return torch.tensor(inps), torch.tensor(tgts)
