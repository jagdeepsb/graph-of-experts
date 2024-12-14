import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

from src.vqvae.vqvae import VQVAE


def get_vae(input_dim: int):
    model = VQVAE(
        input_dim,
    )
    return model


def train_vae(
    model: VQVAE,
    dataset: Dataset,
    log_interval: int = 50,
    lr: float = 3e-4,
    n_updates: int = 1000,
    batch_size: int = 128,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    model.train()

    results = {
        "n_updates": 0,
        "recon_errors": [],
        "loss_vals": [],
        "perplexities": [],
    }

    for i in range(n_updates):
        el = next(iter(train_loader))
        x = el[0]
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2)
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % log_interval == 0:
            """
            save model and print values
            """
            # if args.save:
            #     hyperparameters = args.__dict__
            #     utils.save_model_and_results(
            #         model, results, hyperparameters, args.filename)

            print(
                "Update #",
                i,
                "Recon Error:",
                np.mean(results["recon_errors"][-log_interval:]),
                "Loss",
                np.mean(results["loss_vals"][-log_interval:]),
                "Perplexity:",
                np.mean(results["perplexities"][-log_interval:]),
            )
    save_path = model.get_save_path()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model
