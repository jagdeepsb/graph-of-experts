import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from src.vqvae.vqvae import VQVAE


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--n_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    # parser.add_argument("--embedding_dim", type=int, default=64)
    # parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--n_embeddings", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="CIFAR10")

    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename", type=str, default="vae_weights")

    args = parser.parse_args()
    return args


def get_vae(input_dim: int):
    args = get_args()
    model = VQVAE(
        input_dim,
        args.n_hiddens,
        args.n_residual_hiddens,
        args.n_residual_layers,
        args.n_embeddings,
        args.embedding_dim,
        args.beta,
    )
    return model


def train_vae(model: VQVAE, dataset: Dataset):
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save:
        print("Results will be saved in ./results/vqvae_" + args.filename + ".pth")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    model.train()

    results = {
        "n_updates": 0,
        "recon_errors": [],
        "loss_vals": [],
        "perplexities": [],
    }

    for i in range(args.n_updates):
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

        if i % args.log_interval == 0:
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
                np.mean(results["recon_errors"][-args.log_interval :]),
                "Loss",
                np.mean(results["loss_vals"][-args.log_interval :]),
                "Perplexity:",
                np.mean(results["perplexities"][-args.log_interval :]),
            )

    return model
