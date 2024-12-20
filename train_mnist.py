import argparse
import functools
import multiprocessing
import os
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from typing import List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.binary_tree import (
    BinaryTreeGoE,
    LatentVariableRouter,
    MNISTOracleRouter,
    RandomBinaryTreeRouter,
)
from src.data import RotatedMNISTDataset
from src.metrics import get_model_flops, get_num_parameters
from src.reference import ReferenceModel
from src.train import get_accuracy, save_model, training_loop

CHECKPOINTS_DIR = "checkpoints/rotated_mnist"
RESULTS_DIR = "results/rotated_mnist"

def build_modules(
    param_factor: int,
) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """
    Architecture factory for RotatedMNIST task. Takes in hyperparameter param_factor, outputs list of module architectures which scale linearly in param_factor
    """

    class LayerOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, param_factor, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 1, 28, 28)
            """
            return F.relu(
                F.max_pool2d(self.conv(x), 2)
            )  # (batch_size, param_factor, 14, 14)

    class LayerTwo(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                param_factor, 2 * param_factor, kernel_size=3, padding=1
            )

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 32, 14, 14)
            """
            return F.relu(
                F.max_pool2d(self.conv(x), 2)
            )  # (batch_size, 2 * param_factor, 7, 7)

    class LayerThree(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2 * param_factor * 7 * 7, 4 * param_factor)
            self.fc2 = nn.Linear(4 * param_factor, 10)

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 64, 7, 7)
            """
            x = x.view(
                -1, 2 * param_factor * 7 * 7
            )  # (batch_size, 2 * param_factor * 7 * 7)
            x = F.relu(self.fc1(x))  # (batch_size, 4 * param_factor)
            logits = self.fc2(x)  # (batch_size, 10)
            return logits

    return LayerOne, LayerTwo, LayerThree


@dataclass
class TrainConfig:
    experiment_name: str
    model_name: str
    param_factor: int
    run_num: int
    num_epochs: int
    device: torch.device


@dataclass
class TrainResult:
    experiment_name: str
    model_name: str
    param_factor: int
    run_num: int
    accuracy: float
    flops: int
    num_params: int


def get_oracle_router():
    return MNISTOracleRouter()


@functools.lru_cache(maxsize=None)
def get_random_router(train_dataset):
    random_router = RandomBinaryTreeRouter(depth=3)
    random_router.compute_codebook(train_dataset)
    return random_router


@functools.lru_cache(maxsize=None)
def get_latent_router(train_dataset):
    latent_router = LatentVariableRouter(depth=3, dataset=train_dataset)
    latent_router.compute_codebook(train_dataset)
    return latent_router


def train_model(cfg: TrainConfig):
    # Unpack config
    experiment_name = cfg.experiment_name
    model_name = cfg.model_name
    param_factor = cfg.param_factor
    num_epochs = cfg.num_epochs
    run_num = cfg.run_num
    device = cfg.device

    # Build dataloaders and datasets
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = RotatedMNISTDataset.get_rotated_mnist_loaders()

    modules_by_depth = build_modules(param_factor)

    run_id = f"{model_name}_{param_factor}_{run_num}"
    experiment_dir = os.path.join(CHECKPOINTS_DIR, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    save_path = os.path.join(experiment_dir, f"{run_id}.pt")

    if model_name == "ref_sqB":
        model = ReferenceModel(modules_by_depth=modules_by_depth).to(device)
    elif model_name == "goe_oracle":
        oracle_router = get_oracle_router()
        model = BinaryTreeGoE(
            modules_by_depth=modules_by_depth, router=oracle_router
        ).to(device)
    elif model_name == "goe_random":
        random_router = get_random_router(train_dataset)
        model = BinaryTreeGoE(
            modules_by_depth=modules_by_depth, router=random_router
        ).to(device)
    elif model_name == "goe_latent":
        latent_router = get_latent_router(train_dataset)
        model = BinaryTreeGoE(
            modules_by_depth=modules_by_depth, router=latent_router
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not os.path.exists(save_path):
        print(f"Training {run_id}...")

        _ = training_loop(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=0.005,
            epochs_per_accuracy=5,
        )
        save_model(model, save_path)
    else:
        print(f"Loading {run_id} from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    # Compute accuracies
    accuracy = get_accuracy(model, test_loader, device)

    # Compute flops
    image, rotation_label, digit_label = train_dataset[0]
    image = image.unsqueeze(0).to(device)
    rotation_label = torch.tensor(rotation_label).unsqueeze(0).to(device)
    router_metadata = {
        "oracle_metadata": rotation_label,
    }
    num_params = get_num_parameters(model)
    try:
        flops = get_model_flops(model, (image, router_metadata))
    except RuntimeError:
        flops = 0

    return TrainResult(
        experiment_name=experiment_name,
        model_name=model_name,
        param_factor=param_factor,
        run_num=run_num,
        accuracy=accuracy,
        num_params=num_params,
        flops=flops,
    )


def main(num_workers: int, num_epochs: int, num_runs: int, experiment_name: str):
    # Hyperparameters
    param_factors = [2, 4, 8, 24]
    # model_names = ["ref_sqB", "goe_oracle", "goe_latent"]
    model_names = ["goe_latent"]

    jobs: List[TrainConfig] = []
    num_devices = torch.cuda.device_count()
    for job_idx, (param_factor, model_name) in enumerate(
        product(param_factors, model_names)
    ):
        if (num_devices):
            device = torch.device(f"cuda:{job_idx % num_devices}")
        else:
            device = torch.device(f"cpu")
        for run in range(num_runs):
            jobs.append(
                TrainConfig(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    param_factor=param_factor,
                    num_epochs=num_epochs,
                    run_num=run,
                    device=device,
                )
            )

    compiled_results = {}

    print(f"Running {len(jobs)} jobs on {num_workers} workers...")
    with Pool(num_workers) as pool:
        for job_result in pool.map(train_model, jobs):
            if job_result.model_name not in compiled_results:
                compiled_results[job_result.model_name] = {}
            if job_result.param_factor not in compiled_results[job_result.model_name]:
                compiled_results[job_result.model_name][job_result.param_factor] = {}
            compiled_results[job_result.model_name][job_result.param_factor][
                job_result.run_num
            ] = {
                "accuracy": job_result.accuracy,
                "num_params": job_result.num_params,
                "flops": job_result.flops,
            }

    metadata = {
        "param_factors": param_factors,
        "model_names": model_names,
        "num_epochs": num_epochs,
        "num_runs": num_runs,
    }

    results = {
        "results": compiled_results,
        "metadata": metadata,
    }
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(results, f"{RESULTS_DIR}/{experiment_name}_results.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="rotated_mnist_test")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")
    main(
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        experiment_name=args.experiment_name,
    )
