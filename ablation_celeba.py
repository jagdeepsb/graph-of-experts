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
from torch.utils.data import DataLoader

from src.binary_tree import (
    BinaryTreeGoE,
    CelebAOracleRouter,
    LatentVariableRouter,
    RandomBinaryTreeRouter,
)
from src.data import CelebADataset
from src.metrics import get_model_flops, get_num_parameters
from src.reference import ReferenceModel
from src.train import get_accuracy, save_model, training_loop

import warnings
warnings.filterwarnings("ignore")

CHECKPOINTS_DIR = "checkpoints/celeba"
RESULTS_DIR = "results/celeba"


def build_modules(
    param_factor: int,
) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """
    Architecture factory for RotatedMNIST task. Takes in hyperparameter param_factor, outputs list of module architectures which scale linearly in param_factor
    """

    class LayerOne(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, param_factor, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 1, 32, 32)
            """
            return F.relu(
                F.max_pool2d(self.conv(x), 2)
            )  # (batch_size, param_factor, 16, 16)

    class LayerTwo(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                param_factor, 2 * param_factor, kernel_size=3, padding=1
            )

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 32, 16, 16)
            """
            return F.relu(
                F.max_pool2d(self.conv(x), 2)
            )  # (batch_size, 2 * param_factor, 8, 8)

    class LayerThree(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                2 * param_factor, 4 * param_factor, kernel_size=3, padding=1
            )

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 64, 8, 8)
            """
            return F.relu(
                F.max_pool2d(self.conv(x), 2)
            )  # (batch_size, 4 * param_factor, 4, 4)

    class LayerFour(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4 * param_factor * 4 * 4, 8 * param_factor)
            self.fc2 = nn.Linear(8 * param_factor, 2)

        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 128, 4, 4)
            """
            x = x.view(
                -1, 4 * param_factor * 4 * 4
            )  # (batch_size, 4 * param_factor * 4 * 4)
            x = F.relu(self.fc1(x))  # (batch_size, 8 * param_factor)
            logits = self.fc2(x)  # (batch_size, 10)
            return logits

    return LayerOne, LayerTwo, LayerThree, LayerFour

def get_oracle_router():
    return CelebAOracleRouter()


@functools.lru_cache(maxsize=None)
def get_random_router(train_dataset):
    random_router = RandomBinaryTreeRouter(depth=4)
    random_router.compute_codebook(train_dataset)
    return random_router


@functools.lru_cache(maxsize=None)
def get_latent_router(train_dataset):
    latent_router = LatentVariableRouter(
        depth=4, dataset=train_dataset, experiment_type="celeba"
    )
    latent_router.compute_codebook(train_dataset)
    return latent_router


def main(experiment_name: str, num_runs: int):
    # Hyperparameters
    param_factors = [1, 2, 4, 8, 12, 16]
    # param_factors = [8]
    model_names = ["goe_oracle", "goe_latent"]
    # model_names = ["goe_latent"]

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = CelebADataset.get_celeba_loaders()

    num_devices = torch.cuda.device_count()
    compiled_results = {}
    for job_idx, (param_factor, model_name) in enumerate(
        product(param_factors, model_names)
    ):
        if (num_devices):
            device = torch.device(f"cuda:{job_idx % num_devices}")
        else:
            device = torch.device("cpu")
        for run_num in range(num_runs):
        
            modules_by_depth = build_modules(param_factor)

            run_id = f"{model_name}_{param_factor}_{run_num}"
            experiment_dir = os.path.join(CHECKPOINTS_DIR, experiment_name)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir, exist_ok=True)
            save_path = os.path.join(experiment_dir, f"{run_id}.pt")

            if model_name == "ref_sqA":
                raise NotImplementedError("ref_sqA not implemented for CelebA")
                model = ReferenceModel(modules_by_depth=modules_by_depth_sqA).to(device)
            elif model_name == "ref_sqB":
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
                raise ValueError(f"Model {run_id} not found at {save_path}")
            
            print(f"Loading {run_id} from {save_path}")
            model.load_state_dict(torch.load(save_path, map_location=device))
            
            # Compute accuracies
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
            accuracy = get_accuracy(model, test_loader, device, take_random_path=True)

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
                
            # save
            model_name_key = f"{model_name}_rand"
            if model_name_key not in compiled_results:
                    compiled_results[model_name_key] = {}
            if param_factor not in compiled_results[model_name_key]:
                compiled_results[model_name_key][param_factor] = {}
            compiled_results[model_name_key][param_factor][run_num] = {
                "accuracy": accuracy,
                "num_params": num_params,
                "flops": flops,
            }
    
    metadata = {
        "param_factors": param_factors,
        "model_names": [f"{model_name}_rand" for model_name in model_names],
        "num_runs": num_runs,
    }
    
    results = {
        "results": compiled_results,
        "metadata": metadata,
    }
            
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(results, f"{RESULTS_DIR}/{experiment_name}_rand.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="celeba_test")
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()
    print("Args: ", args)
    main(
        experiment_name=args.experiment_name,
        num_runs=args.num_runs,
    )
