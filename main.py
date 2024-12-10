import os
from typing import Tuple, List, Type
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from torch.functional import F

from src.metrics import get_model_flops
from src.reference import ReferenceModel
from src.data import CelebADataset, FixedSizeWrapper
from src.binary_tree import CelebAOracleRouter, BinaryTreeGoE, LatentVariableRouter, RandomBinaryTreeRouter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_modules(param_factor: int) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
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
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, param_factor, 16, 16)
    
    class LayerTwo(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(param_factor, 2 * param_factor, kernel_size=3, padding=1)
    
        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 32, 16, 16)
            """
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, 2 * param_factor, 8, 8)

    class LayerThree(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2 * param_factor, 4 * param_factor, kernel_size=3, padding=1)
    
        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 64, 8, 8)
            """
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, 4 * param_factor, 4, 4)
    
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
            x = x.view(-1, 4 * param_factor * 4 * 4)  # (batch_size, 4 * param_factor * 4 * 4)
            x = F.relu(self.fc1(x))  # (batch_size, 8 * param_factor)
            logits = self.fc2(x)  # (batch_size, 10)
            return logits
    return LayerOne, LayerTwo, LayerThree, LayerFour

def build_modules_sqA(param_factor: int) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """
    Architecture factory for Status Quo A Model for CelebA task which has the same number of *total* parameters as the Graph of Experts. Takes in hyperparameter param_factor, outputs list of module architectures which scale linearly in param_factor
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
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, param_factor, 16, 16)
    
    class LayerTwo(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(param_factor, 2 * 2 * param_factor, kernel_size=3, padding=1)
    
        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 32, 16, 16)
            """
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, 2 * param_factor, 8, 8)

    class LayerThree(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2 * 2 * param_factor, 2 * 4 * param_factor, kernel_size=3, padding=1)
    
        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 64, 8, 8)
            """
            return F.relu(F.max_pool2d(self.conv(x), 2))  # (batch_size, 4 * param_factor, 4, 4)
    
    class LayerFour(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2 * 4 * param_factor * 4 * 4, 2 * 8 * param_factor)
            self.fc2 = nn.Linear(2 * 8 * param_factor, 2)
    
        def forward(self, x: torch.Tensor):
            """
            Args:
            - x: (batch_size, 128, 4, 4)
            """
            x = x.view(-1, 2 * 4 * param_factor * 4 * 4)  # (batch_size, 4 * param_factor * 4 * 4)
            x = F.relu(self.fc1(x))  # (batch_size, 8 * param_factor)
            logits = self.fc2(x)  # (batch_size, 10)
            return logits
    return LayerOne, LayerTwo, LayerThree, LayerFour

def do_train_epoch(model, loader, optimizer, epoch) -> float:
    model.train()
    epoch_losses = []
    for idx, (images, comb_attr_labels, out_attr_labels) in enumerate(loader):
        images = images.to(device)
        comb_attr_labels = comb_attr_labels.to(device)
        out_attr_labels = out_attr_labels.to(device)
        
        # images to dtype float64
        images = images.float()
        
        router_metadata = {
            "comb_attr_labels": comb_attr_labels,
        }

        optimizer.zero_grad()
        logits = model(images, router_metadata=router_metadata) # (bs, num_out_attr_classes)
        
        # logits_np = logits.detach().cpu().numpy()
        # bs, num_out_attr_classes = logits_np.shape
        # assert num_out_attr_classes == 2
        # print(logits[0])
        
        # print(logits.shape, out_attr_labels.shape)
        # raise
        
        loss = cross_entropy(logits, out_attr_labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    epoch_loss = torch.mean(torch.Tensor(epoch_losses)).item()
    # print(f'Train epoch {epoch} loss: {epoch_loss:.4f}')
    return epoch_loss

def do_val_epoch(model, loader, epoch) -> float:
    model.eval()
    epoch_losses = []
    for idx, (images, comb_attr_labels, out_attr_labels) in tqdm(enumerate(loader)):
        images = images.to(device)
        comb_attr_labels = comb_attr_labels.to(device)
        out_attr_labels = out_attr_labels.to(device)
        
        router_metadata = {
            "comb_attr_labels": comb_attr_labels,
        }

        logits = model(images, router_metadata=router_metadata) # (bs, num_out_attr_classes)
        loss = cross_entropy(logits, out_attr_labels)
        epoch_losses.append(loss.item())
    epoch_loss = torch.mean(torch.Tensor(epoch_losses)).item()
    # print(f'Val epoch {epoch} loss: {epoch_loss:.4f}') 
    return epoch_loss

def get_accuracy(model, loader):
    model.eval()
    total_samples = 0
    total_correct = 0
    for idx, (images, comb_attr_labels, out_attr_labels) in enumerate(loader):
        images = images.to(device)
        comb_attr_labels = comb_attr_labels.to(device)
        out_attr_labels = out_attr_labels.to(device) # (bs,)
        
        router_metadata = {
            "comb_attr_labels": comb_attr_labels,
        }

        logits = model(images, router_metadata=router_metadata) # (bs, num_out_attr_classes)
        prediction = torch.argmax(logits, dim=1) # (bs,)
        total_samples += prediction.shape[0]
        total_correct += torch.sum(prediction == out_attr_labels).item()
    accuracy = total_correct / total_samples * 100
    return accuracy

def get_celebA_loaders(downsample_factor: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Initialize dataset
    dataset = CelebADataset(n=24000)
    
    # Assuming `dataset` is your PyTorch Dataset
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(40)
    )
    
    train_dataset = FixedSizeWrapper(dataset = train_dataset, size = train_size // downsample_factor)
    val_dataset = FixedSizeWrapper(dataset = val_dataset, size = val_size // downsample_factor)
    test_dataset = FixedSizeWrapper(dataset = test_dataset, size = test_size // downsample_factor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def training_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, num_epochs: int = 100, lr: float = 0.005, epochs_per_accuracy: int = 5) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print_model_parameters(model)
    val_losses = []
    accuracy = None
    # make a pbar displaying the epoch number, training loss, and validation loss, and test accuracy
    pbar = tqdm(range(num_epochs), desc='Epochs', total=num_epochs, leave=False, position=0)
    
    for epoch in pbar:
        train_loss = do_train_epoch(model, train_loader, optimizer, epoch)
        val_loss = do_val_epoch(model, val_loader, epoch)
        val_losses.append(val_loss)
        if epoch % epochs_per_accuracy == 0:
            accuracy = get_accuracy(model, test_loader)
    
        pbar.set_postfix({'Train loss': train_loss, 'Val loss': val_loss, 'Test acc': accuracy})
    return val_losses

def save_model(model: nn.Module, save_path: str):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model: nn.Module, save_path: str):
    return model.load_state_dict(torch.load(save_path))

def print_model_parameters(model, show_grad=True, show_details=True):
    """
    Print detailed information about model parameters.
    
    Args:
        model: PyTorch model
        show_grad (bool): Whether to show gradient information
        show_details (bool): Whether to show detailed statistics
    """
    total_params = 0
    trainable_params = 0
    
    print("\nModel Parameter Summary:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Shape':>20} {'Params':>10} {'Grad':>8}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        # Format shape as string
        shape_str = str(tuple(param.shape))
        
        # Get gradient info
        grad_str = "Yes" if param.requires_grad else "No"
        
        print(f"{name:<40} {shape_str:>20} {param_count:>10,d} {grad_str:>8}")
        
        if show_details:
            if param.data is not None:
                print(f"    • Mean: {param.data.mean():.6f}")
                print(f"    • Std:  {param.data.std():.6f}")
                print(f"    • Min:  {param.data.min():.6f}")
                print(f"    • Max:  {param.data.max():.6f}")
            
            if show_grad and param.grad is not None:
                print(f"    • Grad Mean: {param.grad.mean():.6f}")
                print(f"    • Grad Std:  {param.grad.std():.6f}")
            print()
    
    print("-" * 80)
    print(f"Total Parameters:     {total_params:,d}")
    print(f"Trainable Parameters: {trainable_params:,d}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,d}")
    print("-" * 80)

def get_parameter_count(model):
    """
    Get total and trainable parameter counts.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == '__main__':
    # Build loaders
    (
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader
    ) = get_celebA_loaders()

    # MODEL MATRIX TO TRAIN
    # param_factors = [2, 4, 8, 12, 16, 24]
    param_factors = [8, 12, 16, 24, 32]
    # param_factors = [8, 12]
    # model_names = ['ref_sqA', 'ref_sqB', 'goe_oracle', 'goe_random', 'goe_latent']
    model_names = ['goe_latent']
    # model_names = ['goe_oracle', 'goe_latent']
    num_runs_per_model = 5

    num_epochs = 20
    results = {}

    # routers
    oracle_router = CelebAOracleRouter()

    random_router = RandomBinaryTreeRouter(depth=4)
    random_router.compute_codebook(train_dataset)

    latent_router = LatentVariableRouter(depth=4, dataset=train_dataset)
    latent_router.compute_codebook(train_dataset)

    # init results
    for param_factor in param_factors:
        for model_name in model_names:
            results[(param_factor, model_name)] = []

    # train models
    for param_factor in param_factors:
        for model_name in model_names:
            for run in range(num_runs_per_model):
                # Build architectures
                modules_by_depth = build_modules(param_factor)
                modules_by_depth_sqA = build_modules_sqA(param_factor)
                
                if model_name == 'ref_sqA':
                    model = ReferenceModel(modules_by_depth=modules_by_depth_sqA).to(device)
                elif model_name == 'ref_sqB':
                    model = ReferenceModel(modules_by_depth=modules_by_depth).to(device)
                elif model_name == 'goe_oracle':
                    model = BinaryTreeGoE(modules_by_depth = modules_by_depth, router=oracle_router).to(device)
                elif model_name == 'goe_random':
                    model = BinaryTreeGoE(modules_by_depth = modules_by_depth, router=random_router).to(device)
                elif model_name == 'goe_latent':
                    model = BinaryTreeGoE(modules_by_depth = modules_by_depth, router=latent_router).to(device)
                else:
                    raise ValueError(f'Unknown model name: {model_name}')
                
                print(f'Training {model_name} at param_factor={param_factor}, run={run}')
                
                _ = training_loop(
                    model=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    test_loader=test_loader, 
                    num_epochs=num_epochs, 
                    lr = 0.005, 
                    epochs_per_accuracy=5
                )
                results[(param_factor, model_name)].append(model)
                reference_save_path = f'checkpoints/celeba_oracle/param_{param_factor}_{model_name}_{run}.pt'
                save_model(model, reference_save_path)
                
    # For each experiment, compute accuracies, and flops
    accuracies = {}
    mean_accuracies = {}
    std_accuracies = {}
    flops = {}

    # example input
    image, comb_attr_label, out_attr_label = train_dataset[0]
    image = image.unsqueeze(0).to(device)
    comb_attr_label = torch.tensor(comb_attr_label).unsqueeze(0).to(device)
    router_metadata = {
        "comb_attr_labels": comb_attr_label,
    }

    for k, models in tqdm(results.items()):
        n_params, model_name = k
        accuracies[k] = [get_accuracy(model, test_loader) for model in models]
        mean_accuracies[k] = torch.mean(torch.Tensor(accuracies[k])).item()
        std_accuracies[k] = torch.std(torch.Tensor(accuracies[k])).item()
        
        flops[k] = get_model_flops(models[0], (image, router_metadata))
        
    # compile results by model name
    compiled_results = {}

    for (param_factor, model_name) in results:
        if model_name not in compiled_results:
            compiled_results[model_name] = {
                'accuracies': [],
                'mean_accuracies': [],
                'std_accuracies': [],
                'flops': []
            }
        compiled_results[model_name]['accuracies'].append(accuracies[(param_factor, model_name)])
        compiled_results[model_name]['mean_accuracies'].append(mean_accuracies[(param_factor, model_name)])
        compiled_results[model_name]['std_accuracies'].append(std_accuracies[(param_factor, model_name)])
        compiled_results[model_name]['flops'].append(flops[(param_factor, model_name)])
        
        
    torch.save(compiled_results, 'celeba_compiled_results.pt')
    compiled_results = torch.load('celeba_compiled_results.pt')
    
    model_name_to_lablel = {
        'ref_sqA': 'Ref SqA',
        'ref_sqB': 'Ref SqB',
        'goe_oracle': 'GoE Oracle',
        'goe_random': 'GoE Random',
        'goe_latent': 'GoE Latent'
    }
    # dont_plot = ['ref_sqA']
    dont_plot = []

    for model_name, statistics in compiled_results.items():
        if model_name in dont_plot:
            continue
        mean_accuracies = statistics['mean_accuracies']
        std_accuracies = statistics['std_accuracies']
        flops = statistics['flops']
        plt.errorbar(flops, mean_accuracies, yerr=std_accuracies, label=model_name_to_lablel[model_name])
    plt.xlabel('FLOPS')
    plt.ylabel('Accuracy')
    plt.title('Rotated MNIST')
    plt.legend()
    plt.savefig('celeba_oracle.png')