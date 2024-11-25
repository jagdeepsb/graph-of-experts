import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gumbel_probs(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    gumbel_logits = (logits + gumbel_noise) / temperature
    probs = F.softmax(gumbel_logits, dim=-1)
    return probs

class SmallFF(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=16):
        super(SmallFF, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class TreeNode(nn.Module):
    def __init__(self, dims, temperature=0.5, depth=0):
        super(TreeNode, self).__init__()
        self.dims = dims
        self.depth = depth
        self.temperature = temperature
        self.is_leaf = (len(self.dims) == 2) # 2 dims is a function from R^d1 -> R^d2
        self.input_dim, self.output_dim = dims[0], dims[1]
        
        # Node-specific layer for output activation
        self.output_layer = nn.Linear(self.input_dim, self.output_dim)
        # self.output_layer = SmallFF(self.input_dim, self.output_dim)
        
        # Distribution layer to decide which child to attend to
        # self.child_selector = nn.Linear(self.input_dim, 2)
        self.child_selector = SmallFF(self.input_dim, 2)
        
        
        # Define left and right child nodes, if within depth limit
        if not self.is_leaf:
            new_dims = [d for d in dims[1:]] # Remove the input dimension for the current node
            self.left_child = TreeNode(new_dims, temperature=self.temperature, depth=depth + 1)
            self.right_child = TreeNode(new_dims, temperature=self.temperature, depth=depth + 1)

    def forward(self, x):
        """
        Computes forward pass. 
        If eval mode, also returns a signature of the path taken.
        """
        
        raise NotImplementedError("Use inference or expected_loss instead")
    
    def should_go_left_mask(self, x_og: torch.Tensor) -> torch.Tensor:
        if self.depth == 0:
            return x_og < 0
        if self.depth == 1:
            # less than -0.5 and greater than 0.5
            return torch.logical_or((x_og < -0.5), (x_og > 0.5))
        else:
            raise ValueError("Depth not supported")
        
    
    def inference(self, x, x_og):
        """
        Inference mode, returns the output and the path taken.
        """
        
        # Compute the node's output activation
        my_output = self.output_layer(x)
        
        # If this is a leaf node, return the output
        if self.is_leaf:
            return my_output, ""
        
        # Compute logits for the child selection distribution
        logits = self.child_selector(x)
        
        # Assert that the batch size is 1 during inference, can't do routing otherwise
        assert x.size(0) == 1, "Batch size must be 1 during inference"
        
        # During inference, pick the highest probability child
        _, should_go_left = torch.max(logits, dim=-1)
        should_go_left = should_go_left.item() == 0
        
        probs = torch.softmax(logits, dim=-1)
        # should_go_left = (torch.rand(1) < probs[0, 0]).item()
        
        print(logits, probs, should_go_left)
        
        # Override with deterministic selection
        # should_go_left = self.should_go_left_mask(x_og).item() # bool
        
        if should_go_left:
            value, sig = self.left_child.inference(my_output, x_og)
            return value, "L" + sig
        else:
            value, sig = self.right_child.inference(my_output, x_og)
            return value, "R" + sig


    def expected_loss(self, x, x_og, y_t, loss_fn):
        """
        Compute the forward pass and expected loss.
        """
        
        # Compute the node's output activation
        my_output = self.output_layer(x)
        
        # If this is a leaf node, return the output
        if self.is_leaf:
            # return loss_fn(my_output, y_t)
            return (my_output - y_t) ** 2
        
        # Compute logits for the child selection distribution
        logits = self.child_selector(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Sample a soft selection using Gumbel-Softmax during training
        # probs = get_gumbel_probs(logits, self.temperature)
        
        # should_go_left = self.should_go_left_mask(x_og)
        # should_go_right = ~should_go_left
        
        # Recursively compute the children's losses
        left_loss = self.left_child.expected_loss(my_output, x_og, y_t, loss_fn)
        right_loss = self.right_child.expected_loss(my_output, x_og, y_t, loss_fn)
        
        # Weighted average of left and right child losses
        combined_loss = probs[:, 0:1] * left_loss + probs[:, 1:2] * right_loss
        
        # Use deterministic selection
        # combined_loss = should_go_left * left_loss + should_go_right * right_loss
        
        return combined_loss    

if __name__ == "__main__":
    # Example usage:
    dims = [1, 10, 10, 1]
    input_dim = dims[0]
    temperature = 0.5

    model = TreeNode(dims, temperature=temperature)
    input_data = torch.randn(16, input_dim)

    # Training forward pass
    model.train()  # Sets the module to training mode
    output_train = model(input_data)
    print("Training output:", output_train)

    # Inference forward pass
    model.eval()  # Sets the module to evaluation mode
    with torch.no_grad():  # Optional: use no_grad to save memory during inference
        output_infer = model(input_data[0:1])
    print("Inference output:", output_infer)