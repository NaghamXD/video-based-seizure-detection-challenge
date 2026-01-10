import torch
import numpy as np

# Configuration
num_layers = 12   # Standard VSViG depth (or check your model config)
num_joints = 33   # MediaPipe joints

# Create a LIST of random shuffles
orders_list = []

print(f"Generating shuffles for {num_layers} layers and {num_joints} joints...")

for i in range(num_layers):
    # Generate random permutation of 0..32
    # We convert to .tolist() to make it a standard Python list
    random_order = torch.randperm(num_joints).tolist()
    orders_list.append(random_order)
    
# Save to file
output_name = 'my_dy_point_order_33.pt'
torch.save(orders_list, output_name)

print(f"\n✅ Created '{output_name}'")
print(f"Type: {type(orders_list)}")
print(f"First shuffle: {orders_list[0]}")