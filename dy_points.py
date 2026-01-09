import torch

file_path = 'dy_point_order.pt'

print(f"📂 Inspecting: {file_path}")

data = torch.load(file_path, weights_only=False)

if isinstance(data, list):
    # Get dimensions
    num_layers = len(data)
    num_joints = len(data[0]) if num_layers > 0 else 0
    
    print("\n" + "="*30)
    print(f"📊 LIST DIMENSIONS found:")
    print("="*30)
    print(f"• List Length (Layers): {num_layers}")
    print(f"• Inner List Length (Joints): {num_joints}")
    
    # Check compatibility
    if num_joints == 15:
        print("\n⚠️ ALERT: This list contains 15 indices.")
        print("   It is for the original dataset (OpenPose).")
        print("   It will FAIL with your 33-joint MediaPipe data.")
    elif num_joints == 33:
        print("\n✅ SUCCESS: This list fits your 33-joint data.")
        
    print("\n--- Content Sample (First Layer) ---")
    print(data[0])

else:
    print(f"Unexpected type: {type(data)}")