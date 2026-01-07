from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Sampler
import torch, json
import torch.nn as nn
import numpy as np
import os
import random
from collections import defaultdict

# --- CONFIGURATION ---
PROCESSED_FOLDER = "processed_data" 
PATH_TO_DATA_FOLDER = PROCESSED_FOLDER 

# --- PATH CONFIGS ---
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

# --- SMART SAMPLER (FIXES LOADING SPEED) ---
class ChunkBatchSampler(Sampler):
    """
    Shuffles data, but keeps indices from the same chunk together 
    to prevent constant hard drive reloading.
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.chunk_indices = dataset.get_chunk_indices()
        self.chunks = list(self.chunk_indices.keys())
        
    def __iter__(self):
        # 1. Shuffle the order of chunks (e.g., read chunk_5, then chunk_2...)
        random.shuffle(self.chunks)
        
        final_indices = []
        
        for chunk_name in self.chunks:
            # 2. Get all indices in this chunk
            indices = self.chunk_indices[chunk_name]
            
            # 3. Shuffle indices WITHIN the chunk
            random.shuffle(indices)
            
            # 4. Add to list
            final_indices.extend(indices)
            
        # 5. Yield batches
        for i in range(0, len(final_indices), self.batch_size):
            yield final_indices[i : i + self.batch_size]

    def __len__(self):
        return (sum(len(v) for v in self.chunk_indices.values()) + self.batch_size - 1) // self.batch_size

# --- DATASET CLASS ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        with open(label_file, 'rb') as f:
            self._labels = json.load(f)
            
        map_path = os.path.join(data_folder, 'chunk_map.json')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Chunk map not found at {map_path}")
            
        with open(map_path, 'r') as f:
            self._chunk_map = json.load(f) # {'global_id': [filename, local_idx]}
            
        self.last_chunk_name = None
        self.last_chunk_data = None
        self.last_chunk_kpts = None

    def get_chunk_indices(self):
        """Helper for the Sampler to know which ID belongs to which chunk"""
        groups = defaultdict(list)
        for idx, item in enumerate(self._labels):
            global_id = str(item[0])
            if global_id in self._chunk_map:
                chunk_name = self._chunk_map[global_id][0]
                groups[chunk_name].append(idx)
        return groups

    def __getitem__(self, idx):
        target = float(self._labels[idx][1])
        global_id = str(self._labels[idx][0]) 
        
        if global_id not in self._chunk_map:
             # Just return a zero-tensor if missing (safer than crashing mid-training)
             # But strictly you should ensure map is complete.
             raise IndexError(f"ID {global_id} missing from map.")

        filename, local_idx = self._chunk_map[global_id]
        
        # Lazy Loading
        if filename != self.last_chunk_name:
            self.last_chunk_name = filename
            data_path = os.path.join(self._folder, filename)
            kpts_path = os.path.join(self._folder, filename.replace('chunk_data', 'chunk_kpts'))
            self.last_chunk_data = torch.load(data_path, map_location='cpu')
            self.last_chunk_kpts = torch.load(kpts_path, map_location='cpu')
            
        data = self.last_chunk_data[local_idx] # (30, 15, 3, 32, 32)
        kpts = self.last_chunk_kpts[local_idx] # (30, 15, 2)
        
        # --- FIX 1: NORMALIZE KEYPOINTS ---
        kpts = kpts.float()
        kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
        kpts[:, :, 1] = kpts[:, :, 1] / 1080.0

        # --- FIX 2: ADD CONFIDENCE CHANNEL (2 -> 3 CHANNELS) ---
        # Current shape: (30, 15, 2)
        # We need: (30, 15, 3)
        confidence = torch.ones((30, 15, 1), dtype=kpts.dtype)
        kpts = torch.cat((kpts, confidence), dim=2) 
        
        if self._transform: 
            B_frames, P, C, H, W = data.shape 
            data = data.view(B_frames*P*C, H, W)
            data = self._transform(data)
            data = data.view(B_frames, P, C, H, W)
            
        sample = {
            'data': data,
            'kpts': kpts 
        }
        return sample, target
    
    def __len__(self):
        return len(self._labels)

def train():
    train_label_path = os.path.join(PROCESSED_FOLDER, 'train_labels.json')
    val_label_path = os.path.join(PROCESSED_FOLDER, 'val_labels.json')
    
    models_to_train = ['Base'] 
    
    for m in models_to_train:
        print(f"Initializing {m} Model Training...")
        
        # 1. Setup Data
        dataset_train = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=train_label_path)
        dataset_val = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=val_label_path)
        
        train_sampler = ChunkBatchSampler(dataset_train, batch_size=32)
        train_loader = DataLoader(dataset_train, batch_sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
        
        # 2. Setup Model & Hardware
        if m == 'Base':
            model = VSViG_base() 
        elif m == 'Light':
            model = VSViG_light()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device: {device}")
        
        model = model.to(device)
        MSE = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        # --- RESUME LOGIC STARTS HERE ---
        start_epoch = 0
        min_valid_loss = np.inf
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

        # Check for 'last_checkpoint.pth' (Full state for resuming)
        if os.path.exists(PATH_TO_LAST_CKPT):
            print(f"Found checkpoint: {PATH_TO_LAST_CKPT}. Resuming...")
            checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            min_valid_loss = checkpoint['min_valid_loss']
            
            # Load history if it exists
            if 'history' in checkpoint:
                history = checkpoint['history']

        # Fallback: If only 'best_model.pth' exists (like your current situation)
        elif os.path.exists(PATH_TO_BEST_MODEL):
            print(f"Found best_model.pth but no checkpoint. Loading weights only.")
            # We assume a weight-only save for the old file
            try:
                model.load_state_dict(torch.load(PATH_TO_BEST_MODEL, map_location=device))
            except:
                print("Could not load best_model.pth weights. Starting fresh.")
            # Since we don't know the epoch, we might have to start at 0 or guess
            # You mentioned you were at epoch 137, so let's set it manually if you like:
            # start_epoch = 137 

        epochs = 200
        
        # 3. Training Loop
        for e in range(start_epoch, epochs):
            train_loss = 0.0
            model.train()
            optimizer.zero_grad()
            print(f'\n=== Epoch: {e+1} ===')

            for batch_idx, (sample, labels) in enumerate(train_loader):
                data = sample['data'].to(device)
                kpts = sample['kpts'].to(device)
                labels = labels.float().to(device)

                outputs = model(data, kpts)
                
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                loss = MSE(outputs.float(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"\rBatch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", end="")
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f'\nTraining Loss: {avg_train_loss:.4f}')

            # 4. Validation & Saving
            if (e+1) % 5 == 0:
                valid_loss = 0.0
                RMSE_loss = 0.0
                model.eval()
                
                with torch.no_grad():
                    for sample, labels in val_loader:
                        data = sample['data'].to(device)
                        kpts = sample['kpts'].to(device)
                        labels = labels.float().to(device)
                        
                        outputs = model(data, kpts)
                        if outputs.dim() > 1 and outputs.shape[1] == 1:
                            outputs = outputs.squeeze(1)
                        
                        loss = MSE(outputs, labels)
                        valid_loss += loss.item()
                        RMSE_loss += torch.sqrt(MSE(outputs, labels)).item() * 100
                
                avg_val_loss = valid_loss / len(val_loader)
                avg_rmse = RMSE_loss / len(val_loader)
                
                history['val_loss'].append(avg_val_loss)
                history['val_rmse'].append(avg_rmse)
                
                print(f' +++ Val Loss: {avg_val_loss:.3f} | Val RMSE: {avg_rmse:.3f} +++')

                # Save BEST model (Weights only is fine for inference)
                if min_valid_loss > valid_loss:
                    print(f'   -> Saving new best model to {PATH_TO_BEST_MODEL}')
                    min_valid_loss = valid_loss
                    torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
            
            scheduler.step()

            # 5. SAVE REGULAR CHECKPOINT (Every Epoch)
            # This allows you to resume even if you crash between validation runs
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_valid_loss': min_valid_loss,
                'history': history
            }, PATH_TO_LAST_CKPT)

            # 6. SAVE LOGS TO FILE
            with open(PATH_TO_LOG_FILE, 'w') as f:
                json.dump(history, f, indent=4)
                    
if __name__ == '__main__':
    train()