import torch
import torch.nn as nn
from timm.models import register_model
import numpy as np


PATH_TO_DYNAMIC_PARTITIONS = 'dy_point_order.pt'

class InterPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
    def forward(self,x):
        B, C, P, _ = x.shape # BxT, C, P, 1
        tmp_x = x
        x_i = x.repeat(1,1,1,P) # BxT, C, P, P
        x_j = x_i
        for k in range(P):
            x_j[:,:,:,k] = x_i[:,:,k,k].unsqueeze(-1).repeat(1,1,P)
        relative = x_j - x_i
        for part in range(5):
            tmp_relative = relative
            tmp_relative[:,:,:,part*3:(part+1)*3] = relative[:,:,:,part*3:(part+1)*3] - 1e4
            tmp_x_j,_ = torch.max(tmp_relative, -1, keepdim=True)
            tmp_x[:,:,part*3:(part+1)*3,:] = tmp_x_j[:,:,part*3:(part+1)*3,:]
            
        x = torch.cat([x, tmp_x],1)
        return self.nn(x)

class IntraPartMR(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
    def forward(self,x):
        B, C, P, _ = x.shape # BxT, C, P, 1
        tmp_x = x
        x_i = x.repeat(1,1,1,P) # BxT, C, P, P
        x_j = x_i
        for k in range(P):
            x_j[:,:,:,k] = x_i[:,:,k,k].unsqueeze(-1).repeat(1,1,P)
        
        relative = x_j - x_i # BxT, C, P, P
        part = 1
        for point in range(P):
            tmp_x_j,_= torch.max(relative[:,:,point,(part-1)*3+1:part*3+1], -1, keepdim=True)
            # Part_x_j[:,:,point,1] = tmp_x_j.squeeze(-1)
            tmp_x[:,:,point,:] = tmp_x_j
            if point+1 % 3 == 0:
                part = 1+part
        x = torch.cat([x, tmp_x],1)
        return self.nn(x)

class Grapher(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn_inter = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels*5, 1, groups=4),
            nn.BatchNorm2d(out_channels*5),
            nn.ReLU()
            )
        self.nn_intra = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels*2, 1, groups=4),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
            )
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        self.fc4 = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.InterPartMR = InterPartMR(out_channels)
        self.IntraPartMR = IntraPartMR(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # B, T, P, C = x.shape
        B, T, C, P, _ = x.shape # B,T,C,P,1
        x = x.view(-1,C,P,1) # BxT, C, P, 1
        tmp_x = x
        x = self.fc1(x)
        x = self.InterPartMR(x) # BxT, C*5, P, 1
        x = self.fc2(x)
        x = x+tmp_x
        x = self.act(x)
        x = self.fc3(x)
        x = self.IntraPartMR(x)

        x = self.fc4(x)
        x = x + tmp_x
        x = self.act(x)
        return x.view(B,T,C,P,1)

class Part_3DCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, dynamic=False, dynamic_point_order=None, SEED=None, expansion=4):
        super().__init__()
        self.expansion = expansion 
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), stride=stride, padding=1, padding_mode='replicate'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels*self.expansion, 1),
            nn.BatchNorm3d(out_channels*self.expansion),
            nn.ReLU()
        )

        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels*self.expansion, 1, stride=stride),
                nn.BatchNorm3d(out_channels*self.expansion)
            )

        self.dynamic = dynamic
        # self.dynamic_part = self.dynamic_trans()
        self.dynamic_point_order = dynamic_point_order
        self.act = nn.ReLU()
        self.stride = stride
        self.in_ = in_channels
        self.SEED = SEED
    def dynamic_trans(self, x): # B,C,T,P,1, P: 15
        B,C,T,P,_ = x.shape
        x = x.view(-1,P)
        dynamic_order = self.dynamic_point_order[self.SEED]
        raw_order = list(np.arange(15))
        x[:,raw_order] = x[:,dynamic_order]

        return x.view(B,C,T,P,1)
    
    def forward(self, x):
        B,T,C,P,_ = x.shape # B,T,C,P,1
        x = x.transpose(1,2).contiguous() # B,C,T,P,1
        if self.dynamic:
            x = self.dynamic_trans(x)
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x[:,:,:,:,1].unsqueeze(-1)
        x = self.conv3(x)
        residual = self.downsample(residual) 
        x = residual + x
        x = self.act(x)
        return  x.transpose(1,2).contiguous() # B, T, C*expansion, P, 1

class LandmarkStem(nn.Module):
    """
    Replaces the Conv2d Stem. 
    Input: (B, T, P, 5) -> Output: (B, T, P, C_out)
    """
    def __init__(self, input_dim=5, output_dim=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim) if False else nn.Identity(), # BN on linear is tricky with 4D
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (B, T, P, 5)
        return self.stem(x) 

class STViG_Landmark(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # ... (mostly same init logic) ...
        self.dynamic = opt.dynamic
        num_layer = opt.num_layer
        output_channels = opt.output_channels
        dynamic_point_order = opt.dynamic_point_order
        expansion = opt.expansion
        
        # 1. NEW STEM: Linear Projection instead of Conv2d
        # Input dim = 5 (x, y, z, v, p)
        self.stem = nn.Linear(5, output_channels[0]) 
        
        # 2. REMOVED SEPARATE PE (Coordinates are the features!)
        # self.stem_pe = ... (Deleted)
        
        self.in_channels = output_channels[0]
        self.backbone = []
        
        # ... (Backbone construction loop is IDENTICAL to original) ...
        for stage in range(len(num_layer)):
            if stage > 0:
                self.backbone.append(Grapher(in_channels=self.in_channels, out_channels=output_channels[stage]))
                self.backbone.append(Part_3DCNN(stride=(2,1,1),
                                                in_channels= self.in_channels,
                                                out_channels=output_channels[stage],
                                                dynamic=self.dynamic, # fixed variable name
                                                dynamic_point_order=dynamic_point_order,
                                                expansion= expansion,
                                                SEED=stage*num_layer[stage]+0)) # simplified seed #old: +layers))
                self.in_channels = output_channels[stage] * expansion

            for layers in range(num_layer[stage]):
                self.backbone.append(Grapher(in_channels=self.in_channels, out_channels=output_channels[stage]))
                self.backbone.append(Part_3DCNN(in_channels=self.in_channels,
                                                out_channels=output_channels[stage],
                                                dynamic=self.dynamic,
                                                dynamic_point_order=dynamic_point_order,
                                                expansion=expansion,
                                                SEED=stage*num_layer[stage]+layers))
                if stage == 0:
                    self.in_channels = output_channels[stage] * expansion
                    
        self.backbone = nn.Sequential(*self.backbone)
        
        # Final Classifier
        self.fc = nn.Sequential(
            nn.Linear(output_channels[-1] * expansion, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Added dropout
            nn.Linear(256, 1) # Binary classification
        ) 

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x, mask=None):
        # x input: (B, T, P, 5) -> (Batch, 150, 15, 5)
        # mask: (B, 150)

        # 1. Embed Features (Linear Stem)
        x = self.stem(x) # -> (B, T, P, C)
        
        # 2. Reshape for Backbone
        # Backbone expects (B, C, T, P, 1) usually? Or (B, T, C, P, 1)?
        # Original: x.transpose(2,3).contiguous().view(B,T,C,P,1)
        # Check Grapher forward: B, T, C, P, _ = x.shape
        # So we need (B, T, C, P, 1)
        B, T, P, C = x.shape
        x = x.permute(0, 1, 3, 2).unsqueeze(-1) # (B, T, C, P, 1)
        
        # 3. Backbone (Graph + 3D-CNN)
        x = self.backbone(x)
        
        # If mask is provided, we need to downsample it to match T_new
        # because the backbone has strides (downsamples time).
        if mask is not None:
            # Reshape mask for interpolation: (B, 1, T)
            mask_float = mask.float().unsqueeze(1) 
            # Interpolate to new temporal dimension T_new
            mask_down = torch.nn.functional.interpolate(mask_float, size=x.shape[1], mode='nearest')
            # Shape: (B, 1, T_new) -> (B, T_new, 1, 1, 1) to broadcast
            mask_down = mask_down.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            
            # Apply mask: Zero out invalid frames
            x = x * mask_down
            
            # Masked Average Pooling
            # Sum over T and P
            sum_x = x.sum(dim=(1, 3, 4)) # (B, C)
            
            # Count valid elements
            # Valid frames * Points (15)
            # mask_down sum gives number of valid frames per batch
            valid_counts = mask_down.sum(dim=(1, 3, 4)) * x.shape[3] # Frames * Points
            
            # Avoid division by zero
            x = sum_x / (valid_counts + 1e-6)
            
        else:
            # 4. Global Pooling
            # x output: (B, T_new, C_new, P, 1)
            # Average over Time and Points
            x = x.mean(dim=(1, 3, 4)) # (B, C_new)
        
        # 5. Classifier
        #return torch.sigmoid(self.fc(x))
        return (self.fc(x))

@register_model
def VSViG_Landmark_Base(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, **kwargs):
            self.dynamic = True
            self.num_layer = [2,2,6,2]
            self.output_channels = [24,48,96,192]
            self.dynamic_point_order = torch.load(PATH_TO_DYNAMIC_PARTITIONS)
            self.expansion = 2
    opt = OptInit(**kwargs)
    model = STViG_Landmark(opt)
    return model