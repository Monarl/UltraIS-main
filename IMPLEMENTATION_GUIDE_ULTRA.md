# UltraIS Implementation Guide

> **A Dual-Stream-Modulated Learning Framework for Illuminating and Super-Resolving Ultra-Dark Images**
> 
> Published in: IEEE Transactions on Neural Networks and Learning Systems (TNNLS 2024)

This document provides a comprehensive explanation of the UltraIS codebase, covering the folder structure, training pipeline, model architecture, loss functions, logging system, and configuration options. This guide serves as knowledge base for understanding and extending the model.

---

## Table of Contents

1. [Folder Structure Overview](#1-folder-structure-overview)
2. [Training Pipeline](#2-training-pipeline)
3. [Model Architecture Implementation](#3-model-architecture-implementation)
4. [Loss Function Implementation](#4-loss-function-implementation)
5. [Results and Logging System](#5-results-and-logging-system)
6. [Configuration Options](#6-configuration-options)
7. [Key Code References](#7-key-code-references)

---

## 1. Folder Structure Overview

```
UltraIS-main/
├── basicsr/                    # Core framework directory
│   ├── train.py               # Main training script
│   ├── test.py                # Main testing script
│   ├── data/                  # Data loading modules
│   ├── metrics/               # Evaluation metrics (PSNR, SSIM, NIQE)
│   ├── models/                # Model definitions and training logic
│   │   ├── archs/             # Network architectures
│   │   └── losses/            # Loss function implementations
│   └── utils/                 # Utility functions
├── Super_Resolution/          # Task-specific configurations
│   └── Options/               # YAML configuration files
├── Figures/                   # Documentation images
├── experiments/               # Training outputs (created during training)
├── results/                   # Test results (created during testing)
└── train.sh                   # Training launch script
```

### 1.1 basicsr/data/
**Purpose:** Contains all data loading and preprocessing modules.

| File | Purpose |
|------|---------|
| [\_\_init\_\_.py](basicsr/data/__init__.py) | Dynamic dataset/dataloader instantiation |
| [paired_image_dataset.py](basicsr/data/paired_image_dataset.py) | Main dataset class for paired LQ-GT images |
| [transforms.py](basicsr/data/transforms.py) | Augmentation functions (flip, rotation, crop) |
| [data_util.py](basicsr/data/data_util.py) | Path handling utilities |
| [prefetch_dataloader.py](basicsr/data/prefetch_dataloader.py) | CPU/CUDA prefetching for faster data loading |

### 1.2 basicsr/models/
**Purpose:** Contains model training logic and network architectures.

| File | Purpose |
|------|---------|
| [\_\_init\_\_.py](basicsr/models/__init__.py) | Dynamic model class instantiation |
| [base_model.py](basicsr/models/base_model.py) | Base class with common training utilities |
| [image_restoration_model.py](basicsr/models/image_restoration_model.py) | Main model class `CSDLLSRv4` for training |
| [lr_scheduler.py](basicsr/models/lr_scheduler.py) | Learning rate scheduler implementations |

### 1.3 basicsr/models/archs/
**Purpose:** Neural network architecture definitions.

| File | Purpose |
|------|---------|
| [\_\_init\_\_.py](basicsr/models/archs/__init__.py) | Dynamic network instantiation |
| [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | **Main architecture: `CSDLLSRNetv9_7_5`** |
| [hrseg_lib/](basicsr/models/archs/hrseg_lib/) | HRNet segmentation backbone for semantic guidance |

### 1.4 basicsr/models/losses/
**Purpose:** Loss function implementations.

| File | Purpose |
|------|---------|
| [losses.py](basicsr/models/losses/losses.py) | All loss functions (L1, Perceptual, SATV, etc.) |
| [loss_util.py](basicsr/models/losses/loss_util.py) | Loss utility functions |

### 1.5 basicsr/metrics/
**Purpose:** Evaluation metrics.

| File | Purpose |
|------|---------|
| [psnr_ssim.py](basicsr/metrics/psnr_ssim.py) | PSNR and SSIM calculation |
| [niqe.py](basicsr/metrics/niqe.py) | NIQE (no-reference) metric |

### 1.6 basicsr/utils/
**Purpose:** Utility functions.

| File | Purpose |
|------|---------|
| [options.py](basicsr/utils/options.py) | YAML configuration parsing |
| [logger.py](basicsr/utils/logger.py) | Logging and TensorBoard utilities |
| [img_util.py](basicsr/utils/img_util.py) | Image I/O and conversion utilities |
| [misc.py](basicsr/utils/misc.py) | Miscellaneous utilities |

---

## 2. Training Pipeline

### 2.1 Entry Point
**File:** [train.sh](train.sh)
```bash
#!/usr/bin/env bash
CONFIG=$1
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt $CONFIG 
```

**Command:**
```bash
sh train.sh Super_Resolution/Options/CSDLLSR_v9_7_5_3_scale4.yml
```

### 2.2 Main Training Script
**File:** [basicsr/train.py](basicsr/train.py)

The training flow follows these steps:

#### Step 1: Parse Options (Lines 25-56)
```python
def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    # ... distributed settings and random seed
    return opt
```

#### Step 2: Initialize Loggers (Lines 58-76)
```python
def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # Initialize TensorBoard and/or Wandb loggers
    tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger
```

#### Step 3: Create Dataloaders (Lines 78-122)
```python
def create_train_val_dataloader(opt, logger):
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(train_set, dataset_opt, ...)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, ...)
    return train_loader, train_sampler, val_loader, total_epochs, total_iters
```

#### Step 4: Create Model (Lines 170-181)
```python
if resume_state:
    check_resume(opt, resume_state['iter'])
    model = create_model(opt)
    model.resume_training(resume_state)
else:
    model = create_model(opt)
    start_epoch = 0
    current_iter = 0
```

#### Step 5: Training Loop (Lines 227-315)
```python
while current_iter <= total_iters:
    train_sampler.set_epoch(epoch)
    prefetcher.reset()
    train_data = prefetcher.next()
    
    while train_data is not None:
        current_iter += 1
        model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
        
        # Progressive training: adjust patch size and batch size
        mini_gt_size = mini_gt_sizes[bs_j]
        mini_batch_size = mini_batch_sizes[bs_j]
        
        # Feed data to model
        model.feed_train_data({'lq': lq, 'gt': gt, 'gray': atten})
        
        # Optimize
        model.optimize_parameters(current_iter)
        
        # Logging
        if current_iter % opt['logger']['print_freq'] == 0:
            log_vars = {'epoch': epoch, 'iter': current_iter, ...}
            msg_logger(log_vars)
        
        # Save checkpoint
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            model.save(epoch, current_iter)
        
        # Validation
        if current_iter % opt['val']['val_freq'] == 0:
            model.validation(val_loader, current_iter, tb_logger, ...)
```

### 2.3 Progressive Training Strategy
**File:** [basicsr/train.py](basicsr/train.py) (Lines 220-282)

The training uses progressive patch sizes and batch sizes:
```python
iters = opt['datasets']['train'].get('iters')           # [46000, 32000, 24000, 18000, 18000, 12000]
mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')  # [8, 5, 4, 2, 1, 1]
mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')  # [96, 128, 160, 224, 288, 320]

groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

# During training, select appropriate size based on iteration
j = ((current_iter > groups) != True).nonzero()[0]
bs_j = j[0] if len(j) > 0 else len(groups) - 1
mini_gt_size = mini_gt_sizes[bs_j]
mini_batch_size = mini_batch_sizes[bs_j]
```

---

## 3. Model Architecture Implementation

### 3.1 Main Model Class
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py)

The main architecture is `CSDLLSRNetv9_7_5` (Lines 797-861):

```python
class CSDLLSRNetv9_7_5(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, n_feat=64, scale=1, bias=False):
        super(CSDLLSRNetv9_7_5, self).__init__()
        
        self.seg_dims = [59, 48, 96, 192, 384]       
        
        # Illumination estimation branch
        self.illumination = Illumination(2, 1, n_feat, scale)
        
        # Reflectance enhancement branch (main restoration)
        self.reflectance = Enhancement(inp_channels, n_feat, n_feat, scale, seg_dims=self.seg_dims)
        
        # HRNet for semantic segmentation guidance
        self.seg = create_hrnet()
        for p in self.seg.parameters():
            p.requires_grad = False  # Frozen pretrained weights
            
    def forward(self, inp_img_lllr, inp_img_gray):
        # 1. Extract semantic features from HRNet
        _, seg_orin, seg_fea = self.seg(inp_img_lllr)
        
        # 2. Estimate illumination map
        ill_feas, nllr_ill = self.illumination(inp_img_gray)
        
        # 3. Compute reflectance (Retinex decomposition)
        nllr_ill3 = torch.cat((nllr_ill, nllr_ill, nllr_ill), dim=1)
        nllr_ref = inp_img_lllr / nllr_ill3
        nllr_ref = torch.clamp(nllr_ref, 0, 1)
        
        # 4. Enhance reflectance with semantic and illumination guidance
        nlsr_refl = self.reflectance(nllr_ref, seg_orin, seg_fea, ill_feas)
        
        return nlsr_refl, nllr_ill, nlsr_refl
```

### 3.2 Illumination Estimation Module
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 432-498)

```python
class Illumination(nn.Module):
    def __init__(self, inp_channels=2, out_channels=1, n_feat=64, scale=1, bias=False):
        super(Illumination, self).__init__()
        
        # U-Net style encoder-decoder
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        # Encoder (4 levels with RCBdown blocks)
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # ... conv2-conv5 with pooling
        
        # Decoder with skip connections
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv6 = RCBup(n_feat=n_feat)
        # ... up7-up9 with RCBup
        
        self.conv_out_r = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1)
        
    def forward(self, refl):
        # Encoder path
        conv1 = self.lrelu(self.conv_in(refl))
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)
        # ... encoder forward
        
        # Decoder path with skip connections
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        # ... decoder forward
        
        out = torch.sigmoid(self.conv_out_r(out))
        return [conv5, conv6, conv7, conv8, conv9], out  # Multi-scale features + illumination map
```

### 3.3 Enhancement Module (Reflectance Enhancement)
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 712-790)

```python
class Enhancement(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, n_feat=64, scale=1, bias=False, seg_dims=None):
        super(Enhancement, self).__init__()
        
        # U-Net encoder
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        self.conv1-conv5 = RCBdown(n_feat=n_feat)
        
        # Transformer blocks for dual-stream modulation
        # sm5, sm6, ... : Semantic modulation (from HRNet features)
        # sm55, sm66, ... : Illumination modulation
        self.sm5 = TransformerBlock(dim=n_feat, dim2=seg_dims[4])  # Semantic guidance
        self.sm55 = TransformerBlock(dim=n_feat)                   # Illumination guidance
        
        # Super-resolution upsampling
        self.RTFU = RTFU(dim=n_feat, scale=scale)
        
    def forward(self, refl, seg_orin, seg_fea, ill_fea):
        # Encoder with dual modulation at each level
        conv5 = self.conv5(pool4)
        conv5 = self.sm55(conv5, ill_fea[0])   # Illumination modulation
        conv5 = self.sm5(conv5, seg_fea[3])    # Semantic modulation
        
        # Decoder with skip connections and modulation
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        conv6 = self.sm66(conv6, ill_fea[1])
        conv6 = self.sm6(conv6, seg_fea[2])
        # ... continue for all decoder levels
        
        # Final upsampling
        out = self.RTFU(conv9)
        return out
```

### 3.4 Key Building Blocks

#### 3.4.1 TransformerBlock (Cross-Attention Modulation)
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 163-188)

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, 
                 LayerNorm_type='WithBias', dim2=None):
        # dim: Feature dimension (64)
        # num_heads: Number of attention heads (2)
        # dim2: Guidance feature dimension (from ill_fea or seg_fea)
        
        super(TransformerBlock, self).__init__()
        
        # Check if guidance features have different channels
        if dim2 is not None:
            self.dim2 = dim2
            # 1x1 conv to match guidance feature channels to main feature channels
            self.conv_guidence = nn.Conv2d(dim2, dim, 1, 1)
        else:
            self.dim2 = None
            
        # Layer normalization for query features
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
        # Cross-attention module (main feature attends to guidance feature)
        self.attn = Attention(dim, num_heads, bias)
        
        # Layer normalization for feed-forward
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        
        # Feed-forward network for feature refinement
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, refl, ill_feat):
        # refl: Main feature from Enhancement decoder [B, 64, H, W]
        # ill_feat: Guidance feature (illumination OR semantic) [B, dim2, H', W']
        
        # Step 1: Align guidance feature dimensions
        if self.dim2 is not None:
            # Convert guidance channels: [B, dim2, H', W'] → [B, 64, H', W']
            ill_feat = self.conv_guidence(ill_feat)
            
            # Resize guidance to match main feature spatial size
            # [B, 64, H', W'] → [B, 64, H, W]
            ill_feat = F.interpolate(ill_feat, [refl.shape[2], refl.shape[3]])

        # Step 2: Store guidance for attention
        ill = ill_feat
        
        # Step 3: Normalize both features
        refl = self.norm1(refl)      # Normalize main feature
        ill = self.norm1(ill)        # Normalize guidance feature
        
        # Step 4: Cross-attention (main feature queries guidance feature)
        # refl attends to ill: [B, 64, H, W]
        refl = refl + self.attn(refl, ill)  # Residual connection
        
        # Step 5: Feed-forward refinement
        refl = refl + self.ffn(self.norm2(refl))  # Residual connection
        
        return refl  # Enhanced feature [B, 64, H, W]
```

#### 3.4.2 Attention Module
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 126-160)

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        
        # Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Generate Key and Value from input (x)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, 
                                    padding=1, groups=dim * 2, bias=bias)
        
        # Generate Query from guidance feature (y)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, 
                                   padding=1, bias=bias)
        
        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x: Main feature (reflectance) [B, 64, H, W]
        # y: Guidance feature (illumination/semantic) [B, 64, H, W]
        
        b, c, h, w = x.shape

        # === Generate Keys and Values from main feature (x) ===
        kv = self.kv_dwconv(self.kv(x))      # [B, 128, H, W]
        k, v = kv.chunk(2, dim=1)            # Each: [B, 64, H, W]
        
        # === Generate Query from guidance (y) ===
        q = self.q_dwconv(self.q(y))         # [B, 64, H, W]

        # === Reshape for multi-head attention ===
        # [B, 64, H, W] → [B, num_heads, C_per_head, H*W]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # === Normalize Q and K ===
        q = torch.nn.functional.normalize(q, dim=-1)  # L2 normalize
        k = torch.nn.functional.normalize(k, dim=-1)

        # === Compute attention scores ===
        # Q @ K^T: [B, heads, C, HW] @ [B, heads, HW, C] → [B, heads, C, C]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)          # Normalize attention weights

        # === Apply attention to values ===
        # attn @ V: [B, heads, C, C] @ [B, heads, C, HW] → [B, heads, C, HW]
        out = (attn @ v)

        # === Reshape back to spatial ===
        # [B, heads, C, HW] → [B, 64, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                       head=self.num_heads, h=h, w=w)

        # === Project output ===
        out = self.project_out(out)          # [B, 64, H, W]
        
        return out
```

#### 3.4.3 SKFF (Selective Kernel Feature Fusion)
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 192-229)

```python
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels/reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1), nn.LeakyReLU(0.2))
        self.fcs = nn.ModuleList([nn.Conv2d(d, in_channels, 1) for _ in range(height)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        # Fuse multi-scale features with attention
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = self.softmax(torch.stack(attention_vectors, dim=1))
        
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V
```

#### 3.4.4 RTFU (Resolution-Transformable Feature Upsampler)
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 673-710)

```python
class RTFU(nn.Module):
    def __init__(self, dim, scale, kernel_size=3, bias=False):
        super(RTFU, self).__init__()
        self.depth = int(np.log2(scale)) + 1
        
        # Multi-scale feature generation
        self.RTM = ms_fea_fusion(dim=dim, scale=scale, bias=bias)
        
        # Final fusion
        self.final1_conv = nn.Conv2d(self.depth*dim, dim, 1)
        self.final_conv = nn.Conv2d(dim, 3, 1)
        
    def forward(self, inp):
        # Stage 1: Generate multi-scale features via bilinear upsampling
        out_stage = []  # Features at different scales
        
        # Stage 2: Fuse with RTM (Resolution-Transformable Module)
        out_stage2 = self.RTM(out_stage)
        
        # Stage 3: Upsample all to target resolution and concatenate
        out = torch.cat(out_stage3, dim=1)
        out = self.final_conv(self.final1_conv(out))
        return out
```

### 3.5 HRNet Semantic Backbone
**File:** [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) (Lines 14-32)

```python
def create_hrnet():
    args = {}
    args['cfg'] = './basicsr/models/archs/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    update_config(config, args)
    
    model = seg_hrnet.get_seg_model(config)
    
    # Load pretrained weights
    pretrained_dict = torch.load('./experiments/pretrained_models/hrnet_w48_pascal_context_cls59_480x480.pth')
    model.load_state_dict(pretrained_dict)
    return model
```

---

## 4. Loss Function Implementation

### 4.1 Model Training Loss Computation
**File:** [basicsr/models/image_restoration_model.py](basicsr/models/image_restoration_model.py) (Lines 159-230)

```python
def optimize_parameters(self, current_iter):
    self.optimizer_g.zero_grad()
    
    # Forward pass
    _, nllr_ill, img_nlsr = self.net_g(self.lq, self.atten)
    nllr_gray = self.atten[:, 1, :, :].unsqueeze(1)
    
    self.output = img_nlsr[-1]
    
    # Initialize losses
    l_pix_nlsr = 0.
    l_g_percep_nlsr = 0.
    l_smooth_ill_nllr = 0.
    l_color_ill_nllr = 0.
    
    # Pixel loss on SR output
    for pred in img_nlsr:
        l_pix_nlsr += self.cri_pix(pred, self.gt)
        l_g_percep, l_g_style = self.cri_perceptual(pred, self.gt)
        if l_g_percep is not None:
            l_g_percep_nlsr += l_g_percep
    
    # Illumination smoothness loss
    for img_atten, img_attengt in zip(nllr_ill, nllr_gray):
        l_smooth_ill_nllr += self.cri_SmoothL1(img_atten, img_attengt)
    
    # Illumination color loss
    for img_atten in nllr_ill:
        l_color_ill_nllr += self.cri_IllColor(img_atten)
    
    # Total loss
    l_total = l_pix_nlsr + l_g_percep_nlsr + l_smooth_ill_nllr + l_color_ill_nllr
    
    l_total.backward()
    if self.opt['train']['use_grad_clip']:
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
    self.optimizer_g.step()
```

### 4.2 Loss Function Classes

#### 4.2.1 L1Loss (Pixel Loss)
**File:** [basicsr/models/losses/losses.py](basicsr/models/losses/losses.py) (Lines 155-180)

```python
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)
```

#### 4.2.2 PerceptualLoss (VGG Feature Loss)
**File:** [basicsr/models/losses/losses.py](basicsr/models/losses/losses.py) (Lines 246-349)

```python
class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True,
                 perceptual_weight=1.0, style_weight=0., criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
        # VGG feature extractor
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)
        
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        
        # Perceptual loss
        percep_loss = 0
        for k in x_features.keys():
            percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        percep_loss *= self.perceptual_weight
        
        return percep_loss, style_loss
```

#### 4.2.3 SmoothL1 (Illumination Smoothness)
**File:** [basicsr/models/losses/losses.py](basicsr/models/losses/losses.py) (Lines 459-466)

```python
class SmoothL1(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SmoothL1, self).__init__()
        self.loss_weight = loss_weight 
        
    def forward(self, input_I, input_R):   
        smooth_loss = F.smooth_l1_loss(input_I, input_R)
        return self.loss_weight * smooth_loss
```

#### 4.2.4 IllColor (Illumination Color Consistency)
**File:** [basicsr/models/losses/losses.py](basicsr/models/losses/losses.py) (Lines 469-485)

```python
class IllColor(nn.Module):
    def __init__(self, loss_weight=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(IllColor, self).__init__()
        self.loss_weight = loss_weight
        self.mean = torch.from_numpy(np.array(mean)).cuda()
        self.std = torch.from_numpy(np.array(std)).cuda()
        self.relu = nn.ReLU()
        
    def forward(self, res):
        avg = torch.mean(torch.flatten(res, 2), dim=-1)
        dif = abs(avg - self.mean)
        dif = self.relu(dif - self.std)
        avg_error = torch.norm(torch.exp(dif) - 1, p=1, keepdim=True)
        color_loss = torch.mean(avg_error)
        return self.loss_weight * color_loss
```

#### 4.2.5 Other Available Losses

| Loss Class | Purpose | File Location |
|------------|---------|---------------|
| `MSELoss` | L2 pixel loss | Lines 182-203 |
| `CharbonnierLoss` | Robust L1 loss | Lines 220-230 |
| `PSNRLoss` | PSNR-based loss | Lines 205-217 |
| `L_color` | Color consistency loss | Lines 386-397 |
| `L_TV` | Total variation loss | Lines 410-422 |
| `L_SATV` | Structure-aware TV | Lines 432-456 |
| `SSIM` | SSIM loss | Lines 576-599 |

---

## 5. Results and Logging System

### 5.1 Logger Initialization
**File:** [basicsr/utils/logger.py](basicsr/utils/logger.py)

```python
def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    
    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(stream_handler)
    
    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        logger.addHandler(file_handler)
    
    return logger
```

### 5.2 Message Logger for Training Progress
**File:** [basicsr/utils/logger.py](basicsr/utils/logger.py) (Lines 12-77)

```python
class MessageLogger():
    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.max_iters = opt['train']['total_iter']
        self.tb_logger = tb_logger

    def __call__(self, log_vars):
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')
        
        message = f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:({lrs})]'
        
        # Add ETA
        eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
        message += f'[eta: {eta_str}]'
        
        # Add losses
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            # Log to TensorBoard
            if k.startswith('l_'):
                self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
        
        self.logger.info(message)
```

### 5.3 TensorBoard Logger
**File:** [basicsr/utils/logger.py](basicsr/utils/logger.py) (Lines 79-84)

```python
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger
```

### 5.4 Validation and Metrics Logging
**File:** [basicsr/models/image_restoration_model.py](basicsr/models/image_restoration_model.py) (Lines 280-345)

```python
def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    with_metrics = self.opt['val'].get('metrics') is not None
    
    for idx, val_data in enumerate(dataloader):
        self.feed_data(val_data)
        self.nonpad_test()
        
        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
        gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
        
        if save_img:
            save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}_{current_iter}.png')
            imwrite(sr_img, save_img_path)
        
        if with_metrics:
            for name, opt_ in opt_metric.items():
                metric_type = opt_.pop('type')
                self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
    
    # Log metrics
    self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
    log_str = f'Validation {dataset_name},'
    for metric, value in self.metric_results.items():
        log_str += f' # {metric}: {value:.4f}'
        tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
    logger.info(log_str)
```

### 5.5 Output Directory Structure

```
experiments/
└── {experiment_name}/
    ├── models/                 # Saved model checkpoints
    │   ├── net_g_3000.pth
    │   ├── net_g_6000.pth
    │   └── net_g_latest.pth
    ├── training_states/        # Optimizer/scheduler states for resume
    │   ├── 3000.state
    │   └── 6000.state
    ├── visualization/          # Validation images (if save_img=true)
    │   └── {image_name}/
    │       └── {image_name}_{iter}.png
    └── train_{name}_{timestamp}.log  # Training log file

tb_logger/
└── {experiment_name}/          # TensorBoard logs
    └── events.out.tfevents.*

results/
└── {experiment_name}/          # Test results
    ├── visualization/
    └── test_{name}_{timestamp}.log
```

---

## 6. Configuration Options

### 6.1 Configuration File Structure
**File:** [Super_Resolution/Options/CSDLLSR_v9_7_5_3_scale4.yml](Super_Resolution/Options/CSDLLSR_v9_7_5_3_scale4.yml)

### 6.2 General Settings

```yaml
# general settings
name: UltraISRound2_v9_7_5_retrain_scale4   # Experiment name
model_type: CSDLLSRv4                        # Model class (in image_restoration_model.py)
scale: 4                                     # Super-resolution scale factor
num_gpu: 1                                   # Number of GPUs (0 for CPU)
manual_seed: 100                             # Random seed for reproducibility
```

### 6.3 Dataset Options

```yaml
datasets:
  train:
    name: TrainSet
    type: Dataset_PairedImage              # Dataset class name
    dataroot_gt: /path/to/HR/images        # Ground truth (high resolution)
    dataroot_lq: /path/to/LR/images        # Low quality input
    geometric_augs: true                   # Enable flip/rotation augmentation
    use_illguidance: true                  # Enable illumination guidance input
    
    # Data loader settings
    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 8
    
    # Progressive training settings
    mini_batch_sizes: [8, 5, 4, 2, 1, 1]   # Batch sizes per stage
    iters: [46000, 32000, 24000, 18000, 18000, 12000]  # Iterations per stage
    gt_size: 384                           # Maximum patch size
    gt_sizes: [96, 128, 160, 224, 288, 320]  # Patch sizes per stage
    
  val:
    name: ValSet
    type: Dataset_PairedImage
    dataroot_gt: /path/to/val/HR
    dataroot_lq: /path/to/val/LR
    use_illguidance: true
```

### 6.4 Network Architecture Options

```yaml
network_g:
  type: CSDLLSRNetv9_7_5        # Network class (in UltraIS_arch.py)
  inp_channels: 3                # Input channels
  out_channels: 3                # Output channels
  n_feat: 64                     # Feature dimension
  scale: 4                       # Upscaling factor
```

### 6.5 Training Options

```yaml
train:
  total_iter: 150000             # Total training iterations
  warmup_iter: -1                # Warmup iterations (-1 for none)
  use_grad_clip: true            # Enable gradient clipping
  
  # Learning rate scheduler
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods: [46000, 104000]     # Restart periods
    restart_weights: [1, 1]
    eta_mins: [0.0003, 0.000001] # Minimum learning rates
  
  # Data augmentation
  mixing_augs:
    mixup: true
    mixup_beta: 1.2
    use_identity: true
  
  # Optimizer
  optim_g:
    type: AdamW                  # Adam or AdamW
    lr: !!float 2e-4             # Learning rate
    weight_decay: !!float 1e-4
    betas: [0.9, 0.999]
```

### 6.6 Loss Function Options

```yaml
train:
  # Pixel loss
  pixel_opt:
    type: L1Loss                 # L1Loss, MSELoss, CharbonnierLoss
    loss_weight: 1
    reduction: mean
  
  # Perceptual loss
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv1_2': 0.1
      'conv2_2': 0.1
      'conv3_4': 1
      'conv4_4': 1
      'conv5_4': 1
    vgg_type: vgg19
    perceptual_weight: !!float 1.2
    style_weight: 0
    criterion: l1
  
  # Illumination smoothness loss
  IllSmoothL1_opt:
    type: SmoothL1
    loss_weight: 1
  
  # Illumination color loss
  IllColor_opt:
    type: IllColor
    loss_weight: 1
```

### 6.7 Validation Options

```yaml
val:
  window_size: 4                 # Pad to multiple of this size
  val_freq: !!float 3e3          # Validation frequency (iterations)
  save_img: false                # Save validation images
  rgb2bgr: true                  # Convert RGB to BGR for saving
  use_image: true                # Use saved images for metrics
  max_minibatch: 8
  
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 2
      test_y_channel: false
```

### 6.8 Logging Options

```yaml
logger:
  print_freq: 1000               # Print log every N iterations
  save_checkpoint_freq: !!float 3e3  # Save checkpoint every N iterations
  use_tb_logger: true            # Enable TensorBoard
  wandb:
    project: ~                   # Wandb project name (~ for disabled)
    resume_id: ~
```

### 6.9 Path Options

```yaml
path:
  pretrain_network_g: ~          # Path to pretrained model (~ for none)
  strict_load_g: true            # Strict model loading
  resume_state: ~                # Path to resume training state (auto-detected)
```

### 6.10 Available Options Reference

| Category | Option | Type | Description |
|----------|--------|------|-------------|
| **General** | `name` | str | Experiment name |
| | `model_type` | str | Model class: `CSDLLSRv4` |
| | `scale` | int | 2 or 4 |
| | `num_gpu` | int | 0 for CPU |
| **Dataset** | `type` | str | `Dataset_PairedImage` |
| | `geometric_augs` | bool | Flip/rotation augmentation |
| | `use_illguidance` | bool | Enable illumination input |
| | `use_grayatten` | bool | Use grayscale attention |
| **Network** | `type` | str | `CSDLLSRNetv9_7_5` |
| | `n_feat` | int | Feature channels (default: 64) |
| **Scheduler** | `type` | str | See lr_scheduler.py |
| **Optimizer** | `type` | str | `Adam`, `AdamW` |
| **Losses** | `pixel_opt.type` | str | `L1Loss`, `MSELoss`, `CharbonnierLoss` |
| | `perceptual_opt` | dict | VGG perceptual loss |

---

## 7. Key Code References

### 7.1 Quick Reference Table

| Functionality | File | Key Lines |
|--------------|------|-----------|
| **Main training entry** | [train.py](basicsr/train.py) | `main()` L124-355 |
| **Main testing entry** | [test.py](basicsr/test.py) | `main()` L12-70 |
| **Model class** | [image_restoration_model.py](basicsr/models/image_restoration_model.py) | `CSDLLSRv4` L53-405 |
| **Network architecture** | [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | `CSDLLSRNetv9_7_5` L797-861 |
| **Illumination module** | [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | `Illumination` L432-498 |
| **Enhancement module** | [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | `Enhancement` L712-790 |
| **TransformerBlock** | [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | `TransformerBlock` L163-188 |
| **SKFF fusion** | [UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) | `SKFF` L192-229 |
| **Loss functions** | [losses.py](basicsr/models/losses/losses.py) | All losses L1-652 |
| **Dataset** | [paired_image_dataset.py](basicsr/data/paired_image_dataset.py) | `Dataset_PairedImage` L18-200 |
| **Config parsing** | [options.py](basicsr/utils/options.py) | `parse()` L29-85 |
| **Logging** | [logger.py](basicsr/utils/logger.py) | `MessageLogger` L12-77 |
| **LR schedulers** | [lr_scheduler.py](basicsr/models/lr_scheduler.py) | All schedulers L1-233 |
| **Model saving/loading** | [base_model.py](basicsr/models/base_model.py) | L214-305 |

### 7.2 Adding New Components

#### Adding a New Loss Function
1. Create class in [basicsr/models/losses/losses.py](basicsr/models/losses/losses.py)
2. Add to YAML config under `train` section
3. Initialize in [image_restoration_model.py](basicsr/models/image_restoration_model.py) `init_training_settings()`

#### Adding a New Network Architecture
1. Create class in [basicsr/models/archs/UltraIS_arch.py](basicsr/models/archs/UltraIS_arch.py) or new file ending with `_arch.py`
2. Update `network_g.type` in YAML config

#### Adding a New Dataset
1. Create class in [basicsr/data/](basicsr/data/) with filename ending in `_dataset.py`
2. Update `datasets.train.type` in YAML config

---

## Summary

The UltraIS framework implements a **Dual-Stream-Modulated Learning** approach for joint illumination enhancement and super-resolution:

1. **Illumination Stream**: U-Net style network estimates illumination map
2. **Semantic Stream**: Pretrained HRNet provides semantic guidance
3. **Reflectance Enhancement**: Main restoration network with cross-attention modulation from both streams
4. **Progressive Training**: Gradually increases patch size for stable training

Key innovations:
- **TransformerBlock** for cross-attention between restoration features and guidance features
- **SKFF** for multi-scale feature fusion
- **RTFU** for resolution-transformable upsampling
- Retinex-based decomposition (Reflectance = Image / Illumination)
