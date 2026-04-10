import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange
from basicsr.models.archs.hrseg_lib.models import seg_hrnet
from basicsr.models.archs.hrseg_lib.config import config
from basicsr.models.archs.hrseg_lib.config import update_config
from ipdb import set_trace as st

 

def create_hrnet():
    args = {}
    args['cfg'] = './basicsr/models/archs/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    args['opt'] = []
    update_config(config, args)
    if torch.__version__.startswith('1'):
        module = eval('seg_hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
 

    pretrained_dict = torch.load('./experiments/pretrained_models/hrnet_w48_pascal_context_cls59_480x480.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('HRNet load')
    return model

 
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

 
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        # Expand feature dimension
        hidden_features = int(dim * ffn_expansion_factor)  # 64 * 2.66 = 170

        # Custom GELU activation
        self.gelu = GELU()

        # Step 1: Expand channels and split into two paths
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # Input: [B, 64, H, W] → Output: [B, 340, H, W]

        # Step 2: Depthwise convolution for spatial mixing
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 
                               kernel_size=3, stride=1, padding=1,
                               groups=hidden_features * 2, bias=bias)
        # Each channel processed independently

        # Step 3: Project back to original dimension
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # Input: [B, 170, H, W] → Output: [B, 64, H, W]

    def forward(self, x):
        # Input: [B, 64, H, W]
        
        # Expand to 2× hidden features
        x = self.project_in(x)               # [B, 340, H, W]
        
        # Apply depthwise convolution and split
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # Each: [B, 170, H, W]
        
        # Gated activation: GELU(x1) ⊙ x2
        x = self.gelu(x1) * x2               # [B, 170, H, W]
        
        # Project back to original dimension
        x = self.project_out(x)              # [B, 64, H, W]
        
        return x
 
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
 
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V     

  
class ContextBlock(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        
        # Feature extraction head with grouped convolutions
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2),
            # 3x3 convolution with groups=2 (depthwise-like) to extract features
            nn.LeakyReLU(0.2, inplace=True),
            # Non-linearity activation
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2)
            # Another grouped convolution for deeper feature extraction
        )

        # Generate attention mask for spatial locations
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        # 1x1 conv reduces channels to 1 (creates attention weights per pixel)
        
        self.softmax = nn.Softmax(dim=2)
        # Softmax across spatial dimension to normalize attention

        # Channel-wise feature modulation
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            # 1x1 conv to process context
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            # Another 1x1 conv for channel modulation
        )
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def modeling(self, x):
        # x shape: [N, C, H, W]
        batch, channel, height, width = x.size()
        input_x = x
        
        # Reshape: [N, C, H*W] - flatten spatial dimensions
        input_x = input_x.view(batch, channel, height * width)
        
        # Add dimension: [N, 1, C, H*W] - prepare for matmul
        input_x = input_x.unsqueeze(1)
        
        # Generate attention map: [N, 1, H, W]
        context_mask = self.conv_mask(x)
        
        # Reshape mask: [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)
        
        # Normalize attention: [N, 1, H*W] - softmax across spatial locations
        context_mask = self.softmax(context_mask)
        
        # Add dimension: [N, 1, H*W, 1] - prepare for weighted sum
        context_mask = context_mask.unsqueeze(3)
        
        # Weighted sum of features: [N, 1, C, 1]
        # Each channel is weighted by spatial attention
        context = torch.matmul(input_x, context_mask)
        
        # Reshape to: [N, C, 1, 1] - global context descriptor
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        inp = x
        # Extract features using head
        inp = self.head(inp)
        
        # Generate context: [N, C, 1, 1]
        context = self.modeling(inp)

        # Modulate context through channel attention
        channel_add_term = self.channel_add_conv(context)
        
        # Add modulated context to extracted features
        inp = inp + channel_add_term
        
        # Residual connection: add back to original input
        x = x + self.act(inp)

        return x
 
class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential( 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            # nn.BatchNorm2d(n_feat),
            act
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += x
        return res
    
    
class RCBup(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBup, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        # Feature extraction from concatenated input (2*n_feat channels)
        self.body = nn.Sequential(
            # First convolution: reduce from 2*n_feat to n_feat
            nn.Conv2d(2*n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            # Second convolution: maintain n_feat channels
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        
        # Global context modeling
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        # Input x has 2*n_feat channels (from concatenation)
        # Extract features through body
        x = self.body(x)
        
        # Apply context block and activation
        res = self.act(self.gcnet(x))
        
        # Add residual connection (now matching reduced channel size)
        res += x
        
        return res
     
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()
        # nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x
  
class UpSampleModule(nn.Module):
    def __init__(self, num_kernels):
        super(UpSampleModule, self).__init__()
 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
         
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=256, out_channels=3*4, kernel_size=3, stride=1, padding=1)
                                    for _ in range(num_kernels)])
 
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=num_kernels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        kernel_weights = self.gate(out)         
        kernel_outputs = [conv(out) for conv in self.conv3]
        weighted_outputs = [kernel_weights[:, i:i+1, :, :] * kernel_output for i, kernel_output in enumerate(kernel_outputs)]
         
        out = torch.stack(weighted_outputs, dim=1).sum(dim=1)          
        out = self.pixel_shuffle(out)
        return out
 


class Illumination(nn.Module):
    def __init__(self, inp_channels=2, out_channels=1, n_feat=64, scale=1, bias=False):
        super(Illumination, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat) 
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv6 = RCBup(n_feat=n_feat) 
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv7 = RCBup(n_feat=n_feat) 
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv8 = RCBup(n_feat=n_feat) 
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv9 = RCBup(n_feat=n_feat) 
        self.conv10_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)
 
        
        self.conv_out_r = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, refl): 
        
        conv1 = self.lrelu(self.conv_in(refl))        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)        
        conv5 = self.conv5(pool4)         
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)        
        out = self.conv10_1(conv9) 
        out = torch.sigmoid(self.conv_out_r(out))
        
        return [conv5, conv6, conv7, conv8, conv9], out 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
 
class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        # Main feature extraction path
        self.body = nn.Sequential(
            # First convolution: process features
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            # Second convolution: further feature refinement
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        
        # Global context block for capturing long-range dependencies
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        # Store input for residual connection
        inp = x
        
        # Process through convolutions
        x = self.body(x)
        
        # Apply context block and activation
        res = self.act(self.gcnet(x))
        
        # Add residual connection
        res += inp
        
        return res
    
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        # in_channels: Feature channels (64)
        # height: Number of input feature streams to fuse (2 for scale=2, 3 for scale=4)
        # reduction: Channel reduction ratio for attention (8)
        
        super(SKFF, self).__init__()
        
        self.height = height  # Number of feature branches
        
        # Calculate reduced channel dimension (minimum 4)
        d = max(int(in_channels/reduction), 4)  # 64/8 = 8
        
        # Global average pooling for spatial information aggregation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] → [B, C, 1, 1]
        
        # Channel attention encoder: reduce channels for efficiency
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),  # 64 → 8 channels
            nn.LeakyReLU(0.2)
        )

        # Create separate attention branches for each input stream
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            # Each branch generates attention weights for its corresponding stream
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
            # 8 → 64 channels per branch
        
        # Softmax to normalize attention across different streams
        self.softmax = nn.Softmax(dim=1)  # Normalize across height dimension

    def forward(self, inp_feats):
        # inp_feats: List of feature tensors from different scales
        # For scale=2: [feat_bot, feat_top] each [B, 64, H, W]
        # For scale=4: [feat_bot, feat_mid, feat_top]
        
        batch_size = inp_feats[0].shape[0]  # B
        n_feats = inp_feats[0].shape[1]      # 64 (channels)
        
        # === Step 1: Stack features along height dimension ===
        inp_feats = torch.cat(inp_feats, dim=1)  
        # Concatenate: [B, 64*height, H, W]
        
        # Reshape to separate height dimension
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, 
                                    inp_feats.shape[2], inp_feats.shape[3])
        # Shape: [B, height, 64, H, W]
        
        # === Step 2: Aggregate features from all streams ===
        feats_U = torch.sum(inp_feats, dim=1)  # Sum across height dimension
        # Shape: [B, 64, H, W] - unified feature representation
        
        # === Step 3: Generate global descriptor ===
        feats_S = self.avg_pool(feats_U)  # Global average pooling
        # Shape: [B, 64, 1, 1] - spatial information compressed
        
        feats_Z = self.conv_du(feats_S)  # Encode to compact representation
        # Shape: [B, 8, 1, 1] - reduced channel dimension

        # === Step 4: Generate attention weights for each stream ===
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # Each fc output: [B, 64, 1, 1]
        # List of height elements
        
        attention_vectors = torch.cat(attention_vectors, dim=1)
        # Shape: [B, 64*height, 1, 1]
        
        attention_vectors = attention_vectors.view(batch_size, self.height, 
                                                    n_feats, 1, 1)
        # Shape: [B, height, 64, 1, 1]
        
        # === Step 5: Normalize attention across streams ===
        attention_vectors = self.softmax(attention_vectors)
        # Softmax across height dimension
        # Each spatial location gets weighted combination of streams
        
        # === Step 6: Weighted fusion ===
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        # Element-wise multiplication + sum across height
        # Shape: [B, 64, H, W] - fused feature
        
        return feats_V

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
        
    def forward(self, inp):
        return inp


class bilinearup_block(nn.Module):
    def __init__(self, dim, out_dim, upscale, kernel_size=3, bias=False):
        super(bilinearup_block, self).__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=bias),
                                   nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=kernel_size, stride=1, padding=1, bias=bias), 
                                   nn.LeakyReLU(0.2))
 
        
    def forward(self, inp):
        output = self.block(inp)  
        return output
    
class bicubicup_block(nn.Module):
    def __init__(self, dim, upscale, kernel_size=3, bias=False) :
        super(bicubicup_block, self).__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=upscale, mode='bicubic', align_corners=bias),
                                   nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=1, padding=1, bias=bias), 
                                   nn.LeakyReLU(0.2))
 
        
    def forward(self, inp):
        output = self.block(inp)  
        return output
    

class ms_fea_fusion(nn.Module):
    def __init__(self, dim, scale, bias=False):
        # dim: Feature dimension (64)
        # scale: Upsampling factor (2 or 4)
        
        super(ms_fea_fusion, self).__init__()
        
        self.scale = scale
        
        # ========== Configuration for scale=2 ==========
        if scale == 2:
            # Feature refinement blocks for each scale
            self.dau_top = RCBdown(dim, bias=bias, groups=1)  # Process top (2×) scale
            self.dau_bot = RCBdown(dim, bias=bias, groups=1)  # Process bottom (1×) scale
            
            # === Upsampling operations (bot → top) ===
            self.up_x1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            # Upsample bottom feature to match top resolution
            
            self.up_x1_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            # (Unused in scale=2, kept for consistency)
            
            # === Downsampling operations (top → bot) ===
            self.down_x2_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            # Stride-2 conv to downsample top feature to bottom resolution
            
            self.down_x2_2 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            # (Unused in scale=2)
            
            # === SKFF for selective fusion ===
            self.skff_top = SKFF(dim, 2)  # Fuse 2 streams for top scale
            self.skff_bot = SKFF(dim, 2)  # Fuse 2 streams for bottom scale
            
            # === Output projection ===
            self.conv_out_top = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_bot = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
               
        # ========== Configuration for scale=4 ==========
        elif scale == 4:
            # Three scales: bot (1×), mid (2×), top (4×)
            self.dau_top = RCBdown(dim, bias=bias, groups=1)
            self.dau_mid = RCBdown(dim, bias=bias, groups=1)
            self.dau_bot = RCBdown(dim, bias=bias, groups=1)
            
            # === Upsampling operations ===
            self.up_x1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            # bot → mid
            
            self.up_x1_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=bias)
            # bot → top
            
            self.up_x2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            # mid → top
            
            # === Downsampling operations ===
            self.down_x2_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            # mid → bot (or top → mid)
            
            self.down_x4_1 = nn.Conv2d(dim, dim, 1, 2, 0, bias=bias)
            # top → mid
            
            self.down_x4_2 = nn.Conv2d(dim, dim, 1, 4, 0, bias=bias)
            # top → bot
            
            # === SKFF for 3-way fusion ===
            self.skff_top = SKFF(dim, 3)  # Fuse top + mid + bot
            self.skff_mid = SKFF(dim, 3)  # Fuse mid + top + bot
            self.skff_bot = SKFF(dim, 3)  # Fuse bot + mid + top
            
            # === Output projections ===
            self.conv_out_top = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_mid = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
            self.conv_out_bot = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)
        
    def forward(self, inp):
        # inp: List of features at different scales
        # For scale=2: [bot, top]
        # For scale=4: [bot, mid, top]
        
        # ========== Processing for scale=2 ==========
        if self.scale == 2:
            # Extract and clone features
            x_top = inp[1].clone()  # [B, 64, 2H, 2W]
            x_bot = inp[0].clone()  # [B, 64, H, W]
            
            # === Refine features ===
            x_top = self.dau_top(x_top)  # Context + residual
            x_bot = self.dau_bot(x_bot)
            
            # === Bidirectional fusion ===
            # Top scale: fuse top feature + upsampled bottom
            x_top = self.skff_top([x_top, self.up_x1_1(x_bot)])
            # Input: [top_feat, upsampled_bot] → Weighted fusion
            
            # Bottom scale: fuse bottom feature + downsampled top
            x_bot = self.skff_bot([x_bot, self.down_x2_1(x_top)])
            # Input: [bot_feat, downsampled_top] → Weighted fusion
            
            # === Output projection ===
            out_top = self.conv_out_top(x_top)  # [B, 64, 2H, 2W]
            out_bot = self.conv_out_bot(x_bot)  # [B, 64, H, W]
            
            return [out_bot, out_top]
        
        # ========== Processing for scale=4 ==========
        elif self.scale == 4:
            # Extract features at three scales
            x_top = inp[2].clone()  # [B, 64, 4H, 4W]
            x_mid = inp[1].clone()  # [B, 64, 2H, 2W]
            x_bot = inp[0].clone()  # [B, 64, H, W]
            
            # === Refine features ===
            x_top = self.dau_top(x_top)
            x_mid = self.dau_mid(x_mid)
            x_bot = self.dau_bot(x_bot)
            
            # === Tri-directional fusion ===
            # Top: fuse [top, mid↑, bot↑↑]
            x_top = self.skff_top([x_top, 
                                   self.up_x2_1(x_mid),  # mid → top
                                   self.up_x1_2(x_bot)])  # bot → top
            
            # Mid: fuse [top↓, mid, bot↑]
            x_mid = self.skff_mid([self.down_x4_1(x_top),  # top → mid
                                   x_mid, 
                                   self.up_x1_1(x_bot)])   # bot → mid
            
            # Bot: fuse [top↓↓, mid↓, bot]
            x_bot = self.skff_bot([self.down_x4_2(x_top),  # top → bot
                                   self.down_x2_1(x_mid),  # mid → bot
                                   x_bot])
            
            # === Output projections ===
            out_top = self.conv_out_top(x_top)  # [B, 64, 4H, 4W]
            out_mid = self.conv_out_mid(x_mid)  # [B, 64, 2H, 2W]
            out_bot = self.conv_out_bot(x_bot)  # [B, 64, H, W]
            
            return [out_bot, out_mid, out_top]
              
class RTFU(nn.Module):
    def __init__(self, dim, scale, kernel_size=3, bias=False):
        # dim: Feature dimension (64)
        # scale: Super-resolution factor (2 or 4)

        super(RTFU, self).__init__()

        # Calculate number of resolution levels
        # scale=2 → depth=2 (1×, 2×)
        # scale=4 → depth=3 (1×, 2×, 4×)
        self.depth = int(np.log2(scale)) + 1
        
        # ========== Stage 1: Progressive Bilinear Upsampling ==========
        self.up_ps_names = locals()  # Dictionary to store dynamic attributes
        for i in range(self.depth):
            # Create ModuleList for each scale level
            self.up_ps_names['self.bilinear_x%s', 2**i] = nn.ModuleList([])

            if i == 0:
                self.up_ps_names['self.bilinear_x%s', 2**i].append(identity())
            else:
                for j in range(i):
                    # Each block upsamples by 2×
                    self.up_ps_names['self.bilinear_x%s', 2**i].append(bilinearup_block(dim, dim, 2).cuda())

        # ========== Stage 2: Resolution-aware Transformer Module ==========  
        self.RTM = ms_fea_fusion(dim=dim, scale=scale, bias=bias)   
        # Performs cross-scale feature fusion using SKFF    
        
        # ========== Stage 3: Final Bicubic Upsampling ==========
        self.up_bicu_names = locals()

        for i in range(self.depth):
            self.up_bicu_names['self.bicubic_x%s', 2**i] = nn.ModuleList([])

            if i == (self.depth-1):
                # Highest resolution: No further upsampling
                self.up_bicu_names['self.bicubic_x%s', 2**i].append(identity())
            else:
                # Lower resolutions: Upsample to match highest resolution
                for j in range(self.depth-i-1):
                    self.up_bicu_names['self.bicubic_x%s', 2**i].append(bicubicup_block(dim=dim, upscale=2, kernel_size=3, bias=bias).cuda())

        self.final1_conv = nn.Conv2d(self.depth*dim, dim, 1, 1, 0, bias=bias)
        # Reduce concatenated channels: [depth×64] → [64]

        self.final_conv = nn.Conv2d(dim, 3, 1, 1, 0, bias=bias)
        # Project to RGB: [64] → [3]

        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, inp):
        # inp: Feature from Enhancement module [B, 64, H, W]

        # stage1: Progressive Upsampling
        out_stage = []
        for i in range(self.depth):
            temp = inp

            for op in self.up_ps_names['self.bilinear_x%s', 2**i]:
                temp = op(temp)
                # Level 0: identity → [B, 64, H, W]
                # Level 1: 1× bilinear → [B, 64, 2H, 2W]
                # Level 2: 2× bilinear → [B, 64, 4H, 4W]

            out_stage.append(temp) # List of features at multiple resolutions
            
        #stage2: Cross-scale Fusion with RTM
        out_stage2 = self.RTM(out_stage)
        # Apply ms_fea_fusion with SKFF
        # Input: [bot, mid, top] (or [bot, top] for scale=2)
        # Output: Refined multi-scale features with bidirectional information flow
        
        #stage3: Upsample All to Target Resolution
        out_stage3 = []

        for i in range(self.depth):
            temp = out_stage2[i]

            for op in self.up_bicu_names['self.bicubic_x%s', 2**i]:
                temp = op(temp)
                # Level (depth-1): identity → already at target size
                # Level (depth-2): 1× bicubic → match target
                # Level 0: (depth-1)× bicubic → match target

            out_stage3.append(temp)
        # All features now at same resolution: [B, 64, scale×H, scale×W]
        
        # ========== Stage 4: Final Fusion ==========
        out = torch.cat(out_stage3, dim=1)
        # Concatenate along channel: [B, depth×64, scale×H, scale×W]
        
        out = self.act(self.final1_conv(out))
        # Reduce channels: [B, 64, scale×H, scale×W]
        
        out = self.act(self.final_conv(out))
        # Project to RGB: [B, 3, scale×H, scale×W]
                
        return out
 
class Enhancement(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, n_feat=64, scale=1, bias=False, seg_dims=None):
        super(Enhancement, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat)
        self.sm5 = TransformerBlock(dim=n_feat, dim2=seg_dims[4])
        self.sm55 = TransformerBlock(dim=n_feat)
        
         
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = RCBup(n_feat=n_feat)
        self.sm6 = TransformerBlock(dim=n_feat, dim2=seg_dims[3])
        self.sm66 = TransformerBlock(dim=n_feat)
        
 
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv7 = RCBup(n_feat=n_feat)
        self.sm7 = TransformerBlock(dim=n_feat, dim2=seg_dims[2])
        self.sm77 = TransformerBlock(dim=n_feat)
        
         
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv8 = RCBup(n_feat=n_feat)
        self.sm8 = TransformerBlock(dim=n_feat, dim2=seg_dims[1])
        self.sm88 = TransformerBlock(dim=n_feat)
        
         
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv9 = RCBup(n_feat=n_feat)
        self.sm9 = TransformerBlock(dim=n_feat, dim2=seg_dims[0])
        self.sm99 = TransformerBlock(dim=n_feat)
        
        self.conv10_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)        
        self.tanh = nn.Tanh()        
        self.RTFU = RTFU(dim=n_feat, scale=scale)
        
 
    def forward(self, refl, seg_orin, seg_fea, ill_fea):
    
        conv1 = self.lrelu(self.conv_in(refl))
        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)        
        conv5 = self.conv5(pool4)     
         
        conv5 = self.sm55(conv5, ill_fea[0]) 
        conv5 = self.sm5(conv5, seg_fea[3])         
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)        
        conv6 = self.sm66(conv6, ill_fea[1]) 
        conv6 = self.sm6(conv6, seg_fea[2])        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)        
        conv7 = self.sm77(conv7, ill_fea[2]) 
        conv7 = self.sm7(conv7, seg_fea[1])        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)        
        conv8 = self.sm88(conv8, ill_fea[3]) 
        conv8 = self.sm8(conv8, seg_fea[0])        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        conv9 = self.sm99(conv9, ill_fea[4]) 
        conv9 = self.sm9(conv9, seg_orin)
        out = self.conv10_1(conv9) 
        out = self.RTFU(out)
                            
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
 

 
class CSDLLSRNetv9_7_5(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=64,
        scale=1,
        bias=False
    ):
        super(CSDLLSRNetv9_7_5, self).__init__()
        
        kernel_size=3
        self.n_feat = n_feat
        self.scale = scale
        self.seg_dims = [59, 48, 96, 192, 384]       
         
        self.illumination = Illumination(2, 1, n_feat, scale)        
        self.reflectance = Enhancement(inp_channels, n_feat, n_feat, scale, seg_dims=self.seg_dims)   
         
        self.seg = create_hrnet()
        for p in self.seg.parameters():
            p.requires_grad = False
            
 
        
        
    def forward(self, inp_img_lllr, inp_img_gray):

        _, seg_orin, seg_fea = self.seg(inp_img_lllr)
        ill_feas, nllr_ill = self.illumination(inp_img_gray)
        nllr_ill3 = torch.cat((nllr_ill, nllr_ill, nllr_ill), dim=1)
        nllr_ref = inp_img_lllr / nllr_ill3
        nllr_ref = torch.clamp(nllr_ref, 0, 1)
        nlsr_refl = self.reflectance(nllr_ref, seg_orin, seg_fea, ill_feas)
        img_nlsr = nlsr_refl          
        return nlsr_refl, nllr_ill, img_nlsr   
  

if __name__== '__main__':
    inp = torch.randn((1, 3, 128, 128))
    atten = torch.randn((1, 2, 128, 128))
    net = CSDLLSRNetv9_7_5(scale=2).eval()
    
    nlsr_refl, nlsr_ill, img_nlsr = net(inp, atten)
   

