import torch
from torch import nn
from einops import rearrange, reduce


class AdaptiveAvgPooling3D(nn.Module):
    def __init__(self, num_frames=8, output_tokens=576):
        super().__init__()
        # Calculate the target size for each dimension, assuming the number of frames is a power of 2
        # and the total desired output tokens are 512
        self.num_frames = num_frames
        # spatial_target_size = int((output_tokens / num_frames) ** 0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((num_frames, 8, 9))

    def forward(self, x):
        # Assuming input x has the shape '(b t) (h w) c', where h*w=512, and t is num_frames which is a power of 2
        # Calculate spatial dimensions assuming square frames
        h = w = int(x.shape[1] ** 0.5)
        # Reshape for 3D pooling: (batch, channel, depth, height, width)
        x = rearrange(x, '(b t) (h w) c -> b c t h w', t=self.num_frames, h=h, w=w)
        # Apply adaptive average pooling
        x = self.adaptive_pool(x)
        # Flatten the spatial and temporal dimensions to a single dimension per batch
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x


class AdaptiveAvgPooling2DPerFrame(nn.Module):
    def __init__(self, num_frames=8, output_tokens=512):
        super().__init__()
        # Determine the number of tokens per frame to pool to, ensuring the total remains 512 per batch
        self.tokens_per_frame = output_tokens // num_frames
        self.num_frames = num_frames
        # Since we're using a square frame assumption, calculate the target size for each frame
        self.pool_target_size = int(self.tokens_per_frame ** 0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.pool_target_size, self.pool_target_size))

    def forward(self, x):
        # Assuming input x has the shape '(b t) (h w) c', where h*w is the number of pixels per frame and t is num_frames
        # Calculate spatial dimensions assuming square frames
        h = w = int((x.shape[1]) ** 0.5)
        # Reshape to '(b t) c h w' to separate all frames for pooling
        x = rearrange(x, '(b t) (h w) c -> (b t) c h w', t=self.num_frames, h=h, w=w)
        # Apply adaptive pooling to all frames at once
        x = self.adaptive_pool(x)
        # Flatten back to the frame sequence dimension per batch
        x = rearrange(x, '(b t) c h w -> b (t h w) c', t=self.num_frames)
        return x


class AveragePoolingAcrossFrames(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x):
        x = average_pooling_across_frames(x, self.num_frames)
        return x

def average_pooling_across_frames(x, num_frames):
    # Reshape to separate batch, frames, height*width, and channels
    x = rearrange(x, '(b t) (h w) c -> b t (h w) c', t=num_frames)
    # Apply average pooling across frames
    x = reduce(x, 'b t (h w) c -> b (h w) c', 'mean')
    return x


class TransformerDecoderWithQueries(nn.Module):
    def __init__(self, num_frames, embed_dim, num_heads, num_queries=512):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        self.num_frames = num_frames

    def forward(self, x):
        # Reshape to 't h w, b, c' and then attend using learned queries
        x = rearrange(x, '(b t) (h w) c -> (t h w) b c', t=self.num_frames)
        queries = self.queries.unsqueeze(1).repeat(1, x.shape[1], 1)  # Repeat queries for each batch
        x = self.decoder(queries, x)
        x = rearrange(x, 's b c -> b s c')  # Reshape to batch, sequence, channel
        return x
