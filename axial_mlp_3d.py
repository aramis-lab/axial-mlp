import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unfoldNd

def stack3d(x, patch_size):
    # Takes as input a tensor of shape (batch_size, channels, depth, height, width)
    # and outputs a stack of patches of shape
    # (batch_size, num_patches_along_depth, num_patches_along_height, num_patches_along_width, channels, patch_depth, patch_height, patch_width)
    # The argument patch_size can be either an int or a 3-tuple.
    batch_size, channels, depth, height, width = x.shape
    spatial_size = (depth, height, width)
    if isinstance(patch_size, int): patch_size = (patch_size, patch_size, patch_size)
    stride = patch_size
    num_patches = [ int((spatial_size[i] - patch_size[i])/float(stride[i])+1) for i in range(3) ]
    x = unfoldNd.unfoldNd(x, kernel_size=patch_size, dilation=1, padding=0, stride=stride).permute(0, 2, 1).reshape(-1, *num_patches, channels, *patch_size)
    return x

def unstack3d(x):
    # Performs the opposite of stack3d
    batch_size, nd, nh, nw, channels, patch_depth, patch_height, patch_width = x.shape
    num_patches = [nd, nh, nw]
    patch_size = [patch_depth, patch_height, patch_width]
    stride = patch_size
    x = x.reshape(batch_size, nd*nh*nw, channels*patch_depth*patch_height*patch_width).permute(0, 2, 1)
    output_size = [ (num_patches[i]-1)*stride[i] + patch_size[i] for i in range(3) ]
    x = unfoldNd.foldNd(x, output_size=tuple(output_size), kernel_size=(patch_depth, patch_height, patch_width), stride=stride)
    return x

class LinearAlongDims1(nn.Module):
    def __init__(self, num_dims, dims, input_sizes, output_sizes, bias=True, dropout=None):
        super().__init__()
        if isinstance(dims, int): dims = [dims]
        if isinstance(input_sizes, int): input_sizes = [input_sizes]
        if isinstance(output_sizes, int): output_sizes = [output_sizes]
        
        order = np.argsort(dims)
        self.dims = [dims[j] for j in order]
        self.input_sizes = [input_sizes[j] for j in order]
        self.output_sizes = [output_sizes[j] for j in order]
        self.num_dims = num_dims
        
        perm = list(range(num_dims))
        for d in reversed(self.dims):
            perm = perm[:d] + perm[d+1:] + [perm[d]]
        self.perm = perm
        self.inv_perm = np.argsort(perm)
        
        output_view = num_dims*[1]
        for i,d in enumerate(self.dims):
            output_view[d] = self.output_sizes[i]
        output_view = [output_view[j] for j in self.perm]
        self.output_view = output_view
        
        self.linear = nn.Linear(np.prod(input_sizes), np.prod(output_sizes), bias=bias)

    def forward(self, x):
        x = x.permute(*self.perm)
        x = x.flatten(start_dim=self.num_dims-len(self.dims))
        x = self.linear(x)
        x = x.view(*x.shape[:-1], *self.output_view[-len(self.dims):])
        x = x.permute(*self.inv_perm)
        return x

class LinearAlongDims2(nn.Module):
    # This implementation uses torch.einsum
    # It uses slightly less memory but is slightly slower
    def __init__(self, num_dims, dims, input_sizes, output_sizes, bias=True):
        super().__init__()
        if isinstance(dims, int): dims = [dims]
        if isinstance(input_sizes, int): input_sizes = [input_sizes]
        if isinstance(output_sizes, int): output_sizes = [output_sizes]
            
        order = np.argsort(dims)
        dims = [dims[j] for j in order]
        input_sizes = [input_sizes[j] for j in order]
        output_sizes = [output_sizes[j] for j in order]
            
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        einstr = alphabet[:num_dims] + ','
        last_letter_index = num_dims-1
        
        self.weight = nn.Parameter(torch.rand(np.array([input_sizes, output_sizes]).T.flatten().tolist()))
        stdv = 1. / np.sqrt(np.prod(input_sizes))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            bias_shape = num_dims*[1]
            for i,j in enumerate(dims):
                bias_shape[j] = output_sizes[i]
            self.bias = nn.Parameter(torch.zeros(*bias_shape))

        output = '->'
        for i in range(num_dims):
            if i in dims:
                einstr += einstr[i]
                einstr += alphabet[last_letter_index+1]
                output += alphabet[last_letter_index+1]
                last_letter_index += 1
            else:
                output += einstr[i]
        self.einstr = einstr + output
        print('Einstein string:', self.einstr)
        
        self.dims = dims
        
    def forward(self, x):
        x = torch.einsum(self.einstr, x, self.weight)
        if hasattr(self, 'bias'):
            x += self.bias
        return x
    
class AxialDropout(nn.Module):
    def __init__(self, p, dims=None):
        super().__init__()
        if isinstance(dims, int): dims = [dims]
        self.p = p
        self.dims = dims # dims along which the drops are independent (typically 0 for batch dim and -1 for the last dim of a linear layer)
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        if self.p == 0: return x
        if self.dims is not None:
            shape = torch.ones(x.dim(), dtype=torch.int32)
            shape[self.dims] = torch.tensor(x.shape, dtype=torch.int32)[self.dims]
            mask = torch.ones(*shape, device=x.device)
        mask = self.dropout(mask)
        return x*mask
    
class Block3d(nn.Module):
    def __init__(self, input_shape, output_shape, normalize=True, dropout=None):
        super().__init__()
        
        assert len(input_shape) == len(output_shape)

        def make_linear(dims):
            return nn.Sequential(
                AxialDropout(p=dropout, dims=[0] + dims) if dropout is not None else nn.Identity(),
                LinearAlongDims1(num_dims=8, dims=dims, input_sizes=[input_shape[d-1] for d in dims], output_sizes=[output_shape[d-1] for d in dims], bias=True)
            )
        
        self.patch_d = make_linear([1, 5])
        self.patch_h = make_linear([1, 6])
        self.patch_w = make_linear([1, 7])
        self.image_d = make_linear([1, 2])
        self.image_h = make_linear([1, 3])
        self.image_w = make_linear([1, 4])
        
        self.gn = nn.GroupNorm(num_groups=1, num_channels=input_shape[0])
        
    def forward(self, x):
        # x has shape b, c, nd, nh, nw, pd, ph, pw
        
        y = self.patch_d(x) + self.patch_h(x) + self.patch_w(x) + self.image_d(x) + self.image_h(x) + self.image_w(x)
        
        if x.shape == y.shape:
            y = F.leaky_relu(y)
            y = x + y

        x = self.gn(x)
        
        return y
    
class AxialMLP3d(nn.Module):
    def __init__(self, patch_size, input_size, in_channels, out_channels, num_layers=6, filters=8, dropout_rate=0):
        super().__init__()
        self.filters = filters
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        if isinstance(input_size, int):
            input_size = (input_size, input_size, input_size)
        patch_size_tensor = torch.tensor(patch_size)
        input_size_tensor = torch.tensor(input_size)
        self.patch_size = patch_size
        self.interm_size = ((input_size_tensor/patch_size_tensor).int())*patch_size_tensor
        stride = patch_size_tensor
        self.num_patches = ( (self.interm_size - patch_size_tensor)/stride +1).int().tolist()
        self.interm_size = self.interm_size.tolist()
        self.N = self.num_patches[0]*self.num_patches[1]*self.num_patches[2]
        print(f'Input will be resized to', self.interm_size, 'and number of patches is', self.num_patches, '=', self.N)
        input_shape = (filters, self.num_patches[0], self.num_patches[1], self.num_patches[2], *patch_size)
        self.blocks = nn.ModuleList([
            Block3d(
                input_shape=input_shape,
                output_shape=input_shape,
                dropout=dropout_rate if dropout_rate > 0 else None
            ) for _ in range(num_layers)
        ])
        
        self.input_proj = LinearAlongDims1(num_dims=8, dims=[1], input_sizes=[in_channels], output_sizes=[self.filters])
        self.output_proj = LinearAlongDims1(num_dims=5, dims=[1], input_sizes=[self.filters], output_sizes=[out_channels])

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = F.interpolate(x, self.interm_size, mode='trilinear', align_corners=True)
        x = stack3d(x, patch_size=self.patch_size).permute(0, 4, 1, 2, 3, 5, 6, 7)
        x = self.input_proj(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            
        x = unstack3d(x.permute(0, 2, 3, 4, 1, 5, 6, 7))
        x = self.output_proj(x)
        x = F.interpolate(x, (d, h, w), mode='trilinear', align_corners=True)
        return x
