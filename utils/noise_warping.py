import imageio

import numpy as np
import rp
import torch
from einops import rearrange


def downscale_noise(noise, downscale_factor=4):
    down_noise = rp.torch_resize_image(noise, 1 / downscale_factor, interp='area')  
    down_noise = down_noise * downscale_factor 
    return down_noise


def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False)
    # unique_colors, inverse_indices, counts = torch.unique_consecutive(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output

def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image

_arange_cache={}
def _cached_arange(length, device, dtype):
    code=hash((length,device,dtype))
    if code in _arange_cache:
        return _arange_cache[code]

    
    _arange_cache[code]= torch.arange(length , device=device, dtype=dtype)
    return _arange_cache[code]

def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
    # assert rp.r.is_torch_image(image), "image must be a torch tensor with shape [C, H, W]"
    # assert is_torch_tensor(x) and is_a_matrix(x), "x must be a torch tensor with shape [H_out, W_out]"
    # assert is_torch_tensor(y) and is_a_matrix(y), "y must be a torch tensor with shape [H_out, W_out]"
    # assert x.shape == y.shape, "x and y must have the same shape, but got x.shape={} and y.shape={}".format(x.shape, y.shape)
    # assert image.device==x.device==y.device, "all inputs must be on the same device"

    # pip_import('torch')

    import torch

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if torch.is_floating_point(x): x = x.round_().long()
    if torch.is_floating_point(y): y = y.round_().long()

    if relative:
        # assert in_height == out_height, "For relative warping, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        # assert in_width  == out_width , "For relative warping, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x += _cached_arange(in_width , device=x.device, dtype=x.dtype)
        y += _cached_arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    x.clamp_(0, in_width - 1)
    y.clamp_(0,in_height-1)
    out = image[:, y, x]

    expected_c = in_c+1 if add_alpha_mask else in_c
    assert out.shape == (expected_c, out_height, out_width), "Expected output shape: ({}, {}, {}), but got: {}".format(expected_c, out_height, out_width, out.shape)

    return out

def warp_noise(noise, dx, dy, s=1):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    dx=dx.round_().int()
    dy=dy.round_().int()

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    if s!=1:
        up_dx = rp.torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
        up_dy = rp.torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
        up_dx *= s
        up_dy *= s

        up_noise = rp.torch_resize_image(noise, (hs, ws), interp="nearest")
    else:
        up_dx = dx
        up_dy = dy
        up_noise = noise
    assert up_noise.shape == (c, hs, ws)

    # Warp the noise - and put 0 where it lands out-of-bounds
    # up_noise = rp.torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest")
    up_noise = fast_nearest_torch_remap_image(up_noise, up_dx, up_dy, relative=True)
    assert up_noise.shape == (c, hs, ws)
    
    # Regaussianize the noise
    output, _ = regaussianize(up_noise)

    #Now we resample the noise back down again
    if s!=1:
        output = rp.torch_resize_image(output, (h, w), interp='area')
        output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output

def regaussianize(noise):
    c, hs, ws = noise.shape

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    assert output.shape == noise.shape == (c, hs, ws)

    return output, counts_image
    
@rp.memoized
def _xy_meshgrid(h,w,device,dtype):
    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
    )

    output = torch.stack(
        [x, y],
    ).to(device, dtype)

    assert output.shape == (2, h, w)
    return output

def xy_meshgrid_like_image(image):
    """
    Example:
        >>> image=load_image('https://picsum.photos/id/28/367/267')
        ... image=as_torch_image(image)
        ... xy=xy_meshgrid_like_image(image)
        ... display_image(full_range(as_numpy_array(xy[0])))
        ... display_image(full_range(as_numpy_array(xy[1])))
    """
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    return _xy_meshgrid(h,w,image.device,image.dtype)

def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros=torch.zeros_like(noise[0][None])
    ones =torch.ones_like (noise[0][None])

    #Prepend [dx=0, dy=0, weights=1] channels
    output=torch.concat([zeros, zeros, ones, noise])
    return output

def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0]>3, 'xyωc should have at least one noise channel'
    noise=xyωc[3:]
    return noise

def warp_xyωc(
    I,
    F,
    xy_mode="none",
    # USED FOR ABLATIONS:
    expand_only=False,
):
    """
    For ablations, set:
        - expand_only=True #No contraction
        - expand_only='bilinear' #Bilinear Interpolation
        - expand_only='nearest' #Nearest Neighbors Warping
    """
    #Input assertions
    assert F.device==I.device
    assert F.ndim==3, str(F.shape)+' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, str(I.shape)+' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    #How I'm going to address the different channels:
    x   = 0        #          // index of Δx channel
    y   = 1        #          // index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]     // index of weight channel
    c   = xyωc-xyω # I[-c:]   // num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    # h_dim = 1
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[-c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)

    #The original plan was to use init xy during expand, because the query position is arbitrary....
    #It doesn't actually make deep sense to copy the offsets during this step, but it doesn't seem to hurt either...
    #BUT I think I got slightly better results...?...so I'm going to do it anyway.
    # pre_expand[:xy] = init[:xy] # <---- Original algorithm I wrote on paper

    #ABLATION STUFF IN THIS PARAGRAPH
    #Using F_index instead of F so we can use ablations like bilinear, bicubic etc
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp=='nearest':
        #Default behaviour, ablations or not
        F_index=F_index.round()

    pre_expand[:xy] = rp.torch_remap_image(I[:xy], * -F, relative=True, interp=interp)# <---- Last minute change
    pre_expand[-ωc:] = rp.torch_remap_image(I[-ωc:], * -F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω]==0]=1 #Give new noise regions a weight of 1 - effectively setting it to init there

    if expand_only:
        if regauss:
            #This is an ablation option - simple warp + regaussianize
            #Enable to preview expansion-only noise warping
            #The default behaviour! My algo!
            pre_expand[-c:]=regaussianize(pre_expand[-c:])[0]
        else:
            #Turn zeroes to noise
            pre_expand[-c:]=torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:]==0) + pre_expand[-c:]
        return pre_expand

    #Calculate initial pre-shrink
    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0<= pos[x]) & (pos[x] < w) & (0<= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None] #Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype==torch.bool
    assert out_of_bounds.shape==(1,h,w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    #FLOATING POINT POSITIONS: I will disable this for now. It does in fact increase sensitivity! But it also makes it less long-term coherent
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode=='none':
        pre_shrink[:xy] = 0 #DEBUG: Uncomment to ablate floating-point swarm positions

    if isinstance(xy_mode, int):
        # XY quantization: best to use odd numbers!
        quant = xy_mode
        pre_shrink[:xy] = (
            pre_shrink[:xy] * quant
        ).round() / quant  

    #OTHER ways I tried reducing sensitivity to motion. They work - but 0 is best. Let's just use high resolution.
    # pre_shrink[:xy][pre_shrink[:xy].abs()<.1] = 0  #DEBUG: Uncomment to ablate floating-point swarm positions
    # pre_shrink[:xy] *= -1 #I can't even tell that this is wrong.....
    # pre_shrink[:xy] *= .9 
    # sensitivity_factor = 4

    scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True)

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device) #The purpose is to get zeroes where no element is used
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    # rp.cv_imshow(rp.tiled_images([out_of_bounds[0],shrink_mask[0]]),label='OOB') ; return I #DEBUG - uncomment to see the masks

    #Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)
    # rp.cv_imshow(pre_expand[-c:]/5+.5,'preex')

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, 2*w)
    # rp.cv_imshow(concat[-c:]/5+.5,label='regauss') ; return pre_expand #DEBUG - Uncomment to preview regaussianization

    #Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3+c, h, w)
 
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean() #Don't let them get too big or too small
    ε = .00001
    output[ω] += ε #Don't let it go too low
    
    # rp.debug_comment([output[ω].min(),output[ω].max()])# --> [tensor(0.0010), tensor(2.7004)]
    # rp.debug_comment([shrink[ω].min(),shrink[ω].max()])# --> [tensor(0.), tensor(2.7004)]
    # rp.debug_comment([expand[ω].min(),expand[ω].max()])# --> [tensor(0.0001), tensor(0.3892)]
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/output[ω].mean()/4),label='weight')
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/10),label='weight')
    assert (output[ω]>0).all()
    # print(end='\r%.08f %.08f'%(float(output[ω].min()), float(output[ω].max())))

    output[ω] **= .9999 #Make it tend towards 1


    return output


class NoiseWarper:
    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha = 0,
        progressive_noise_alpha = 0,
        warp_kwargs=dict(),
    ):

        #Some non-exhaustive input assertions
        assert isinstance(c,int) and c>0
        assert isinstance(h,int) and h>0
        assert isinstance(w,int) and w>0
        assert isinstance(scale_factor,int) and w>=1

        #Record arguments
        self.c=c
        self.h=h
        self.w=w
        self.device=device
        self.dtype=dtype
        self.scale_factor=scale_factor
        self.progressive_noise_alpha=progressive_noise_alpha
        self.post_noise_alpha=post_noise_alpha
        self.warp_kwargs=warp_kwargs

        #Initialize the state
        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        #TODO: The noise should be downsampled to respect the weights!! 
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None] #xyωc
        noise = (
              rp.torch_resize_image(noise * weights, (self.h, self.w), interp="area")
            / rp.torch_resize_image(weights**2     , (self.h, self.w), interp="area").sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise

    def __call__(self, dx, dy):

        if rp.is_numpy_array(dx): dx = torch.tensor(dx).to(self.device, self.dtype)
        if rp.is_numpy_array(dy): dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape #Original height and width of the flow
        
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow is in [x y]·h·w form"
        flow = rp.torch_resize_image(
            flow,
            (
                self.h * self.scale_factor,
                self.w * self.scale_factor,
            ),
        ) 

        _, flowh, floww = flow.shape

        #Multiply the flow values by the size change
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow)
        return self

    #The following three methods can be overridden in subclasses:

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow):

        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)

        return warp_xyωc(state, flow, **self.warp_kwargs)
    
def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise)      , alpha)
    elif isinstance(noise, np.ndarray): return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else: raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")

def resize_noise(noise, size, alpha=None):
    """
    Can resize gaussian noise, adjusting for variance and preventing cross-correlation
    """

    if rp.is_numpy_array(noise):
        noise=rp.as_torch_image(noise)
        output = resize_noise(noise, size, alpha)
        output = rp.as_numpy_array(output)
        output = einops.rearrange(output, "C H W -> H W C")
        return output

    assert noise.ndim == 3, "resize_noise: noise should be a CHW tensor"
    num_channels, old_height, old_width = noise.shape

    if noise.ndim==4:
        #If given a batch of noises, do it for each one
        return torch.stack([resize_noise(x, new_height, new_width) for x in noise])

    if rp.is_number(size):
        new_height, new_width = int(old_height * size), int(old_width * size)
    else:
        new_height, new_width = size

    assert new_height<=old_height, 'resize_noise: Only useful for shrinking noise, not growing it'
    assert new_width <=old_width , 'resize_noise: Only useful for shrinking noise, not growing it'
    
    x, y = rp.xy_torch_matrices(
        old_height,
        old_width,
        max_x=new_width,
        max_y=new_height,
    )

    if alpha is not None:
        #Prepend the alpha
        assert alpha.ndim==2,alpha.shape
        assert alpha.shape==noise.shape[1:],(alpha.shape,noise.shape)
        noise=torch.cat((alpha[None],noise))
        
    resized = rp.torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_height,
        width=new_width,
        interp='floor',
        prepend_ones=alpha is None
    )
    
    total, resized = resized[:1], resized[1:]

    adjusted = resized / total**.5

    return adjusted
    

def save_noise_as_png(tensor, filename):
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize to [0, 1]
    arr = (arr * 255).astype(np.uint8).transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
    imageio.imwrite(filename, arr)


import torch
import torchvision.transforms
from PIL import Image
import torch.nn.functional as F

def create_coordinate_grid(height, width, device):
    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)
    
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    coordinate_grid = torch.stack([grid_x, grid_y], dim=0)
    
    return coordinate_grid

def warp_image(image, flow, mode='bilinear', device='cuda'):
    """
    Args:
        image: Tensor, shape [B, C, H, W]
        flow: Tensor, shape [B, 2, H, W]
        mode: 'bilinear' or 'nearest'
        device: 'cuda'

    Returns:
        warped_image: Tensor, shape [B, C, H, W]
    """
    B, C, H, W = image.shape

    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow).float().to(device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    coordinate_grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
    coordinate_grid = coordinate_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    new_coordinates = coordinate_grid - flow  # [B, 2, H, W]

    new_coordinates[:, 0, :, :] = 2.0 * new_coordinates[:, 0, :, :] / (W - 1) - 1.0
    new_coordinates[:, 1, :, :] = 2.0 * new_coordinates[:, 1, :, :] / (H - 1) - 1.0
    sampling_grid = new_coordinates.permute(0, 2, 3, 1)
    warped_image = F.grid_sample(
        image,
        sampling_grid,
        mode=mode,
        padding_mode='border',
        align_corners=True
    )

    return warped_image
    

if __name__ == '__main__':
    warper = NoiseWarper(
        c = 3,
        h = 256,
        w = 256,
        device = 'cuda',
    )

    # dx, dy = raft_model(prev_video_frame, video_frame)
    predicted_flow = np.load("flow.npy")
    dx, dy = predicted_flow


    noise_ori = torch.randn(3, 256, 256, device=warper.device)
    warper._state = noise_to_xyωc(noise_ori)

    noise_warp = warper(dx, dy).noise

    save_noise_as_png(noise_ori, "noise_ori.png")
    save_noise_as_png(noise_warp, "noise_warp.png")

    noise_warp2 = warp_image(noise_ori.unsqueeze(0), predicted_flow)

    save_noise_as_png(noise_warp2.squeeze(0), "noise_warp2.png")
