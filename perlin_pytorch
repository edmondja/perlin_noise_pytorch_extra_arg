# Perlin noise in PyTorch improved from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# Added batch_size, num_channels, random seed, device
import torch
import math

def rand_perlin_2d(batch_size, num_channels, shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3, 
                   generator=None, device='cpu'):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1).to(device)
    angles = 2*math.pi*torch.rand((batch_size, num_channels, res[0]+1, res[1]+1), generator=generator, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)

    tile_grads = lambda slice1, slice2: gradients[:, :, slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 2).repeat_interleave(d[1], 3)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1)[None][None] * 
                               grad[:, :, :shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])[None][None]
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(batch_size, num_channels, shape, octaves=5, persistence=0.5, generator=None, device='cpu'):
    res = (int(shape[-2]*8/255), int(shape[-1]*8/255)) # Set as a default value proportional to the image size
    noise = torch.zeros([batch_size, num_channels] + list(shape), device=device)
    
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(batch_size, num_channels, shape, (frequency*res[0], frequency*res[1]),
                                            generator=generator, device=device)
        frequency *= 2
        amplitude *= persistence
    return noise

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    device = 'cpu'
    noise = rand_perlin_2d(batch_size=10, num_channels=3, shape=(256, 256), res=(8, 8), 
                                   generator=None, device=device)
    plt.figure()
    plt.imshow(noise[2, 1], cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig('perlin.png')
    plt.close()
    
    local_generator = torch.Generator(device)
    local_generator.manual_seed(1)

    noise = rand_perlin_2d_octaves(batch_size=10, num_channels=3, shape=(256, 256), 
                                   generator=local_generator, device=device)
    plt.figure()
    plt.imshow(noise[6, 0], cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig('perlino.png')
    plt.close()
