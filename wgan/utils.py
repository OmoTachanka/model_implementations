import torch

def gradient_penalty(disc, real, fake, device = "cpu"):
    batch_size, channels, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, H, W).to(device)
    interpolated = real * epsilon  + fake * (1 - epsilon)
    

    mixed_scores = disc(interpolated)

    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim = 1)
    gradient_penalty = torch.mean((grad_norm - 1)**2)
    return gradient_penalty