import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
batch_size = 64
img_size = 64
channels_img = 3
z_dim = 100
num_epochs = 5
feats_disc = 64
feats_gen = 64
disc_iterations = 5
# weight_clip = 0.01
lambda_gp = 10

trans = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
])

# dataset = datasets.MNIST(root="dataset/", train=True, transform=trans, download=True)
dataset = datasets.ImageFolder(root="./celeba", transform=trans)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, channels_img, feats_gen).to(device)
disc = Discriminator(channels_img, feats_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr = lr, betas = (0.0, 0.9))
# criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        for _ in range(disc_iterations):
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            gp = gradient_penalty(disc, real, fake, device=device)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp*gp
            disc.zero_grad()
            loss_disc.backward(retain_graph = True)
            opt_disc.step()
            # for p in disc.parameters():
                # p.data.clamp_(-weight_clip, weight_clip)

        ###Loss Gen min log(1-D(G(z))) <--> max( log(d(G(z))))
        op = disc(fake).reshape(-1)
        loss_gen = -torch.mean(op)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D {loss_disc: .4f}, Loss G {loss_gen: .4f}")

            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step+=1




