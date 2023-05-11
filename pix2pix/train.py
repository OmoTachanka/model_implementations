import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from gen import Generator
from disc import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(disc, gen, loader, opt_disc, opt_gen, bce, L1, d_scaler, g_scaler):
    loop = tqdm(loader, leave = True)
    for idx, (x, y)in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = disc(x, y)
            d_fake = disc(x, y_fake.detach())
            d_real_loss = bce(d_real, torch.ones_like(d_real))
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss)/2
        
        disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            d_fake = disc(x,y_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1 = L1(y_fake, y)*config.L1_LAMBDA
            g_loss = g_fake_loss + l1
        
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LR, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LR, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHCKPT_GEN, gen, opt_gen, config.LR)
        load_checkpoint(config.CHCKPT_DISC, disc, opt_disc, config.LR)

    train_dataset = MapDataset(root="./maps/train/")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root="./maps/val/")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc, opt_gen, BCE, L1_LOSS, d_scaler, g_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc, opt_disc, filename=f"./chkpt/disc_epoch_{epoch}.pth.tar")
            save_checkpoint(gen, opt_gen, filename=f"./chkpt/gen_epoch_{epoch}.pth.tar")

        save_some_examples(gen, val_loader, epoch, folder="eval")

if __name__ == "__main__":
    main()