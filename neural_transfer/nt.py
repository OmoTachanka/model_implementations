import torch
import timm
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

vgg19 = models.vgg19(weights = models.VGG19_Weights.DEFAULT).features

class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = model


    def forward(self, x):
        feats = []

        for lay_num, layer in enumerate(self.model):
            x = layer(x)

            if str(lay_num) in self.chosen_features:
                feats.append(x)

        return feats

def load_image(img_name):
    img = Image.open(img_name).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 224
loader = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()
 ])

og = load_image("./sup nura.png")
# sty = load_image("./sergey-kolesov-delilah-throne-painting-color.jpg")
sty = load_image("./piotr-jablonski-portrait-of-the-eyeless-leaders-s.jpg")

generated = og.clone().requires_grad_(True)

total_steps = 6000
lr = 1e-3
a = 1
b = 0.01
optimizer = optim.Adam([generated], lr=lr)

model = VGG(vgg19).to(device)

for step in range(total_steps):
    gen_feats = model(generated)
    og_feats = model(og)
    sty_feats = model(sty)

    sty_loss = og_loss = 0

    for gen_feat, og_feat, sty_feat in zip(gen_feats, og_feats, sty_feats):
        batch_size, channel, height, width = gen_feat.shape
        og_loss += torch.mean((gen_feat - og_feat)**2)
        G = gen_feat.view(channel, height*width).mm(gen_feat.view(channel, height*width).t())

        A = sty_feat.view(channel, height*width).mm(sty_feat.view(channel, height*width).t())

        sty_loss += torch.mean((G - A)**2)

    total_loss = a*og_loss + b*sty_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200  == 0:
        print("Loss at {}: {}".format(step, total_loss))
        save_image(generated, "./op/generated_{}.png".format(step))

