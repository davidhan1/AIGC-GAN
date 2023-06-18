import torch
import torch.nn as nn
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 1, 28, 28)

noise = torch.randn(100, 100)
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
fake_images = generator(noise)
# print(fake_images.shape)
save_image(fake_images.data[:100], 'images/fake_images.png', nrow=10, normalize=True)