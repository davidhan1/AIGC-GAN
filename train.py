import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Define the generator and discriminator models
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

# Define the dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = dset.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the loss function, optimizer, and noise vector
criterion = nn.BCELoss()
generator = Generator()
discriminator = Discriminator()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
noise = torch.randn(128, 100)

# Configure TensorBoard
writer = SummaryWriter()

# Train the generator and discriminator models
num_epochs = 500
for epoch in tqdm(range(num_epochs)):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)

        # Train discriminator on real images
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        d_loss_real.backward()

        # Train discriminator on fake images
        fake_labels = torch.zeros(batch_size, 1)
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        # Train generator
        generator.zero_grad()
        fake_labels = torch.ones(batch_size, 1)
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, fake_labels)
        g_loss.backward()
        optimizer_g.step()

        # Print training progress and write to TensorBoard
        if i == len(dataloader)-1:
            print('[%d/%d] Discriminator Loss: %.4f, Generator Loss: %.4f'
                  % (epoch+1, num_epochs, (d_loss_real+d_loss_fake).item(), g_loss.item()))
            writer.add_scalar("Loss/Discriminator", (d_loss_real+d_loss_fake).item(), epoch)
            writer.add_scalar("Loss/Generator", g_loss.item(), epoch)

    # Save generated images
    fake_images = generator(noise)
    # save_image(fake_images.data[:25], 'images/fake_images_%03d.png' % (epoch+1), nrow=5, normalize=True)
    writer.add_images("Images/Fake Images", fake_images[:25], epoch)

# Save trained generator model
torch.save(generator.state_dict(), 'generator.pth')

# Close TensorBoard writer
writer.close()