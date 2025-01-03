import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FS2KDataset(Dataset):
    def __init__(self, root_dir, anno_file, transform=None, verify_files=True):
        self.root_dir = root_dir
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        
        # Store valid file paths during initialization
        self.valid_pairs = []
        if verify_files:
            self._verify_and_store_paths()

    def _verify_and_store_paths(self):
        print("Verifying dataset files...")
        for entry in tqdm(self.annotations):
            image_path = entry['image_name']
            photo_dir = os.path.dirname(image_path)
            image_name = os.path.basename(image_path)
            sketch_dir = f"sketch{photo_dir[-1]}"
            sketch_name = image_name.replace('image', 'sketch')
            
            photo_path = None
            sketch_path = None
            
            # Check photo with multiple extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(self.root_dir, 'photo', photo_dir, image_name + ext)
                if os.path.exists(temp_path):
                    photo_path = temp_path
                    break
            
            # Check sketch with multiple extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(self.root_dir, 'sketch', sketch_dir, sketch_name + ext)
                if os.path.exists(temp_path):
                    sketch_path = temp_path
                    break
            
            if photo_path and sketch_path:
                self.valid_pairs.append({
                    'photo': photo_path,
                    'sketch': sketch_path,
                    'annotation': entry
                })
        
        print(f"Found {len(self.valid_pairs)} valid image-sketch pairs out of {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        try:
            photo = Image.open(pair['photo']).convert('RGB')
            sketch = Image.open(pair['sketch']).convert('L')
            sketch = sketch.convert('RGB')
            
            if self.transform:
                photo = self.transform(photo)
                sketch = self.transform(sketch)
            
            return {'sketch': sketch, 'photo': photo}
            
        except Exception as e:
            print(f"Error loading image pair: {pair['photo']} - {pair['sketch']}")
            raise e
        

        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Encoder
            self.conv_block(3, 64, normalize=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            
            # Residual blocks
            *[ResidualBlock(512) for _ in range(6)],
            
            # Decoder
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=256):
        super(Discriminator, self).__init__()
        
        # Calculate output size
        # Starting from 256x256
        # After each strided conv: size = (size - kernel_size + 2*padding) / stride + 1
        
        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            # input is (6, 256, 256)
            conv_block(6, 64, normalize=False),  # (64, 128, 128)
            conv_block(64, 128),                 # (128, 64, 64)
            conv_block(128, 256),                # (256, 32, 32)
            conv_block(256, 512),                # (512, 16, 16)
            nn.Conv2d(512, 1, 4, stride=1, padding=1)  # (1, 15, 15)
        )

    def forward(self, sketch, photo):
        combined = torch.cat([sketch, photo], dim=1)
        return self.model(combined)

def train_step(generator, discriminator, data, criterion_GAN, criterion_pixel, 
               g_optimizer, d_optimizer):
    sketch = data['sketch'].to(device)
    real_photo = data['photo'].to(device)
    batch_size = sketch.size(0)
    
    # Calculate discriminator output size
    disc_patch = 15  # Based on the discriminator architecture calculations
    valid = torch.ones((batch_size, 1, disc_patch, disc_patch), device=device)
    fake = torch.zeros((batch_size, 1, disc_patch, disc_patch), device=device)

    # Train Generator
    g_optimizer.zero_grad()
    gen_photo = generator(sketch)
    pred_fake = discriminator(sketch, gen_photo)
    
    g_loss = criterion_GAN(pred_fake, valid) + \
             100 * criterion_pixel(gen_photo, real_photo)
    
    g_loss.backward()
    g_optimizer.step()

    # Train Discriminator
    d_optimizer.zero_grad()
    pred_real = discriminator(sketch, real_photo)
    pred_fake = discriminator(sketch, gen_photo.detach())
    
    d_loss = (criterion_GAN(pred_real, valid) + 
              criterion_GAN(pred_fake, fake)) / 2
    
    d_loss.backward()
    d_optimizer.step()
    
    return g_loss.item(), d_loss.item()

# Helper function to print model output sizes
def get_output_size(model, input_size=(6, 256, 256)):
    x = torch.randn(1, *input_size)
    out = model(x)
    return out.shape[2:]  # Return spatial dimensions

def visualize_samples(generator, test_dataloader, num_samples=5):
    generator.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        sketches = batch['sketch'].to(device)[:num_samples]
        real_photos = batch['photo'][:num_samples]
        
        gen_photos = generator(sketches).cpu()
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        for i in range(num_samples):
            # Display sketch
            axes[i, 0].imshow(sketches[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            axes[i, 0].set_title('Sketch')
            axes[i, 0].axis('off')
            
            # Display generated photo
            axes[i, 1].imshow(gen_photos[i].permute(1, 2, 0) * 0.5 + 0.5)
            axes[i, 1].set_title('Generated')
            axes[i, 1].axis('off')
            
            # Display real photo
            axes[i, 2].imshow(real_photos[i].permute(1, 2, 0) * 0.5 + 0.5)
            axes[i, 2].set_title('Real')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    generator.train()

if __name__ == "__main__":
    # Paths - update these to match your setup
    root_dir = r"C:\Users\srees\OneDrive\Documents\projects\research project\FS2K\FS2K"  # Update this
    train_anno = os.path.join(root_dir, "anno_train.json")
    test_anno = os.path.join(root_dir, "anno_test.json")
    
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    lr = 0.0002
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Datasets and dataloaders
    train_dataset = FS2KDataset(root_dir, train_anno, transform)
    test_dataset = FS2KDataset(root_dir, test_anno, transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4)
    
    start_epoch = 20  # Change this to the epoch number you want to start from
    generator_checkpoint = 'generator_epoch_20.pth'  # Change to your saved generator file
    discriminator_checkpoint = 'discriminator_epoch_20.pth'  # Change to your saved discriminator file

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Load saved states if they exist
    if os.path.exists(generator_checkpoint) and os.path.exists(discriminator_checkpoint):
        print(f"Loading checkpoint from epoch {start_epoch}")
        generator.load_state_dict(torch.load(generator_checkpoint))
        discriminator.load_state_dict(torch.load(discriminator_checkpoint))
    else:
        print("No checkpoints found, starting from scratch")
        start_epoch = 0

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Add this before training loop
    print("Verifying model architectures...")
    test_sketch = torch.randn(1, 3, 256, 256).to(device)
    test_photo = torch.randn(1, 3, 256, 256).to(device)

    # Test generator
    gen_output = generator(test_sketch)
    print(f"Generator output size: {gen_output.shape}")

    # Test discriminator
    disc_output = discriminator(test_sketch, test_photo)
    print(f"Discriminator output size: {disc_output.shape}")

    assert disc_output.shape[2] == disc_output.shape[3] == 15, \
        f"Unexpected discriminator output size: {disc_output.shape}"


    # Training loop
    for epoch in range(start_epoch, num_epochs):  # Modified to start from start_epoch
        g_losses = []
        d_losses = []
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            g_loss, d_loss = train_step(generator, discriminator, batch,
                                    criterion_GAN, criterion_pixel,
                                    g_optimizer, d_optimizer)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"G Loss: {sum(g_losses)/len(g_losses):.4f}")
        print(f"D Loss: {sum(d_losses)/len(d_losses):.4f}")
        
        if (epoch + 1) % 5 == 0:
            visualize_samples(generator, test_dataloader)
            
        # Save models
        if (epoch + 1) % 10 == 0:
            save_path = 'models'  # Create a dedicated directory for model checkpoints
            os.makedirs(save_path, exist_ok=True)
            torch.save(generator.state_dict(), 
                    os.path.join(save_path, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), 
                    os.path.join(save_path, f'discriminator_epoch_{epoch+1}.pth'))