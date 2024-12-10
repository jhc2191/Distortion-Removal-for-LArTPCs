import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from networks import CorrectionNet, DistortionNet
import os
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torch.autograd as autograd
from PIL import Image
import torch.nn.utils as utils
import argparse
from pytorch_msssim import ssim

# Command-line arguments
parser = argparse.ArgumentParser(description="Train and test a distortion-correcting model.")
parser.add_argument('--train_dir', type=str, required=True, help="Path to the training data directory")
parser.add_argument('--test_dir', type=str, required=True, help="Path to the testing data directory")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimization")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # For 3 channels
])

class DistortedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get list of all images
        self.image_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def save_corrected_images(corrected_images, batch_idx, epoch=None, phase='train', dist=False):
    corrected_images = corrected_images.cpu()
    to_pil = ToPILImage()

    # Handle the case where epoch is None (during testing)
    if epoch is not None:
        save_dir = os.path.join('corrected_images', phase, f'epoch_{epoch + 1}', f'batch_{batch_idx + 1}')
    else:
        save_dir = os.path.join('corrected_images', phase)

    os.makedirs(save_dir, exist_ok=True)

    for b in range(corrected_images.size(0)):
        img_tensor = corrected_images[b]

        # Denormalize the image from [-1, 1] to [0, 1]
        img_tensor = img_tensor * 0.5 + 0.5

        img = to_pil(img_tensor.clamp(0, 1))  # Ensure values are within [0, 1]
        n = batch_idx*4+b
        if dist==False:
            img_filename = f'image_{n}.png'
        else:
            img_filename = f'dist_image_{n}.png'
        img.save(os.path.join(save_dir, img_filename))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, data_range=2, size_average=self.size_average)


class LineThicknessLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(LineThicknessLoss, self).__init__()
        self.threshold = threshold

    def forward(self, img):
        # img shape: (B, C, H, W)
        # Convert to grayscale
        img_gray = img.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Binarize using the threshold
        binary_mask = (img_gray > self.threshold).float()

        # Compute the fraction of pixels above the threshold
        # This will serve as a penalty. The more above-threshold pixels, the higher the loss.
        loss = binary_mask.mean()
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Define Sobel kernels for edge detection
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        # Register the kernels as buffers so they're moved with the model's device
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, img1, img2):
        # img1 and img2: (B, C, H, W)
        # Rescale images from [-1, 1] to [0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # Clip to prevent extreme values
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)

        # Convert images to grayscale
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        
        # Apply Sobel filters to detect edges
        grad_x1 = F.conv2d(img1_gray, self.sobel_x, padding=1)
        grad_y1 = F.conv2d(img1_gray, self.sobel_y, padding=1)
        edges1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2)
        
        grad_x2 = F.conv2d(img2_gray, self.sobel_x, padding=1)
        grad_y2 = F.conv2d(img2_gray, self.sobel_y, padding=1)
        edges2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2)
        
        # Compute L1 loss between the edge maps
        loss = F.l1_loss(edges1, edges2)
        return loss



class CurvatureLoss(nn.Module):
    def __init__(self):
        super(CurvatureLoss, self).__init__()
        # Define kernels for second derivatives
        self.kernel_xx = torch.tensor([[1, -2, 1]], dtype=torch.float32).reshape(1, 1, 1, 3) / 4.0
        self.kernel_yy = torch.tensor([[1], [-2], [1]], dtype=torch.float32).reshape(1, 1, 3, 1) / 4.0

    def forward(self, img):
        # img shape: (B, C, H, W)
        img_gray = img.mean(dim=1, keepdim=True)  # Convert to grayscale

        # Compute second derivatives
        d2x = nn.functional.conv2d(img_gray, self.kernel_xx.to(img.device), padding=(0,1))
        d2y = nn.functional.conv2d(img_gray, self.kernel_yy.to(img.device), padding=(1,0))

        # Compute curvature
        curvature = torch.abs(d2x) + torch.abs(d2y)
        loss = curvature.mean()
        return loss

class TotalVariationLoss(nn.Module):
    def forward(self, img):
        loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
               torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        return loss

def weights_init_normal(m):
    """
    Initialize weights for Conv2d and BatchNorm2d layers.

    Args:
        m (nn.Module): The module to initialize.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


# Initialize networks
correction_net = CorrectionNet(
    input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8,
    norm='in', activ='relu', pad_type='reflect', final_activ='tanh'
).to(device)

distortion_net = DistortionNet(
    input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8,
    norm='in', activ='relu', pad_type='reflect', final_activ='tanh'
).to(device)

# Apply weights initialization
correction_net.apply(weights_init_normal)
distortion_net.apply(weights_init_normal)

# Initialize optimizers
optimizer_G = optim.Adam(correction_net.parameters(), lr=args.lr)
optimizer_F = optim.Adam(distortion_net.parameters(), lr=args.lr)

from torch.cuda.amp import GradScaler
from torch.amp import autocast

# Initialize GradScaler
scaler = GradScaler()

# Initialize loss functions
criterion_cycle = nn.L1Loss()
criterion_curvature = CurvatureLoss().to(device)
criterion_tv = TotalVariationLoss().to(device)
criterion_thickness = LineThicknessLoss().to(device)
criterion_edge = EdgeLoss().to(device)  # New Edge Loss
criterion_ssim = SSIMLoss().to(device)



def compute_loss(cycle_loss, curvature_loss, tv_loss, thickness_loss, weights):
    total_loss = (weights['cycle'] * cycle_loss +
                  weights['curvature'] * curvature_loss +
                  weights['tv'] * tv_loss +
                  weights['thickness'] * thickness_loss)
    return total_loss



from torch.cuda.amp import GradScaler
from torch.amp import autocast


def train(epochs, batch_size):
    train_dataset = DistortedDataset(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    correction_net.train()
    distortion_net.train()

    # Loss weights
    weights = {
        'cycle': 3.0,
        'curvature': 3.0,
        'tv': 0.1,  # Adjust this weight as needed
        'thickness': 5.0  # Adjust this as needed
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, images in enumerate(train_loader):
            images = images.to(device)
            optimizer_G.zero_grad()
            optimizer_F.zero_grad()

            with autocast(device_type='cuda'):
                # Forward pass
                corrected_images = correction_net(images)
                reconstructed_images = distortion_net(corrected_images)

                # Compute losses
                cycle_loss = criterion_cycle(reconstructed_images, images)
                curvature_loss = criterion_curvature(corrected_images)
                tv_loss = criterion_tv(corrected_images)
                thickness_loss = criterion_thickness(corrected_images)
                #edge_loss = criterion_edge(corrected_images, images)
                #ssim_loss = criterion_ssim(corrected_images, images)

                # Compute total loss
                total_loss = compute_loss(cycle_loss, curvature_loss, tv_loss, thickness_loss, weights)

            # Backward pass and optimization
            scaler.scale(total_loss).backward()

            # Unscale gradients and clip
            scaler.unscale_(optimizer_G)
            scaler.unscale_(optimizer_F)
            utils.clip_grad_norm_(correction_net.parameters(), max_norm=1.0)
            utils.clip_grad_norm_(distortion_net.parameters(), max_norm=1.0)

            scaler.step(optimizer_G)
            scaler.step(optimizer_F)
            scaler.update()

            if (i + 1) % 10 == 0:
                print(f"Batch [{i + 1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}, "
                      f"Cycle Loss: {cycle_loss.item():.4f}, Curvature Loss: {curvature_loss.item():.4f}, "
                      f"TV Loss: {tv_loss.item():.4f}, Thickness Loss: {thickness_loss.item():.4f} ")
                      
            # Save intermediate outputs every N batches
            if (i + 1) % 500 == 0:
                save_corrected_images(corrected_images, i, epoch, phase='train', dist=False)
                save_corrected_images(reconstructed_images, i, epoch, phase='train', dist=True)

            # Clean up
            del images, corrected_images, reconstructed_images
            torch.cuda.empty_cache()


def test(batch_size):
    test_dataset = DistortedDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correction_net.eval()
    to_pil = ToPILImage()

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)  # images shape: (B, C, H, W)

            # Forward pass through CorrectionNet
            corrected_images = correction_net(images)  # Output shape: (B, C, H, W)

            # Save corrected images without needing to specify epoch
            save_corrected_images(corrected_images, batch_idx, phase='test')

            print(f"Processed and saved batch {batch_idx + 1} of size {images.size(0)}")

            # Clean up
            del images, corrected_images
            torch.cuda.empty_cache()


if __name__ == '__main__':
    print("Starting training...")
    train(args.epochs, args.batch_size)
    print("Training complete. Starting testing...")
    test(args.batch_size)
    print("Testing complete.")
