import torch
import matplotlib.pyplot as plt

class StabilizationChecker:
    def __init__(self, num_epochs_to_track, tolerance=1e-5):
        self.num_epochs_to_track = num_epochs_to_track
        self.tolerance = tolerance
        self.generator_losses = []
        self.discriminator_losses = []
    
    def update_losses(self, generator_loss, discriminator_loss):
        self.generator_losses.append(generator_loss)
        self.discriminator_losses.append(discriminator_loss)
        # Only keep the most recent losses for tracking
        self.generator_losses = self.generator_losses[-self.num_epochs_to_track:]
        self.discriminator_losses = self.discriminator_losses[-self.num_epochs_to_track:]
    
    def has_stabilized(self):
        # Check if we have enough data to determine stabilization
        if len(self.generator_losses) < self.num_epochs_to_track or len(self.discriminator_losses) < self.num_epochs_to_track:
            return False  # Not enough data to conclude stabilization
        
        # Check if the absolute change in loss is below the tolerance for both generator and discriminator
        gen_stable = all(abs(self.generator_losses[i] - self.generator_losses[i - 1]) < self.tolerance for i in range(1, len(self.generator_losses)))
        disc_stable = all(abs(self.discriminator_losses[i] - self.discriminator_losses[i - 1]) < self.tolerance for i in range(1, len(self.discriminator_losses)))
        
        return gen_stable and disc_stable

def imshow(tensor, title=''):
    """Imshow for Tensor."""
    # Unnormalize the image
    tensor = tensor / 2 + 0.5  # assuming the tensor was normalized by mean=0.5 and std=0.5
    tensor = tensor.to('cpu').detach().numpy()
    # Convert from CxHxW to HxWxC format for plotting
    plt.imshow(np.transpose(tensor, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')



