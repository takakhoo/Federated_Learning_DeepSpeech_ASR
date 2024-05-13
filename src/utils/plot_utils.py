# Install necessary libraries if you haven't already
# !pip install matplotlib numpy librosa

import numpy as np
# import matplotlib.pyplot as plt
import librosa
import librosa.display

# plot utilities
import matplotlib.pyplot as plt
import torch
import seaborn as sns
def plot_loss_over_epoch(loss_history,title,x_title,y_title):
    """
    Plot loss values over epochs.
    
    Args:
        loss_history (list): List of loss values for each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, linestyle='-')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.show()

def plot_spectrogram(tensor):
    """
    Plot a spectrogram from a PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    """
    # Ensure that the input tensor is on the CPU and in the numpy format
    spectrogram = tensor.squeeze().cpu().numpy()
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='viridis', origin='lower', aspect='auto')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show();

def plot_four_graphs(gt_tensor, reconstructed_tensor, loss, loss_grad, loss_reg,epoch=0):
    """
    Plot four graphs: ground truth spectrogram, reconstructed spectrogram, 
    difference between the two, and loss over epoch.
    
    Args:
        gt_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, channels, height, width).
        reconstructed_tensor (torch.Tensor): Reconstructed tensor of same shape as `gt_tensor`.
        loss_array (List[float]): Array of loss values over epochs.
    """
    diff_tensor = torch.abs(gt_tensor - reconstructed_tensor)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ground Truth Spectrogram
    axs[0, 0].imshow(gt_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[0, 0].set_title('Ground Truth Spectrogram')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Frequency')

    # Reconstructed Spectrogram
    axs[0, 1].imshow(reconstructed_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[0, 1].set_title('Reconstructed Spectrogram')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Frequency')
    
    # Difference Spectrogram
    axs[1, 0].imshow(diff_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[1, 0].set_title('Difference Spectrogram')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency')

    # Loss over epoch
    axs[1, 1].plot(loss,label='loss')
    axs[1, 1].plot(loss_grad,label='loss gm')
    axs[1, 1].plot(loss_reg,label='loss reg')
    axs[1, 1].set_title('Loss Over Epochs')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    
    plt.tight_layout()
    # save figure with name that has date time hour min and epoch number
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")
    plt.savefig('figures/{}_{}.png'.format(now, epoch))
    plt.show()

    
def plot_value_distribution_input_reconstructed(tensor_input, tensor_reconstructed):
    """
    Plot the value distributions of the input and its reconstructed version.
    
    Args:
        tensor_input (torch.Tensor): The input tensor.
        tensor_reconstructed (torch.Tensor): The reconstructed version of the input tensor.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(tensor_input.cpu().numpy().flatten(), color='blue', label='Input', kde=True, bins=50, alpha=0.5)
    sns.histplot(tensor_reconstructed.cpu().numpy().flatten(), color='red', label='Input Reconstructed', kde=True, bins=50, alpha=0.5)
    plt.legend()
    plt.title('Value Distribution of Input and Input Reconstructed')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot waveform, matplotlib 3.7
def plot_waveform(signal, sr, title="Waveform"):
    plt.figure(figsize=(15, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_tensor_as_image(tensor):
    # Scale tensor to the range [0, 1]
    scaled_tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

    # Convert tensor to a NumPy array
    image_array = scaled_tensor.squeeze().numpy()

    # Plot the array as an image
    plt.imshow(image_array, cmap='gray')  # Assuming a grayscale image
    plt.axis('off')
    plt.show()
