import cv2
import numpy as np
import torch
from torchvision import transforms

def process_image(image, target_width, target_height):
    """
    Process a grayscale image for input into a Vision Transformer (ViT) model.
    
    Args:
    - image_path (str): Path to the input grayscale image.
    - target_width (int): Target width for resizing the image.
    - target_height (int): Target height for resizing the image.
    
    Returns:
    - torch.Tensor: Processed image tensor with shape [3, target_height, target_width].
    """
    
    # Resize the image to target dimensions
    resized_image = cv2.resize(image, (target_width, target_height))
    
    # Extend the grayscale image to three channels
    image_3ch = np.stack([resized_image] * 3, axis=-1)
    
    # Normalize the image using mean and std for each channel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply the transformations
    image_tensor = transform(image_3ch)
    
    return image_tensor

def process_label(file_obj):
    """
    Process an opened label file object to extract the labeled sequence.
    
    Args:
    - file_obj: An opened file object containing the label sequence.
    
    Returns:
    - list: List of label elements.
    """
    label_line = file_obj.readline().rstrip()
    label_elements = label_line.split('\t')
    return label_elements

def pad_sequences(sequence, start_token_id, end_token_id, pad_token_id, max_length):
    """
    Prepare input and target sequences for the transformer decoder with padding.
    
    Args:
    - sequence (list): List of token ids representing the target sequence.
    - start_token_id (int): Token id for the <START> token.
    - end_token_id (int): Token id for the <END> token.
    - pad_token_id (int): Token id for the <PAD> token.
    - max_length (int): The maximum length for padding the sequences.
    
    Returns:
    - input_sequence (list): Padded input sequence for the decoder.
    - target_sequence (list): Padded target sequence for calculating loss.
    - attention_mask (list): Attention mask for the input sequence.
    """
    # Prepare input sequence with <START> token and pad to max_length
    input_sequence = [start_token_id] + sequence
    input_sequence = input_sequence[:max_length]
    input_sequence += [pad_token_id] * (max_length - len(input_sequence))
    
    # Prepare target sequence with <END> token and pad to max_length
    target_sequence = sequence + [end_token_id]
    target_sequence = target_sequence[:max_length]
    target_sequence += [pad_token_id] * (max_length - len(target_sequence))
    
    return input_sequence, target_sequence









