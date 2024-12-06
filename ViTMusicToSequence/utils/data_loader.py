import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Importing the utility functions
from utils.data_utils import process_image, process_label, pad_sequences
from utils.tokenization import Tokenizer  # Assuming tokenizer.py is in utils folder

class MusicDataset(Dataset):
    def __init__(self, corpus_dirpath, corpus_filepath, dictionary_path, target_width, target_height, max_seq_length, val_split=0.0):
        """
        Args:
            corpus_dirpath (str): Directory with all the images and labels.
            corpus_filepath (str): Path to the file with list of sample file names.
            dictionary_path (str): Path to the vocabulary file.
            target_width (int): Target width for resizing the images.
            target_height (int): Target height for resizing the images.
            max_seq_length (int): Maximum sequence length for padding.
            val_split (float): Fraction of data to use for validation.
        """
        self.corpus_dirpath = corpus_dirpath
        self.target_width = target_width
        self.target_height = target_height
        self.max_seq_length = max_seq_length

        # Load corpus list
        with open(corpus_filepath, 'r') as corpus_file:
            corpus_list = corpus_file.read().splitlines()

        # Initialize tokenizer
        self.tokenizer = Tokenizer(dictionary_path)
        self.start_token_id = self.tokenizer.start_token_id
        self.end_token_id = self.tokenizer.end_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Train and validation split
        random.shuffle(corpus_list)
        val_idx = int(len(corpus_list) * val_split)
        self.training_list = corpus_list[val_idx:]
        self.validation_list = corpus_list[:val_idx]

        self.current_idx = 0

        print(f'Training with {len(self.training_list)} samples and validating with {len(self.validation_list)} samples.')

    def next_batch(self, batch_size):
        images = []
        input_sequences = []
        target_sequences = []
        attention_masks = []

        for _ in range(batch_size):
            # Get the file path for the sample
            sample_filepath = self.training_list[self.current_idx]
            sample_fullpath = os.path.join(self.corpus_dirpath, sample_filepath, sample_filepath)

            # IMAGE
            image_path = sample_fullpath + '.png'
            image = Image.open(image_path).convert('L')  # Open the image in grayscale mode
            image = np.array(image)
            image_tensor = process_image(image, self.target_width, self.target_height)
            images.append(image_tensor)

            # GROUND TRUTH
            label_path = sample_fullpath + '.semantic'
            with open(label_path, 'r') as file_obj:
                label_elements = process_label(file_obj)
            label_ids = self.tokenizer.tokenize(' '.join(label_elements))

            # Pad the sequences
            input_seq, target_seq, = pad_sequences(
                label_ids, self.start_token_id, self.end_token_id, self.pad_token_id, self.max_seq_length
            )
            input_sequences.append(torch.tensor(input_seq, dtype=torch.long))
            target_sequences.append(torch.tensor(target_seq, dtype=torch.long))

            # Move to the next sample
            self.current_idx = (self.current_idx + 1) % len(self.training_list)

        # Stack images and sequences to create batch
        batch_images = torch.stack(images)
        batch_input_sequences = torch.stack(input_sequences)
        batch_target_sequences = torch.stack(target_sequences)

        return {
            'images': batch_images,
            'input_sequences': batch_input_sequences,
            'target_sequences': batch_target_sequences,
        }
    
    def get_validation(self):
        images = []
        labels = []

        # Read files
        for sample_filepath in self.validation_list:
            sample_fullpath = os.path.join(self.corpus_dirpath, sample_filepath, sample_filepath)

            # IMAGE
            image_path = sample_fullpath + '.png'
            image = Image.open(image_path).convert('L')  # Open the image in grayscale mode
            image = np.array(image)
            image_tensor = process_image(image, self.target_width, self.target_height)
            images.append(image_tensor)

            # GROUND TRUTH
            label_path = sample_fullpath + '.semantic'
            with open(label_path, 'r') as file_obj:
                label_elements = process_label(file_obj)
            label_ids = self.tokenizer.tokenize(' '.join(label_elements))

            labels.append(label_ids)

        # Transform to batch
        batch_images = torch.stack(images)
        batch_labels = []

        for label_ids in labels:
            _, target_seq, _ = pad_sequences(
                label_ids, self.start_token_id, self.end_token_id, self.pad_token_id, self.max_seq_length
            )
            batch_labels.append(torch.tensor(target_seq, dtype=torch.long))

        batch_labels = torch.stack(batch_labels)

        validation_dict = {
            'images': batch_images,
            'targets': batch_labels,
        }
        
        return validation_dict