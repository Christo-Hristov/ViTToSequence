import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, BertConfig, get_linear_schedule_with_warmup

# Importing the dataset and utility functions
from utils.data_loader import MusicDataset
from models import TransformerDecoder, ViTWithDecoder
from predict import evaluate_model 

# Configuration parameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
WARMUP_STEPS = 5000
MAX_SEQ_LENGTH = 100
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR =  '/content/drive/MyDrive/cs231nViT/ViTMusicToSequence' #os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
CORPUS_DIRPATH = '/content/drive/MyDrive/cs231nViT/primusCalvoRizoAppliedSciences2018'
CORPUS_FILEPATH = os.path.join(BASE_DIR, 'data/test.txt')
VOCAB_FILEPATH = os.path.join(BASE_DIR, 'data/vocabulary_semantic.txt')

test_dataset = MusicDataset(
        corpus_dirpath=CORPUS_DIRPATH,
        corpus_filepath=CORPUS_FILEPATH,
        dictionary_path=VOCAB_FILEPATH,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        max_seq_length=MAX_SEQ_LENGTH,
        val_split=1.0  # 10% validation split
    )

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the Transformer Decoder
d_encoder = 768
hidden_size = 512  # Ensure the hidden size matches ViT
num_layers = 6
num_heads = 8
ff_hidden_size = 2048  # Dimension of the inner layer in the feed-forward network
pad_idx = test_dataset.pad_token_id
vocab_size = len(test_dataset.tokenizer.token_to_id)
d_model = hidden_size
d_word_vec = hidden_size
d_k = hidden_size // num_heads
d_v = hidden_size // num_heads

    # Initialize the Transformer Decoder
decoder = TransformerDecoder(n_trg_vocab=vocab_size, d_word_vec=d_word_vec,
                             n_layers=num_layers, n_head=num_heads, 
                             d_k=d_k, d_v=d_v, d_encoder=d_encoder, d_model=d_model,
                             d_inner=ff_hidden_size, pad_idx=pad_idx).to(DEVICE)

    # Combine ViT encoder and Transformer decoder
model = ViTWithDecoder(vit_model=vit_model, decoder=decoder, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH).to(DEVICE)

model.load_state_dict(torch.load('best_model.pt'))
val_accuracy = evaluate_model(
                model=model,
                dataset=test_dataset,
                max_seq_length=MAX_SEQ_LENGTH,
                device=DEVICE
            )

print(f'Validation Accuracy: {val_accuracy:.4f}')