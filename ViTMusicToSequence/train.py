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

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
CORPUS_DIRPATH = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir, os.pardir, os.pardir, 'Downloads', 'primus(1)'))
CORPUS_FILEPATH = os.path.join(BASE_DIR, 'data/train.txt')
VOCAB_FILEPATH = os.path.join(BASE_DIR, 'data/vocabulary_semantic.txt')

def train():
    # Initialize dataset
    train_dataset = MusicDataset(
        corpus_dirpath=CORPUS_DIRPATH,
        corpus_filepath=CORPUS_FILEPATH,
        dictionary_path=VOCAB_FILEPATH,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        max_seq_length=MAX_SEQ_LENGTH,
        val_split=0.1  # 10% validation split
    )

    # Load pre-trained ViT model
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Initialize the Transformer Decoder
    d_encoder = 768
    hidden_size = 512  # Ensure the hidden size matches ViT
    num_layers = 6
    num_heads = 8
    ff_hidden_size = 2048  # Dimension of the inner layer in the feed-forward network
    pad_idx = train_dataset.pad_token_id
    vocab_size = len(train_dataset.tokenizer.token_to_id)
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

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataset.training_list) // BATCH_SIZE * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token_id)
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for _ in range(len(train_dataset.training_list) // BATCH_SIZE):
            print('Batch training commencing')
            batch = train_dataset.next_batch(BATCH_SIZE)
            pixel_values = batch['images'].to(DEVICE)
            decoder_input_ids = batch['input_sequences'].to(DEVICE)
            decoder_target_ids = batch['target_sequences'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pixel_values, decoder_input_ids)
            print('Model outputs gained')
            loss = criterion(outputs.view(-1, vocab_size), decoder_target_ids.view(-1))
            print('loss calculated')
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / (len(train_dataset.training_list) // BATCH_SIZE)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}')

        # Validation
        if (epoch + 1) % 5 == 0:
            val_accuracy = evaluate_model(
                model=model,
                dataset=train_dataset,
                max_seq_length=MAX_SEQ_LENGTH,
                device=DEVICE
            )
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            # Save the model with the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                checkpoint_path = f'best_model.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Saved best model checkpoint to {checkpoint_path}')

if __name__ == "__main__":
    train()
