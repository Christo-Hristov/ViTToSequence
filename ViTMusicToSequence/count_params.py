from transformers import ViTModel, BertModel, BertConfig

# Load ViT-B/16 model
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the Transformer Decoder with the given configuration
hidden_size = 768  # hidden size of ViT-B/16
vocab_size = 30522  # Assuming using the same vocab size as BERT
num_layers = 6
num_heads = 8

config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    num_hidden_layers=num_layers,
    intermediate_size=hidden_size * 4,  # Setting intermediate FFN layer size
    is_decoder=True,  # This indicates that the model is a decoder
    add_cross_attention=True  # This adds cross-attention layers for encoder-decoder attention
)

# Load the BERT model with the specified config
decoder = BertModel(config)

# Calculate the total number of parameters
vit_params = sum(p.numel() for p in vit_model.parameters())
decoder_params = sum(p.numel() for p in decoder.parameters())
total_params = vit_params + decoder_params

print(f"Total number of parameters in the model: {total_params}")