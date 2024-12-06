import os

class Tokenizer:
    def __init__(self, vocab_file):
        self.token_to_id, self.id_to_token = self.load_vocab(vocab_file)
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Get the token IDs for special tokens
        self.pad_token_id = self.token_to_id.get(self.pad_token, None)
        self.start_token_id = self.token_to_id.get(self.start_token, None)
        self.end_token_id = self.token_to_id.get(self.end_token, None)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as file:
            tokens = file.read().splitlines()
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return token_to_id, id_to_token

    def tokenize(self, text):
        tokens = text.split()
        token_ids = [self.token_to_id.get(token, self.token_to_id.get('[UNK]')) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        tokens = [self.id_to_token.get(token_id, '[UNK]') for token_id in token_ids]
        return ' '.join(tokens)

