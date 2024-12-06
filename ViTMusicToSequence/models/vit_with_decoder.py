import torch.nn as nn
import math
import torch

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class ViTWithDecoder(nn.Module): # figure out how to finetune on differently sized images, not just 224 x 224
    def __init__(self, vit_model, decoder, target_height, target_width):
        super(ViTWithDecoder, self).__init__()
        self.vit_model = vit_model
        patch_size = 16
        grid_size_height = target_height // patch_size
        grid_size_width = target_width // patch_size
        self.vit_model.embeddings.position_embeddings = nn.Parameter(
            self._resize_pos_embed(
                self.vit_model.embeddings.position_embeddings,
                grid_size_height,
                grid_size_width
            )
        )
        self.decoder = decoder

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        old_grid_size = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb
        
    def forward(self, pixel_values, decoder_input_ids, trg_mask=None, src_mask=None):
        encoder_outputs = self.vit_model(pixel_values).last_hidden_state
        trg_mask = get_subsequent_mask(decoder_input_ids)
        dec_output, *_ = self.decoder(
            trg_seq=decoder_input_ids,
            trg_mask=trg_mask,
            enc_output=encoder_outputs,
            src_mask=src_mask
        )
        return dec_output
