import torch

def generate_sequence(model, pixel_values, start_token_id, end_token_id, pad_token_id, max_seq_length, device):
    """
    Generate sequences from the model given pixel values of images.

    Args:
    - model: The combined ViT encoder and Transformer decoder model.
    - pixel_values: Tensor of pixel values of the images.
    - start_token_id: Token ID for the <START> token.
    - end_token_id: Token ID for the <END> token.
    - pad_token_id: Token ID for the <PAD> token.
    - max_seq_length: Maximum sequence length for generation.
    - device: Device to run the model on (CPU or CUDA).

    Returns:
    - List of lists containing generated token IDs for each image.
    """
    model.eval()
    batch_size = pixel_values.size(0)
    generated_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long).to(device)
    input_ids = generated_ids

    with torch.no_grad():
        for _ in range(max_seq_length):
            outputs = model(pixel_values, input_ids)
            next_token_logits = outputs[:, -1, :]  # Get the logits for the last generated token
            next_token_ids = next_token_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

            if (next_token_ids == end_token_id).all():
                break

    generated_ids = input_ids.cpu().numpy().tolist()
    generated_ids = [[id for id in seq if id != pad_token_id] for seq in generated_ids]
    return generated_ids

def evaluate_model(model, dataset, max_seq_length, device):
    """
    Evaluate the model on the given dataset.

    Args:
    - model: The combined ViT encoder and Transformer decoder model.
    - dataset: The dataset to evaluate on.
    - max_seq_length: Maximum sequence length for generation.
    - device: Device to run the model on (CPU or CUDA).

    Returns:
    - Validation accuracy.
    """
    correct_predictions = 0
    total_predictions = 0

    validation_dict = dataset.get_validation()

    with torch.no_grad():
        pixel_values = validation_dict['images'].to(device)
        decoder_target_ids = validation_dict['targets'].to(device)

        batch_size = pixel_values.size(0)
        generated_ids = generate_sequence(
            model=model,
            pixel_values=pixel_values,
            start_token_id=dataset.start_token_id,
            end_token_id=dataset.end_token_id,
            pad_token_id=dataset.pad_token_id,
            max_seq_length=max_seq_length,
            device=device
        )

        target_ids = [[id for id in target_seq if id != dataset.pad_token_id] for target_seq in decoder_target_ids.cpu().numpy()]

        for gen_ids, tgt_ids in zip(generated_ids, target_ids):
            correct_predictions += int(gen_ids == tgt_ids)
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy
