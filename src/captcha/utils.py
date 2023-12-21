def decode_preds(outputs, decoder):
    # outputs -> (full_data, seq_len, num_classes)
    # We need to find the argmax for each timestep
    # and convert them to characters
    outputs = outputs.argmax(dim=-1)  # (full_data, seq_len)
    outputs = outputs.detach().cpu().numpy()
    labels = [decoder(output) for output in outputs]
    return labels # ["".join(label) for label in labels]
