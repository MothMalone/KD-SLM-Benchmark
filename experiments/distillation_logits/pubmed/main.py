import torch

# Set paths to the weight files
pth_file_1 = "/home/lexuanan/.cache/huggingface/hub/models--meta-llama--Llama-2-13b/snapshots/5a3ad81c857aaf765c7a229a449490745a9004c9/consolidated.00.pth"
pth_file_2 = "/home/lexuanan/.cache/huggingface/hub/models--meta-llama--Llama-2-13b/snapshots/5a3ad81c857aaf765c7a229a449490745a9004c9/consolidated.01.pth"

# Set the device for loading weights
device = "cuda" if not torch.cuda.is_available() else "cpu"

try:
    # Load the model weights from both .pth files
    weights_1 = torch.load(pth_file_1, map_location=device)
    weights_2 = torch.load(pth_file_2, map_location=device)

    # Combine the weights (assuming they are dictionaries)
    merged_weights = {**weights_1, **weights_2}

    # Set the output path for the merged model
    output_path = "/home/lexuanan/.cache/huggingface/hub/models--meta-llama--Llama-2-13b/snapshots/5a3ad81c857aaf765c7a229a449490745a9004c9/pytorch_model.bin"

    # Save the combined weights as pytorch_model.bin
    torch.save(merged_weights, output_path)

    print(f"Model saved as {output_path}")

except FileNotFoundError as e:
    print(f"Error: One or more weight files not found. Please check the file paths. {e}")
    raise e  # Re-throw the exception for further handling or logging if needed

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise e  # Re-throw the exception
