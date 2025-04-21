import numpy as np
import torch
import torch.nn as nn
from visualize_modified import visualize
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, List


import torch
import numpy as np

# Assume 'model', 'test_dataLoader', and 'device' are defined elsewhere
# e.g., device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# model = YourPyTorchModel(...)
# test_dataLoader = torch.utils.data.DataLoader(...)


def get_data_from_last_time_step(padSeq, original_seq_lens, device):
    # Gather the outputs at the actual sequence lengths
    last_seq_idxs = (original_seq_lens - 1).to(device) # Move indices to correct device
    # Create batch indices [0, 1, ..., batch_size-1]
    batch_indices = torch.arange(0, padSeq.size(0)).to(device)
    # Index the target and prediction using batch_indices and last_seq_idxs
    last_step = padSeq[batch_indices, last_seq_idxs, :]
    
    return last_step

def get_mse_last_time_step(y_padSeq, y_pred, original_seq_lens, device):
    batch_size = y_padSeq.size(0)

    y_val_last_step = get_data_from_last_time_step(y_padSeq, original_seq_lens, device)
    pred_last_step = get_data_from_last_time_step(y_pred, original_seq_lens, device)

    sq_error = torch.pow(pred_last_step - y_val_last_step, 2)
    batch_total_loss = torch.sum(sq_error)
    
    return batch_total_loss, batch_size, pred_last_step # batch_size is equivalent num_valid_in_batch 
    

def get_mse_entire_sequence(y_padSeq, y_pred, original_seq_lens, loss_fn, device):
    # --- Masked Loss Calculation ---
    loss_elementwise = loss_fn(y_pred, y_padSeq) # Needs reduction='none'
    # 1. Create the mask (True/1 for non-padded, False/0 for padded) on CPU
    max_len = y_pred.size(1) # Get max sequence length from output
    indices_cpu = torch.arange(max_len) # Shape: [max_len] on CPU
    # Expand indices to batch shape and compare with lengths (on CPU)
    mask_cpu = indices_cpu.expand(y_padSeq.size(0), -1) < original_seq_lens.unsqueeze(1) # Shape: [batch, max_len] (boolean) on CPU
    mask = mask_cpu.unsqueeze(-1).to(device) # Shape: [batch, max_len, 1] (boolean) on Device

    # 2. Apply mask and calculate sum of loss over valid elements
    # Ensure mask dtype is compatible with multiplication (convert boolean to float/int)
    masked_loss = loss_elementwise * mask.float() # Or mask.int()
    batch_total_loss = torch.sum(masked_loss)
    num_valid_in_batch = torch.sum(mask).item()
    
    pred_last_step = get_data_from_last_time_step(y_pred, original_seq_lens, device)
    
    return batch_total_loss, num_valid_in_batch, pred_last_step


def evaluate_data(model, dataLoader, loss_fn, device=torch.device('cpu'), compare_entire_seq=True):
    model.eval()
    total_sq_error_sum = 0.0
    total_samples = 0 # Count samples based on test_dataLoader size
    predictions_list = [] # Store predictions (on CPU)

    with torch.no_grad():
        # Use validation set for evaluating performance
        for i, (x_padSeq, original_seq_lens, y_padSeq) in enumerate(dataLoader):
            # Move validation data to device
            x_padSeq = x_padSeq.to(device)
            y_padSeq = y_padSeq.to(device)

            # Validation sequences can be variable, pass lengths from test_dataLoader
            y_pred = model(x_padSeq, original_seq_lens.cpu()) # Assuming model handles lengths=None if needed
            
            if compare_entire_seq:
                batch_total_loss, num_valid_in_batch, pred_last_step = get_mse_entire_sequence(y_padSeq=y_padSeq,
                                                                                                y_pred=y_pred,
                                                                                                original_seq_lens=original_seq_lens,
                                                                                                loss_fn=loss_fn,
                                                                                                device=device)
            else:
                batch_total_loss, num_valid_in_batch, pred_last_step = get_mse_last_time_step(y_padSeq=y_padSeq,
                                                                                                y_pred=y_pred,
                                                                                                original_seq_lens=original_seq_lens,
                                                                                                device=device)
            # Store predictions (move to CPU) for potential visualization
            predictions_list.append(pred_last_step.cpu().numpy())
            
            # Avoid division by zero if a batch has no valid elements (e.g., all padding)
            if num_valid_in_batch > 0:
                total_sq_error_sum += batch_total_loss.item()
                total_samples += num_valid_in_batch
            else:
                # Handle case where batch has no valid elements (skip backward/step)
                print(f"Warning: Batch {i+1} has no valid elements based on lengths.")
                pass # No loss contribution, no backward pass

    # Calculate validation RMSE after iterating through all batches
    rmse = np.sqrt(total_sq_error_sum / total_samples) if total_samples > 0 else 0
    return rmse, predictions_list
        


def train_model(
    model: nn.Module,
    train_dataLoader: torch.utils.data.DataLoader,
    test_dataLoader: torch.utils.data.DataLoader,
    loss_fn: nn.Module, # IMPORTANT: Expects loss_fn with reduction='none'
    optimizer: torch.optim.Optimizer,
    device: torch.device, # Pass the device object
    writer: Optional[SummaryWriter] = None,
    num_epochs: int = 100,
    y_test: Optional[np.ndarray] = None, # Assuming y_test is for visualization comparison
    start_epoch: int = 1,
    patience: int = 10, # Number of epochs to wait for improvement before stopping
    best_model_path: str = 'best_model.pth', # Path to save the best model
    compare_entire_seq_in_val: bool = False # Employs different RMSE calculations based on this 
    ) -> Tuple[float, float, float, int]:
    """
    Trains and validates a PyTorch model for sequence prediction (e.g., RUL).

    Args:
        model: The PyTorch model to train (should already be on 'device').
        train_dataLoader: DataLoader for training data. Yields (padded_sequences, lengths, padded_targets).
        test_dataLoader: DataLoader for validation data. Yields (sequences, targets).
                         Assumes fixed length or model handles variable length internally for validation.
        loss_fn: Loss function instance (MUST have reduction='none').
        optimizer: Optimizer instance.
        device: The torch.device (cuda, mps, cpu) to run training on.
        writer: Optional TensorBoard SummaryWriter instance.
        num_epochs: Total number of epochs to train for.
        y_test: Optional ground truth for visualization during validation.
        start_epoch: The starting epoch number (e.g., for resuming training).
        patience: Number of epochs with no validation loss improvement to trigger early stopping.
        best_model_path: File path to save the best performing model state_dict.

    Returns:
        Tuple containing:
            - Final training loss (average per sample).
            - Best validation loss achieved
            - Best validation rmse achieved (RMSE on last time step).
            - The epoch number when training finished.
    """
    print(f"Using device: {device}")
    # --- Ensure loss function has reduction='none' ---
    try:
        if hasattr(loss_fn, 'reduction') and loss_fn.reduction != 'none':
            print(f"Warning: loss_fn.reduction is '{loss_fn.reduction}'. "
                  "For masked loss, it should be 'none'. Attempting to proceed, "
                  "but results might be incorrect if loss isn't element-wise.")
    except Exception as e:
        print(f"Could not check loss_fn.reduction: {e}. Assuming it's 'none'.")


    best_val_loss = float('inf')
    epochs_no_improve = 0
    current_batch_size = train_dataLoader.batch_size # Get batch size for printing

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_train_loss_sum = 0  # Sum of losses over valid elements in epoch
        total_valid_elements = 0  # Total number of valid (non-padded) elements

        # Optional: Use tqdm for a cleaner progress bar
        # train_iterator = tqdm(train_dataLoader, desc=f"Epoch {epoch: 3}/{num_epochs + start_epoch - 1} Training", leave=True)
        # for i, (x_padSeq, original_seq_lens, y_padSeq) in enumerate(train_iterator):

        print(f"Epoch {epoch: 3}/{num_epochs + start_epoch - 1} | Training...   ", end="")
        for i, (x_padSeq, original_seq_lens, y_padSeq) in enumerate(train_dataLoader):
            # Simple progress indicator
            print(f"\b\b{(i+1)*current_batch_size:02}", end='')

            # --- Move data to device ---
            # NOTE: original_seq_lens stays on CPU for masking/packing
            x_padSeq = x_padSeq.to(device)
            y_padSeq = y_padSeq.to(device)
            # original_seq_lens = original_seq_lens.to(device) # Keep on CPU

            optimizer.zero_grad()
            # --- Forward pass ---
            # Assumes model handles lengths correctly if passed
            y_pred = model(x_padSeq, original_seq_lens.cpu()) # Ensure lengths are CPU for model

            batch_total_loss, num_valid_in_batch, _ = get_mse_entire_sequence(y_padSeq=y_padSeq,
                                                                              y_pred=y_pred,
                                                                              original_seq_lens=original_seq_lens,
                                                                              loss_fn=loss_fn,
                                                                              device=device)

            # Avoid division by zero if a batch has no valid elements (e.g., all padding)
            if num_valid_in_batch > 0:
                 # --- Backward pass and optimize ---
                 batch_total_loss.backward()
                 # Optional: Gradient clipping
                 # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                 optimizer.step()

                 total_train_loss_sum += batch_total_loss.item()
                 total_valid_elements += num_valid_in_batch
            else:
                 # Handle case where batch has no valid elements (skip backward/step)
                 print(f"Warning: Batch {i+1} has no valid elements based on lengths.")
                 pass # No loss contribution, no backward pass

        # --- End of Training Epoch ---
        # Calculate average loss per valid element for the epoch
        avg_train_loss = total_train_loss_sum / total_valid_elements if total_valid_elements > 0 else 0
        print(f" | Train Loss (avg/element): {avg_train_loss:.4f} | ", end="")

        if writer is not None:
            writer.add_scalar("Loss/train_avg_per_element", avg_train_loss, epoch)

        #### VALIDATION ####
        val_loss, val_predictions_list = evaluate_data(model,
                                                       test_dataLoader,
                                                       loss_fn,
                                                       device,
                                                       compare_entire_seq=compare_entire_seq_in_val)
        if compare_entire_seq_in_val:
            print(f"Val Loss: {val_loss:.4f}")
        else:
            print(f"Val RMSE (last step): {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/validation_RMSE", val_loss, epoch)
            writer.flush()

        # --- Early Stopping & Model Saving ---
        if val_loss < best_val_loss:
            print(f"Validation RMSE improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation RMSE did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"\n--- Early stopping triggered after {epoch} epochs ---")
            break

        # --- Visualization ---
        if epoch % 10 == 0 and y_test is not None and len(val_predictions_list) > 0:
            # ---- Calculate rmse using only final steps if validation compares entire sequences ----
            val_rmse, _ = evaluate_data(model,
                                        test_dataLoader,
                                        loss_fn,
                                        device,
                                        compare_entire_seq=False) if compare_entire_seq_in_val else val_loss
            # Consolidate predictions from batches
            all_val_preds = np.concatenate(val_predictions_list, axis=0)
            # Call your visualization function
            visualize(all_val_preds, y_test, 100, val_rmse, epoch, writer)

    # --- End of Training ---
    if writer is not None:
        writer.flush()
        # writer.close() # Close writer if finished
        
    # Load the best model weights back before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    if compare_entire_seq_in_val:
        val_rmse, _ = evaluate_data(model,
                                    test_dataLoader,
                                    loss_fn,
                                    device,
                                    compare_entire_seq=False) if compare_entire_seq_in_val else best_val_loss
            
        print(f"\nTraining finished after epoch {epoch}. Best Validation RMSE: {best_val_loss:.4f}. Validation RMSE (Last Step): {val_rmse:.4f}.")
    else:
        val_rmse = best_val_loss
        print(f"\nTraining finished after epoch {epoch}. Best Validation RMSE: {best_val_loss:.4f}.")
    
    return avg_train_loss, best_val_loss, val_rmse, epoch # Return last train loss, best val loss, final epoch