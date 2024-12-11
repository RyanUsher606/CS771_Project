import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
import numpy as np
from PIL import Image

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.model import MovieClassifier  # Import your model definition
from src.data.movie_dataset import MovieDataset, get_genre_list  # Import your dataset code

def main():
    # Paths to your CSV splits and processed posters
    test_csv  = r"C:\Users\ryanu\Desktop\Newfolder\dataset\training_testing\test.csv"
    processed_folder = r"C:\Users\ryanu\Desktop\Newfolder\dataset\filtered_dataset\processed_posters"
    metadata_path = r"C:\Users\ryanu\Desktop\Newfolder\dataset\filtered_dataset\filtered_movies_metadata.csv"  
    save_path = r"C:\Users\ryanu\Desktop\Newfolder\model_checkpoint\model_epoch_20.pth"

    # Directory to save example images
    save_examples_dir = "saved_examples"
    os.makedirs(save_examples_dir, exist_ok=True)

    # Get genre list
    genre_list = get_genre_list(metadata_path)
    num_genres = len(genre_list)

    # Create test dataset & DataLoader
    # IMPORTANT: Ensure that MovieDataset returns (img, target, movie_id) in __getitem__
    test_dataset = MovieDataset(csv_path=test_csv, processed_folder=processed_folder, genre_list=genre_list, return_id=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MovieClassifier(backbone="resnet18", num_genres=num_genres).to(device)

    model_state_dict = torch.load(save_path, map_location=device)  
    model.load_state_dict(model_state_dict)

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    test_loss = 0.0

    all_targets = []
    all_preds = []

    # We'll keep track of the IDs, so we can map back after predictions
    movie_ids_list = []

    # To store indices of correct and incorrect predictions
    correct_indices = []
    incorrect_indices = []

    needed_correct = 2
    needed_incorrect = 2

    # Run inference on test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on Test Set")):
            images, targets, movie_ids = batch
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            test_loss += loss.item()

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            # Convert probabilities to binary predictions (threshold=0.5)
            preds = (probs > 0.5).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            movie_ids_list.extend(movie_ids)

            # Determine correctness: exact match per sample
            correct_per_sample = torch.all(preds == targets, dim=1).cpu().numpy()

            # Check if we still need correct/incorrect samples
            if (needed_correct > 0) or (needed_incorrect > 0):
                batch_size = images.size(0)
                for i in range(batch_size):
                    if correct_per_sample[i] and needed_correct > 0:
                        correct_indices.append((batch_idx, i))
                        needed_correct -= 1
                    elif not correct_per_sample[i] and needed_incorrect > 0:
                        incorrect_indices.append((batch_idx, i))
                        needed_incorrect -= 1

                    if needed_correct == 0 and needed_incorrect == 0:
                        # We have what we need
                        break

            # If we have found all needed samples, we still must finish loop to get full metrics
            # but no more logic needed here for collecting examples.

    avg_test_loss = test_loss / len(test_loader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Exact Match Accuracy: fraction of samples perfectly predicted
    exact_match_acc = (all_targets == all_preds).all(axis=1).mean()

    # At least one match accuracy: fraction of samples with at least one correct label
    at_least_one_match = ((all_targets * all_preds).sum(axis=1) > 0).mean()

    # Hamming loss: fraction of individual label errors
    h_loss = hamming_loss(all_targets, all_preds)

    # Label-based accuracy (test accuracy):
    # Flatten the arrays to compare each label prediction as an independent binary classification
    test_acc = accuracy_score(all_targets.flatten(), all_preds.flatten())

    print(f"Test Loss = {avg_test_loss:.4f}")
    print(f"Test Exact Match Accuracy (Accuracy All/Match) = {exact_match_acc:.4f}")
    print(f"Test At Least One Match = {at_least_one_match:.4f}")
    print(f"Test Hamming Loss = {h_loss:.4f}")
    print(f"Test Accuracy (Label-Based) = {test_acc:.4f}")

    # Compute per-genre metrics
    genre_precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    genre_recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    genre_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    genre_counts = all_targets.sum(axis=0)

    print("\nGenre      Recall Precision F1  Count")
    for i, genre in enumerate(genre_list):
        print(f"{genre:<10s} {genre_recall[i]:.2f}  {genre_precision[i]:.2f}    {genre_f1[i]:.2f}  {int(genre_counts[i])}")

    # --- Save the selected correct and incorrect examples ---
    # We'll load the original poster using the movie_id.
    def save_example(index_tuple, is_correct):
        batch_idx, i_in_batch = index_tuple
        global_idx = batch_idx * test_loader.batch_size + i_in_batch

        # Retrieve movie_id
        movie_id = movie_ids_list[global_idx]
        image_path = os.path.join(r"C:\Users\ryanu\Desktop\Newfolder\dataset\filtered_dataset\posters", f"{movie_id}.jpg")

        # Load the original image
        pil_img = Image.open(image_path).convert("RGB")

        # Retrieve predicted and true labels
        pred_labels = all_preds[global_idx]
        true_labels = all_targets[global_idx]

        # Get genre names for predicted and true labels
        pred_genre_str = "-".join([genre_list[j] for j, val in enumerate(pred_labels) if val == 1])
        if pred_genre_str == "":
            pred_genre_str = "none"
        true_genre_str = "-".join([genre_list[j] for j, val in enumerate(true_labels) if val == 1])
        if true_genre_str == "":
            true_genre_str = "none"

        # Filename
        status = "correct" if is_correct else "incorrect"
        filename = f"{status}_{movie_id}_pred_{pred_genre_str}_true_{true_genre_str}.png"
        filepath = os.path.join(save_examples_dir, filename)
        pil_img.save(filepath)
        print(f"Saved {status} example at: {filepath}")

    # Save correct examples
    for idx_tuple in correct_indices:
        save_example(idx_tuple, is_correct=True)

    # Save incorrect examples
    for idx_tuple in incorrect_indices:
        save_example(idx_tuple, is_correct=False)

if __name__ == '__main__':
    main()
