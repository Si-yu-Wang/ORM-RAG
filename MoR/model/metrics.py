from tqdm import tqdm
import torch
from model.retriever import search_retriever
from model.extract_feature import extract_embeddings
from model.tools import process_images, judge_label

def compute_metrics(model, label_form, device, judge_data, index, dataset, labels):
    """
    Compute the top-1 accuracy metric for the model.

    Args:
        model: The model used for feature extraction.
        label_form: Label format used for filtering labels.
        device: Computation device (e.g., 'cuda' or 'cpu').
        judge_data: Data used for label judgment.
        index: Retriever index.
        dataset: Dataset containing image paths to be evaluated.
        labels: List of all image labels in the retrieval database.

    Returns:
        top_1_acc: Top-1 accuracy.
    """
    top_1 = 0  # Top-1 hit count
    total_queries = 0  # Total number of queries

    # Iterate over each item in the dataset
    for _, _, image_paths in tqdm(dataset, desc="Evaluating metrics"):
        # Get current image labels and group dictionary
        judgement_labels, rag_dict = judge_label(judge_data, image_paths)
        # Iterate over each label
        for per_label in judgement_labels:
            # Only process specific labels
            if per_label in [label_form*2, label_form*2+1]:
                group_images = rag_dict[per_label]  # Get image group for the label
                image_tensor = process_images(group_images)  # Process images to tensor
                image_tensor = image_tensor.to(device)  # Move to specified device
                image_tensor = extract_embeddings(model, image_tensor)  # Extract features

                # Retrieve most similar image indices
                _, I = search_retriever(index, image_tensor.cpu().numpy())
                per_label = per_label % 2  # Normalize label

                # Get labels of retrieved results
                retrieved_labels = [labels[i] for i in I]
                # Check if top-1 is hit
                if per_label in retrieved_labels[:1]:
                    top_1 += 1
                total_queries += 1
        # Return accuracy every 1000 queries
        if total_queries % 1000 == 0 and total_queries > 0:
            top_1_acc = top_1 / total_queries
            torch.cuda.empty_cache()
            return top_1_acc
    torch.cuda.empty_cache()
    # Return final top-1 accuracy
    return top_1_acc