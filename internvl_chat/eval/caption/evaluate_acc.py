import json
import re

def extract_abnormality(caption):
    """
    Extracts the last matching abnormality label from a given caption string.

    This function searches the input caption for any of the predefined abnormality labels
    and returns the last one found. If no label is found, it returns an empty string.

    Args:
        caption (str): The text string to search for abnormality labels.

    Returns:
        str: The last matched abnormality label, or an empty string if none are found.

    Usage:
        >>> extract_abnormality("The scan shows Cardiac abnormalities and Abnormal placenta.")
        'Abnormal placenta'
        >>> extract_abnormality("No abnormalities detected. Normal.")
        'Normal'
        >>> extract_abnormality("No findings.")
        ''
    """
    labels = ["Cardiac abnormalities", "Abnormal placenta", "Facial abnormalities", "Abdominal abnormalities", "Abnormal brain function", "Abnormal limbs", "Normal"]
    pattern = '(' + '|'.join(labels) + ')'
    matches = re.findall(pattern, caption)
    if matches:
        return matches[-1]
    return ""

def is_abnormal(label):
    return label != "Normal"

def extract_abnormalities(caption):
    """
    Extracts a set of abnormality labels found in a given caption string.

    This function checks if any predefined abnormality labels are present in the input caption.
    If none of the abnormality labels are found and the word "Normal" is present, it adds "Normal" to the result set.

    Args:
        caption (str): The text caption to be analyzed.

    Returns:
        set: A set of strings representing the abnormalities found in the caption, or {"Normal"} if no abnormalities are found but "Normal" is present.

    Usage:
        >>> extract_abnormalities("The patient shows Cardiac abnormalities and Abnormal limbs.")
        {'Cardiac abnormalities', 'Abnormal limbs'}

        >>> extract_abnormalities("Normal brain function observed.")
        {'Normal'}
    """
    abnormal_labels = ["Cardiac abnormalities", "Abnormal placenta", "Facial abnormalities", "Abdominal abnormalities", "Abnormal brain function", "Abnormal limbs"]
    found = set()
    for label in abnormal_labels:
        if label in caption:
            found.add(label)
    if not found and "Normal" in caption:
        found.add("Normal")
    return found

def calculate_acc(gt_path, pred_path):
    """
    Evaluate caption prediction accuracy using various metrics.
    This function compares predicted captions to ground truth captions (in COCO format)
    and computes several accuracy metrics for image captioning tasks, including:
      - Single-label multi-class accuracy (for samples with only one abnormality label)
      - Per-category accuracy (for each abnormality category)
      - Binary classification metrics (normal vs. abnormal, for all samples)
      - Multi-label accuracy (for samples with multiple abnormalities)
      - Overall sample accuracy (strict multi-label match, for all samples)
    Args:
        gt_path (str): Path to the ground truth JSON file (COCO format, with 'annotations').
        pred_path (str): Path to the prediction JSON file (list of dicts with 'image_id' and 'caption').
    Returns:
        list[str]: A list of formatted strings summarizing the evaluation results for each metric.
    Usage Example:
        >>> outlog = calculate_acc('ground_truth.json', 'prediction.json')
        >>> for line in outlog:
        ...     print(line)
    """
    # Load prediction.json file
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load ground_truth.json file (COCO format)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_annotations = gt_data.get('annotations', [])
    
    # Use image_id as key for predictions for easy lookup
    pred_dict = {item['image_id']: item for item in predictions}
    
    total_samples = 0  # For binary and overall sample metrics (all samples)
    
    # ---------------------------
    # Single-label multi-class metrics (only for samples with one label in ground truth)
    single_label_total = 0
    single_label_correct = 0
    per_category_total = {}
    per_category_correct = {}
    
    # ---------------------------
    # Binary classification metrics (all samples)
    TP = 0  # ground truth abnormal and predicted abnormal
    TN = 0  # ground truth normal and predicted normal
    FP = 0  # ground truth normal but predicted abnormal
    FN = 0  # ground truth abnormal but predicted normal
    
    # ---------------------------
    # Multi-label metrics (only for samples with multiple abnormal labels in ground truth)
    multi_label_total = 0
    multi_label_correct = 0
    
    # ---------------------------
    # Overall sample accuracy (multi-label method, all samples)
    overall_multi_correct = 0
    outlog=[]
    
    for ann in gt_annotations:
        total_samples += 1
        image_id = ann['image_id']
        gt_caption = ann['caption']
        gt_label_single = extract_abnormality(gt_caption)      # Single label (last occurrence of keyword)
        gt_labels_multi = extract_abnormalities(gt_caption)    # Multi-label set
        
        pred_item = pred_dict.get(image_id)
        if pred_item:
            pred_caption = pred_item['caption']
            pred_label_single = extract_abnormality(pred_caption)
            pred_labels_multi = extract_abnormalities(pred_caption)
        else:
            pred_label_single = ""
            pred_labels_multi = set()
        
        # ---------------------------
        # Overall sample accuracy (multi-label method): all samples
        if gt_labels_multi == pred_labels_multi:
            overall_multi_correct += 1
        
        # ---------------------------
        # Single-label (multi-class) metrics: only for samples with one label in ground truth
        if len(gt_labels_multi) == 1:
            single_label_total += 1
            per_category_total[gt_label_single] = per_category_total.get(gt_label_single, 0) + 1
            if pred_label_single == gt_label_single:
                single_label_correct += 1
                per_category_correct[gt_label_single] = per_category_correct.get(gt_label_single, 0) + 1
            else:
                per_category_correct[gt_label_single] = per_category_correct.get(gt_label_single, 0)
        
        # ---------------------------
        # Binary classification metrics: based on single label, "Normal" is normal, others are abnormal (all samples)
        gt_binary = is_abnormal(gt_label_single)
        pred_binary = is_abnormal(pred_label_single)
        if gt_binary and pred_binary:
            TP += 1
        elif (not gt_binary) and (not pred_binary):
            TN += 1
        elif (not gt_binary) and pred_binary:
            FP += 1
        elif gt_binary and (not pred_binary):
            FN += 1
        
        # ---------------------------
        # Multi-label metrics: only for samples with multiple abnormal labels in ground truth
        if len(gt_labels_multi) > 1:
            multi_label_total += 1
            if gt_labels_multi == pred_labels_multi:
                multi_label_correct += 1
    
    # ---------------------------
    # Output single-label multi-class metrics
    if single_label_total > 0:
        average_acc=0
        overall_single_label_acc = single_label_correct / single_label_total
        outlog.append("[Single-label Multi-class Metrics] (only single-label samples)")
        outlog.append("Overall accuracy: {:.2%} ({} / {})".format(overall_single_label_acc, single_label_correct, single_label_total))
        outlog.append("Per-category accuracy:")
        for category, total_count in per_category_total.items():
            correct_count = per_category_correct.get(category, 0)
            cat_acc = correct_count / total_count if total_count > 0 else 0
            average_acc += cat_acc
            outlog.append("  {}: {:.2%} ({} / {})".format(category, cat_acc, correct_count, total_count))
        average_acc = average_acc / 9
        outlog.append("Average accuracy: {:.2%}".format(average_acc))
    else:
        outlog.append("No single-label samples, cannot compute single-label multi-class metrics.")
    
    # ---------------------------
    # Output binary classification metrics (all samples)
    binary_accuracy = (TP + TN) / total_samples if total_samples > 0 else 0
    abnormal_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    abnormal_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    outlog.append("[Binary Classification Metrics] (all samples)")
    outlog.append("Overall accuracy: {:.2%} (TP+TN={} / Total samples={})".format(binary_accuracy, (TP + TN), total_samples))
    outlog.append("Abnormal (positive) recall: {:.2%} (TP={} / (TP+FN)={})".format(abnormal_recall, TP, (TP + FN)))
    outlog.append("Abnormal (positive) precision: {:.2%} (TP={} / (TP+FP)={})".format(abnormal_precision, TP, (TP + FP)))
    
    # ---------------------------
    # Output multi-label metrics (only for samples with multiple abnormalities)
    outlog.append("[Multi-label Metrics] (only for samples with multiple abnormalities)")
    if multi_label_total > 0:
        multi_label_accuracy = multi_label_correct / multi_label_total
        outlog.append("Multiple abnormality accuracy (strict match): {:.2%} ({} / {})".format(multi_label_accuracy, multi_label_correct, multi_label_total))
    else:
        outlog.append("No samples with multiple abnormalities, cannot compute multi-label accuracy.")
    
    # ---------------------------
    # Output overall sample accuracy (multi-label method for all samples)
    overall_multi_label_acc = overall_multi_correct / total_samples if total_samples > 0 else 0
    outlog.append("[Overall Sample Accuracy]")
    outlog.append("Overall sample accuracy: {:.2%} ({} / {})".format(overall_multi_label_acc, overall_multi_correct, total_samples))
    return outlog

if __name__ == '__main__':
    log  = calculate_acc('/public/Report-Ge/code/InternVL-wsy/internvl_chat/datasets/datasets/coco_en_test_with_retrieval_info_confidence_one_stage.json','/public/Report-Ge/code/InternVL-wsy/internvl_chat/wsy_results/rag_test_en_formal_retrieval_data_faiss_one_to_one_250516184926.json')
    for i in range(len(log)):
        print(log[i])
