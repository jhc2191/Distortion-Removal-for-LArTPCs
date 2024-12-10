import cv2
import numpy as np
import glob
import os
import csv
from scipy.spatial.distance import directed_hausdorff

def line_similarity_metric(img_ground_truth, img_predicted, method='iou'):
    """
    Compute a similarity metric focusing on line location rather than background color.
    
    Parameters:
        img_ground_truth: BGR image (numpy array)
        img_predicted: BGR image (numpy array)
        method: str, 'iou' or 'dice'
        
    Returns:
        A floating point similarity score between 0 and 1.
    """
    # Convert to grayscale
    gray_gt = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate the line (Adjust threshold value based on your images)
    _, mask_gt = cv2.threshold(gray_gt, 128, 255, cv2.THRESH_BINARY_INV)
    _, mask_pred = cv2.threshold(gray_pred, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Convert masks to boolean
    mask_gt_bool = (mask_gt > 0)
    mask_pred_bool = (mask_pred > 0)
    
    intersection = np.logical_and(mask_gt_bool, mask_pred_bool).sum()
    union = np.logical_or(mask_gt_bool, mask_pred_bool).sum()

    if method.lower() == 'iou':
        # Intersection over union
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    elif method.lower() == 'dice':
        # Dice coefficient
        gt_sum = mask_gt_bool.sum()
        pred_sum = mask_pred_bool.sum()
        denom = gt_sum + pred_sum
        if denom == 0:
            return 1.0 if intersection == 0 else 0.0
        return (2.0 * intersection) / denom
    else:
        raise ValueError("Unknown method: choose 'iou' or 'dice'")

def compute_hausdorff_distance(img_ground_truth, img_predicted):
    """
    Compute the Hausdorff Distance between the ground truth and predicted lines.
    
    Parameters:
        img_ground_truth: BGR image (numpy array)
        img_predicted: BGR image (numpy array)
        
    Returns:
        Hausdorff Distance as a floating point number.
    """
    # Convert to grayscale
    gray_gt = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges_gt = cv2.Canny(gray_gt, 50, 150)
    edges_pred = cv2.Canny(gray_pred, 50, 150)
    
    # Find coordinates of edge pixels
    points_gt = np.column_stack(np.where(edges_gt > 0))
    points_pred = np.column_stack(np.where(edges_pred > 0))
    
    if len(points_gt) == 0 or len(points_pred) == 0:
        print("Warning: One of the images has no edge pixels.")
        return np.nan
    
    # Compute directed Hausdorff distances
    hausdorff_gt_to_pred = directed_hausdorff(points_gt, points_pred)[0]
    hausdorff_pred_to_gt = directed_hausdorff(points_pred, points_gt)[0]
    
    # The Hausdorff Distance is the maximum of the two directed distances
    hausdorff_distance = max(hausdorff_gt_to_pred, hausdorff_pred_to_gt)
    
    return hausdorff_distance

def compute_chamfer_distance(img_ground_truth, img_predicted):
    """
    Compute the Chamfer Distance between the ground truth and predicted lines.
    
    Parameters:
        img_ground_truth: BGR image (numpy array)
        img_predicted: BGR image (numpy array)
        
    Returns:
        Chamfer Distance as a floating point number.
    """
    from scipy.spatial import cKDTree

    # Convert to grayscale
    gray_gt = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges_gt = cv2.Canny(gray_gt, 50, 150)
    edges_pred = cv2.Canny(gray_pred, 50, 150)
    
    # Find coordinates of edge pixels
    points_gt = np.column_stack(np.where(edges_gt > 0))
    points_pred = np.column_stack(np.where(edges_pred > 0))
    
    if len(points_gt) == 0 or len(points_pred) == 0:
        print("Warning: One of the images has no edge pixels.")
        return np.nan
    
    # Build KD-Trees
    tree_gt = cKDTree(points_gt)
    tree_pred = cKDTree(points_pred)
    
    # Compute nearest neighbor distances
    distances_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    distances_pred_to_gt, _ = tree_gt.query(points_pred, k=1)
    
    # Compute Chamfer Distance
    chamfer_dist = (np.mean(distances_gt_to_pred) + np.mean(distances_pred_to_gt)) / 2
    
    return chamfer_dist

# Main code
if __name__ == "__main__":
    # Paths to ground truth and predicted images
    gt_dir = "/Users/jackcleeve/Desktop/Research/distortionFields/data creation/base_data"
    pred_main_dir = "/Users/jackcleeve/Desktop/updated_median_corrected_images_8"
    
    # Output CSV file path
    output_csv = "similarity_scores.csv"
    
    # Get all ground truth images sorted by filename
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.*")))  # Adjust extension if needed
    pred_images = sorted(glob.glob(os.path.join(pred_main_dir, "*.*")))
    """""
    # Get all predicted images from all batch_* folders
    batch_folders = sorted(glob.glob(os.path.join(pred_main_dir, "batch_*")))
    pred_images = []
    for batch_folder in batch_folders:
        # Add all images from this batch folder sorted by filename
        batch_images = sorted(glob.glob(os.path.join(batch_folder, "*.*")))  # Adjust extension if needed
        pred_images.extend(batch_images)
    """""
    # Check we have the same number of GT and predicted images
    if len(gt_images) != len(pred_images):
        raise ValueError(
            f"Number of ground truth images ({len(gt_images)}) does not match "
            f"the number of predicted images ({len(pred_images)})."
        )
    # Prepare to store results
    results = []
    
    # Compute similarity scores for each pair
    for idx, (gt_path, pred_path) in enumerate(zip(gt_images, pred_images)):
        img_gt = cv2.imread(gt_path)
        img_pred = cv2.imread(pred_path)
        
        if img_gt is None:
            print(f"Warning: Unable to read ground truth image: {gt_path}. Skipping.")
            continue
        if img_pred is None:
            print(f"Warning: Unable to read predicted image: {pred_path}. Skipping.")
            continue
        
        # Compute similarity metrics
        iou_score = line_similarity_metric(img_gt, img_pred, method='iou')
        dice_score = line_similarity_metric(img_gt, img_pred, method='dice')
        hausdorff_distance = compute_hausdorff_distance(img_gt, img_pred)
        chamfer_distance = compute_chamfer_distance(img_gt, img_pred)
        
        # Append to results
        results.append({
            "ground_truth_filename": os.path.basename(gt_path),
            "predicted_filename": os.path.basename(pred_path),
            "iou_score": iou_score,
            "dice_score": dice_score,
            "hausdorff_distance": hausdorff_distance,
            "chamfer_distance": chamfer_distance
        })
        
        # Handle NaN for Hausdorff Distance
        hausdorff_display = f"{hausdorff_distance:.2f}" if not np.isnan(hausdorff_distance) else "NaN"
        
        print(f"Pair {idx+1}: {os.path.basename(gt_path)} vs {os.path.basename(pred_path)} | IoU: {iou_score:.4f} | Dice: {dice_score:.4f} | Hausdorff Distance: {hausdorff_display}")
    
    # Compute average scores
    if results:
        # Filter out NaN Hausdorff distances for averaging
        valid_hausdorff_distances = [r["hausdorff_distance"] for r in results if not np.isnan(r["hausdorff_distance"])]
        valid_chamfer_distances = [r["chamfer_distance"] for r in results if not np.isnan(r["chamfer_distance"])]        
        avg_iou = np.mean([r["iou_score"] for r in results])
        avg_dice = np.mean([r["dice_score"] for r in results])
        avg_hausdorff = np.mean(valid_hausdorff_distances) if valid_hausdorff_distances else np.nan
        avg_chamfer = np.mean(valid_chamfer_distances) if valid_chamfer_distances else np.nan

        
        print(f"\nAverage IoU score across all images: {avg_iou:.4f}")
        print(f"Average Dice score across all images: {avg_dice:.4f}")
        if not np.isnan(avg_hausdorff):
            print(f"Average Hausdorff Distance across all images: {avg_hausdorff:.2f}")
        else:
            print("Average Hausdorff Distance across all images: NaN (No valid distances)")
        if not np.isnan(avg_chamfer):
            print(f"Average Chamfer Distance across all images: {avg_chamfer:.2f}")
        else:
            print("Average Chamfer Distance across all images: NaN (No valid distances)")
    else:
        print("No results to compute averages.")
    
    # Write results to CSV
    csv_columns = ["ground_truth_filename", "predicted_filename", "iou_score", "dice_score", "hausdorff_distance"]
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
        print(f"\nSimilarity scores have been saved to {output_csv}")
    except IOError as e:
        print(f"I/O error while writing CSV: {e}")
