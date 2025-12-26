import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import copy
from scipy.optimize import linear_sum_assignment
from SAM2UNet import SAM2UNet
import torch
import imageio
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 
Visualize_Lines = True
Visualize_Corners = True

# Configuration
INPUT_SIZE = 512
NUM_CLASSES = 7
CHECKPOINT_PATH = "/home/yousof/Downloads/35epochs_0008lr_28batchsize_512inputsize.pth"
IMAGE_PATH = "/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data3/train/images/170330-3-_jpg.rf.8fcd63ba579269f120084b7269a95ea9.jpg"
OUTPUT_PATH = "output_segmentation.png"

def rgb_loader(path):
    """Load image and convert to RGB"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
 
def colorize_segmentation(seg_map, num_classes):
    """
    Convert class indices to RGB colors for visualization
    seg_map: [H, W] numpy array with class indices
    Returns: [H, W, 3] RGB image
    """
    colors = [
        [0, 0, 0],       # Class 0: Black (background)
        [255, 0, 0],     # Class 1: Red
        [0, 255, 0],     # Class 2: Green
        [0, 0, 255],     # Class 3: Blue
        [255, 255, 0],   # Class 4: Yellow
        [255, 0, 255],   # Class 5: Magenta
        [0, 255, 255],   # Class 6: Cyan
    ]
    
    # Extend if needed
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(num_classes):
        mask = seg_map == cls
        colored[mask] = colors[cls]
    
    return colored

def run_inference(image_path, checkpoint_path, input_size=512, num_classes=7, output_path=None):
    """
    Run inference on a single image
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        input_size: Input size for the model
        num_classes: Number of segmentation classes
        output_path: Path to save output (optional)
    
    Returns:
        binary_mask: [H, W] numpy array with binary mask (0=background, 255=foreground)
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess image
    img = rgb_loader(image_path)
    original_size = img.size  # (W, H)
    print(f"Original image size: {original_size}")
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Input tensor shape: {img_tensor.shape}")
    
    # Load model
    model = SAM2UNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()
    print("Model loaded successfully")
    
    # Run inference
    with torch.no_grad():
        res, _, _ = model(img_tensor)  # [1, num_classes, H, W]
        print(f"Output shape: {res.shape}")
        
        # Resize to original image size
        res = F.interpolate(res, size=(original_size[1], original_size[0]), 
                          mode='bilinear', align_corners=False)
        
        # Get class predictions
        pred_class = torch.argmax(res, dim=1).squeeze().cpu().numpy()  # [H, W]
        print(f"Prediction shape: {pred_class.shape}")
        
        # Get unique classes
        unique_classes = np.unique(pred_class)
        print(f"Predicted classes: {unique_classes}")
        
        # Create binary mask (background=0, all other classes=255)
        binary_mask = np.where(pred_class > 0, 255, 0).astype(np.uint8)
        print(f"Binary mask: 0 (background) and 255 (foreground)")
        
        # Save output if path provided
        if output_path:
            imageio.imsave(output_path, binary_mask)
            print(f"Binary mask saved to: {output_path}")
    
    return binary_mask

# Main execution
binary_mask = run_inference(
    image_path=IMAGE_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    input_size=INPUT_SIZE,
    num_classes=NUM_CLASSES,
    output_path=OUTPUT_PATH
)

# COnvert to bgr
image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)




def detect_corners_convex_hull(binary_mask):
    """
    Using convex hull and extreme points to find the outermost corners
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get convex hull
    hull = cv2.convexHull(largest_contour)
    
    # Simplify to get 4 corners (approximate as quadrilateral)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # If we don't get exactly 4 points, try adjusting epsilon
    attempts = 0
    while len(approx) != 4 and attempts < 20:
        if len(approx) > 4:
            epsilon *= 1.1
        else:
            epsilon *= 0.9
        approx = cv2.approxPolyDP(hull, epsilon, True)
        attempts += 1
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    else:
        raise ValueError("Could not approximate to 4 corners.")


def visualize_corners(binary_mask, corners, method_name="", image=None):
    """
    Visualize the detected corners on the mask
    """
    # Create RGB image for visualization
    if image is not None:
        vis_img = copy.deepcopy(image)
    else:
        vis_img = cv2.cvtColor(binary_mask.astype(np.uint8) * 255, 
                           cv2.COLOR_GRAY2RGB)
        
    # Draw lines connecting corners
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i+1)%4].astype(int))
        cv2.line(vis_img, pt1, pt2, (100, 100, 255), 2)
    
    return vis_img


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)


def find_corner_pairs(corners1, corners2, excluded_corners1=None, excluded_corners2=None): 
    """
    Find the two corner pairs that minimize total distance using Hungarian matching.
    Args:
        corners1: 4x2 array of corners from vertebra 1
        corners2: 4x2 array of corners from vertebra 2
        excluded_corners1: Set of indices from corners1 that should not be used
        excluded_corners2: Set of indices from corners2 that should not be used
    Returns:
        Tuple of (selected_pairs, used_corners1, used_corners2)
    """
    if excluded_corners1 is None:
        excluded_corners1 = set()
    if excluded_corners2 is None:
        excluded_corners2 = set()
    
    # Get available corners
    available_idx1 = [i for i in range(len(corners1)) if i not in excluded_corners1]
    available_idx2 = [j for j in range(len(corners2)) if j not in excluded_corners2]
    
    if len(available_idx1) < 2 or len(available_idx2) < 2:
        raise ValueError("Not enough available corners for matching")
    
    # Build cost matrix for available corners
    n1 = len(available_idx1)
    n2 = len(available_idx2)
    cost_matrix = np.zeros((n1, n2))
    
    for i, idx1 in enumerate(available_idx1):
        for j, idx2 in enumerate(available_idx2):
            cost_matrix[i, j] = calculate_distance(corners1[idx1], corners2[idx2])
    
    # If we need exactly 2 pairs and have more corners, we need to solve a restricted assignment
    # Try all combinations of selecting 2 from each set and find the best matching
    from itertools import combinations
    
    best_total_distance = float('inf')
    best_pairs = None
    best_used1 = None
    best_used2 = None
    
    # Generate all possible combinations of 2 corners from each set
    for combo1 in combinations(range(n1), 2):
        for combo2 in combinations(range(n2), 2):
            # Build 2x2 cost matrix for this combination
            small_cost = np.zeros((2, 2))
            for i, idx1 in enumerate(combo1):
                for j, idx2 in enumerate(combo2):
                    small_cost[i, j] = cost_matrix[idx1, idx2]
            
            # Apply Hungarian algorithm to this 2x2 matrix
            row_ind, col_ind = linear_sum_assignment(small_cost)
            total_distance = small_cost[row_ind, col_ind].sum()
            
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                # Map back to original indices
                best_used1 = {available_idx1[combo1[i]] for i in row_ind}
                best_used2 = {available_idx2[combo2[j]] for j in col_ind}
                best_pairs = [
                    (tuple(corners1[available_idx1[combo1[row_ind[k]]]]), 
                     tuple(corners2[available_idx2[combo2[col_ind[k]]]]))
                    for k in range(2)
                ]
    
    return best_pairs, best_used1, best_used2


def check_vertebrae_ordering(s1_box, vertebra_box):
    """
    Check the y coordinates of s1 box and any other vertebra box to determine if upside down.
    """
    s1_average_y = (s1_box[1] + s1_box[3]) / 2
    vertebra_average_y = (vertebra_box[1] + vertebra_box[3]) / 2
    return s1_average_y > vertebra_average_y # if True, then no need to flip


def align_spine_and_get_transform(image, vertebra_data):
    print("Aligning spine...")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    
    # --- PCA Logic ---
    y_idxs, x_idxs = np.nonzero(img)
    coords = np.vstack([x_idxs, y_idxs]).T.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    center_x, center_y = mean[0, 0], mean[0, 1]
    
    angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    rotation_angle = np.degrees(angle_rad) + 90 

    # --- Matrix Calculation ---
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
    M[0, 2] += (w/2 - center_x)
    M[1, 2] += (h/2 - center_y)

    # Warp
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    # Transform vertebrae data
    dims = (w, h)
    vertebra_data_transformed = transform_vertebrae_data(vertebra_data, M, False, dims)

    # --- Upside Down Check (Bounding Box Method) ---
    # This is my new method to check if the spine is upside down
    # S1 box values must have the highest y values (lowest in image)
    if not check_vertebrae_ordering(vertebra_data_transformed['s1']['box_coordinates'], vertebra_data_transformed['l3']['box_coordinates']):
        # Since the order is false, then flip the image and all vertebrae data
        aligned_img = cv2.rotate(aligned_img, cv2.ROTATE_180)
        vertebra_data_transformed = transform_vertebrae_data(vertebra_data, M, True, dims) # Recalculate with flip=True
        
    # RETURN EVERYTHING
    return aligned_img, vertebra_data_transformed


def transform_vertebrae_data(data, M, flipped, img_dims):
    w, h = img_dims
    data_copy = copy.deepcopy(data)
    # We iterate over every vertebra (l1, l2, s1, etc.)
    for key, value in data_copy.items():
        # 1. Get the absolute corners (N, 2)
        corners = np.array(value['corners_absolute'], dtype=np.float32)
        
        # 2. Reshape for OpenCV: (N, 1, 2)
        # OpenCV needs this specific 3D shape for the transform function
        corners_reshaped = np.array([corners]) 
        
        # 3. Apply the Affine Transform (Rotation + Centering)
        # This aligns the points exactly like the image pixels
        transformed_corners = cv2.transform(corners_reshaped, M)[0]
        
        # 4. Apply 180 Flip if necessary
        if flipped:
            # print("Flipping vertebrae data...")
            # Formula: new_x = width - x, new_y = height - y
            transformed_corners[:, 0] = w - transformed_corners[:, 0]
            transformed_corners[:, 1] = h - transformed_corners[:, 1]

        # 5. Update the Dictionary
        # Round to integers
        transformed_corners = transformed_corners.astype(np.int32)
        value['corners_absolute'] = transformed_corners
        
        # Recalculate Bounding Box [min_x, min_y, max_x, max_y]
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        
        value['box_coordinates'] = [int(min_x), int(min_y), int(max_x), int(max_y)]
        
        # Update relative corners (corners relative to the new box)
        # Logic: relative = absolute - top_left_of_box
        top_left = np.array([min_x, min_y])
        value['corners'] = transformed_corners - top_left

    return data_copy


def calculate_slope(l5_points, s1_points, image=None):
    """
    Calculate the slope between two points.
    Each point is a tuple (x, y).
    Slope = (y2 - y1) / (x2 - x1)
    
    Pairs leftmost L5 point with leftmost S1 point, and rightmost with rightmost.
    """
    average_slope = 0
    l5_point1, l5_point2 = l5_points
    s1_point1, s1_point2 = s1_points
    
    # Sort L5 points by x-coordinate (left to right)
    if l5_point1[0] < l5_point2[0]:
        l5_left = l5_point1
        l5_right = l5_point2
    else:
        l5_left = l5_point2
        l5_right = l5_point1
    
    # Sort S1 points by x-coordinate (left to right)
    if s1_point1[0] < s1_point2[0]:
        s1_left = s1_point1
        s1_right = s1_point2
    else:
        s1_left = s1_point2
        s1_right = s1_point1
    
    # Pair left with left
    if image is not None:
        cv2.line(image, tuple(l5_left), tuple(s1_left), (255, 0, 0), 2)
    slope1 = (s1_left[1] - l5_left[1]) / (s1_left[0] - l5_left[0] + 1e-7)
    
    # Pair right with right
    if image is not None:
        cv2.line(image, tuple(l5_right), tuple(s1_right), (0, 255, 0), 2)
    slope2 = (s1_right[1] - l5_right[1]) / (s1_right[0] - l5_right[0] + 1e-7)
    
    average_slope = (slope1 + slope2) / 2
    return average_slope


def transform_pairs(pairs_dict, value=0):
    result = {}
    
    for key, point_pairs in pairs_dict.items():
        pair1, pair2 = point_pairs
        
        # Calculate average x for each pair
        avg_x1 = (pair1[0][0] + pair1[1][0]) / 2
        avg_x2 = (pair2[0][0] + pair2[1][0]) / 2
        
        if value > 0:
            # Pair with more x values gets "Left Points"
            if avg_x1 < avg_x2:
                result[key] = {
                    "Abdominal Points": pair1,
                    "Back Points": pair2
                }
            else:
                result[key] = {
                    "Abdominal Points": pair2,
                    "Back Points": pair1
                }
        else:
            # Opposite: pair with less x values gets "Right Points"
            if avg_x1 < avg_x2:
                result[key] = {
                    "Back Points": pair1,
                    "Abdominal Points": pair2
                }
            else:
                result[key] = {
                    "Back Points": pair2,
                    "Abdominal Points": pair1
                }
    
    return result


def calculate_slip_distance(data):
    """
    Calculate spondylolisthesis slip distance.
    
    Args:
        data: Dictionary with 'Abdominal Points' and 'Back Points'
              Each contains ((upper_x, upper_y), (lower_x, lower_y))
    
    Returns:
        dict: Contains slip_distance, ap_length, slip_percentage, and intersection_point
    """
    # Extract points
    abdominal_points = data['Abdominal Points']
    back_points = data['Back Points']
    
    # S1 (lower vertebra) superior endplate points
    s1_anterior = np.array(abdominal_points[1])  # (305, 631)
    s1_posterior = np.array(back_points[1])      # (386, 662)
    
    # L5 (upper vertebra) posterior corner
    l5_posterior = np.array(back_points[0])      # (419, 612)
    
    # Create infinite line through S1 superior endplate
    # Line direction vector
    s1_direction = s1_anterior - s1_posterior
    
    # Find perpendicular projection of L5 posterior onto S1 line
    # Using vector projection formula
    # Point on line closest to l5_posterior
    t = np.dot(l5_posterior - s1_posterior, s1_direction) / np.dot(s1_direction, s1_direction)
    intersection_point = s1_posterior + t * s1_direction
    
    # Calculate slip distance (from S1 posterior to intersection point)
    slip_distance = np.linalg.norm(intersection_point - s1_posterior)
    
    # Calculate AP length (full length of S1 superior surface)
    ap_length = np.linalg.norm(s1_anterior - s1_posterior)
    
    # Calculate slip percentage
    slip_percentage = (slip_distance / ap_length) * 100
    
    return {
        'slip_distance': slip_distance,
        'ap_length': ap_length,
        'slip_percentage': slip_percentage,
        'intersection_point': tuple(intersection_point),
        'l5_posterior': tuple(l5_posterior),
        's1_posterior': tuple(s1_posterior),
        's1_anterior': tuple(s1_anterior)
    }


def smooth_mask(mask):
    
    smooth_mask = cv2.GaussianBlur(mask, (5, 21), 0) 
    _, smooth_mask = cv2.threshold(smooth_mask, 220, 255, cv2.THRESH_BINARY)

    return smooth_mask


# Reading the image and model
# image = cv2.imread("/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/sam_data3/train/masks/73850-2-_jpg.rf.6ec76acfb38ce2d3607cf966a1ebe1f5.png")
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Gray
model = YOLO("/home/yousof/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts/runs/detect/train3/weights/best.pt")
results = model(image, verbose=False)

# Smooth the mask
binary_image = smooth_mask(image_grey)
# binary_image = image_grey
# Class names mapping
vertebra_classes = {
    0: 'l1',
    1: 'l2',
    2: 'l3',
    3: 'l4',
    4: 'l5',
    5: 's1'
}

Image_To_Vis = copy.deepcopy(image)

# Dictionary to store vertebra data
vertebra_data = {}

# Process all detected vertebrae
for box in results[0].boxes:
    cls_id = int(box.cls)
    
    if cls_id in vertebra_classes:
        vertebra_name = vertebra_classes[cls_id]
        
        # Get bounding box coordinates
        box_coords = list(map(int, box.xyxy[0].cpu().numpy()))
        
        # Extract mask region
        mask = binary_image[box_coords[1]:box_coords[3], 
                           box_coords[0]:box_coords[2]]
        
        # Initialize vertebra data entry
        vertebra_data[vertebra_name] = {
            'box_coordinates': box_coords,
            'corners': None,
            'corners_absolute': None  # Corners in original image coordinates
        }
        
        # Detect corners
        corners = detect_corners_convex_hull(mask)
        if corners is not None:                
            # Store relative corners (within the bounding box)
            vertebra_data[vertebra_name]['corners'] = corners
            
            # Convert to absolute coordinates (in original image)
            corners_absolute = corners.copy()
            corners_absolute[:, 0] += box_coords[0]  # Add x offset
            corners_absolute[:, 1] += box_coords[1]  # Add y offset
            vertebra_data[vertebra_name]['corners_absolute'] = corners_absolute
            


# Image Alignment and transform data
aligned_img, vertebra_data_transformed = align_spine_and_get_transform(image, vertebra_data) # Takes BGR image Returns aligned grayscale image
aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_GRAY2BGR)

aligned_img_to_vis = copy.deepcopy(aligned_img)

if Visualize_Lines:
    # Visualize Lines
    for vertebra_data_key in vertebra_data_transformed.keys():

        box_coords = vertebra_data_transformed[vertebra_data_key]['box_coordinates']
        corners = vertebra_data_transformed[vertebra_data_key]['corners']
        mask = binary_image[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]]

        # Visualize corners
        cropped = aligned_img_to_vis[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2]]
        mask_vis = visualize_corners(mask, corners, f"{vertebra_data_key.upper()}", image=cropped)
        aligned_img_to_vis[box_coords[1]:box_coords[3],
                        box_coords[0]:box_coords[2]] = mask_vis

if Visualize_Corners:
    # Draw the boxes of vertebrae on the aligned image for visualization
    for key, value in vertebra_data_transformed.items():
        box = value['box_coordinates']
        # Draw corners absolute
        for corner in value['corners_absolute']:
            cv2.circle(aligned_img_to_vis, tuple(corner), 5, (255, 0, 0), -1)  # Blue corners
        

# Transform the coordinates to the aligned space
Pairs = {('l1', 'l2'):[], ('l2', 'l3'):[], ('l3', 'l4'):[], ('l4', 'l5'):[], ('l5', 's1'):[]}

# Track which corners were used from each vertebra
previous_used_corners = {}

for (v1, v2) in Pairs.keys():
    if v1 in vertebra_data_transformed and v2 in vertebra_data_transformed:
        corners1 = vertebra_data_transformed[v1]['corners_absolute']
        corners2 = vertebra_data_transformed[v2]['corners_absolute']
        
        if corners1 is not None and corners2 is not None:
            # Get excluded corners
            excluded_v1 = previous_used_corners.get(v1, set())
            excluded_v2 = previous_used_corners.get(v2, set())
                        
            closest_pairs, used_corners1, used_corners2 = find_corner_pairs(
                corners1, corners2, excluded_v1, excluded_v2
            )
            Pairs[(v1, v2)] = closest_pairs
            
            # Store which corners were used from both vertebrae
            previous_used_corners[v1] = used_corners1
            previous_used_corners[v2] = used_corners2
                        
"""
Pairs structure example:
{
('l1', 'l2'): [((l1_x1, l1_y1), (l2_x1, l2_y1)), ((l1_x2, l1_y2), (l2_x2, l2_y2))],
}
"""

# Calculate the slop between the l5 and s1 vertebrae bottom surfaces to know the abdominal and back side.

# Find the two points that have the longest distance between them
# 1. Get the l5 bottom corners
l5_bottom_1, l5_bottom_2 = Pairs[('l5', 's1')][0][0], Pairs[('l5', 's1')][1][0]
# 2. Get the s1 bottom corners, The points that are not in Pairs[('l5', 's1')]
s1_bottom_1, s1_bottom_2 = None, None
all_s1_corners = vertebra_data_transformed['s1']['corners_absolute']

for corner in all_s1_corners:
    if not (np.array_equal(corner, Pairs[('l5', 's1')][0][1]) or np.array_equal(corner, Pairs[('l5', 's1')][1][1])):
        if s1_bottom_1 is None:
            s1_bottom_1 = corner
        else:
            s1_bottom_2 = corner

# 3. Calculate slope between the two bottom points of l5 and s1
slope = calculate_slope((l5_bottom_1, l5_bottom_2), (s1_bottom_1, s1_bottom_2), aligned_img)
print(f"Slope between L5 and S1 bottom surfaces: {slope}")

# Transform the dictionary
new_pairs = transform_pairs(Pairs, slope)


for key, value in new_pairs.items(): 
    print(f"Calculating slip distance for vertebra pair: {key}")   
    result = calculate_slip_distance(value)
    # print(f"Slip Distance: {result['slip_distance']:.2f} pixels")
    # print(f"AP Length: {result['ap_length']:.2f} pixels")
    print(f"Slip Percentage: {result['slip_percentage']:.2f}%")
    # Visualize intersection point and lines
    cv2.circle(aligned_img_to_vis, tuple(map(int, result['intersection_point'])), 3, (255, 255, 0), -1)  # Cyan
    

# Draw the pairs on the image
for (v1, v2), corner_pairs in Pairs.items():
    for (corner1, corner2) in corner_pairs:
        cv2.line(aligned_img_to_vis, corner1, corner2, (255, 0, 255), 2)  # Magenta lines

# Show final visualization
cv2.imshow("Aligned Image", aligned_img_to_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()