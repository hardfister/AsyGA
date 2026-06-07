import cv2
import numpy as np
import os

def extract_palm_roi(img_path, roi_size=128):
    """Extract palm ROI and enhance its contrast."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarization
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find the largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_cnt = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max_cnt], -1, 255, -1)

    # 3. Distance transform to find the central point
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    cx, cy = max_loc
    half_side = int(max_val * 0.8) 
    
    x1, y1 = cx - half_side, cy - half_side
    x2, y2 = cx + half_side, cy + half_side

    # 4. Crop and resize
    h, w = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi = gray[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
        
    roi_resized = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE to enhance texture features
    roi_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi_resized)

    return roi_final

# --- Path Configuration ---
input_dir = './images'      # Replace with the source directory containing images like '001_l_460_01.jpg'
output_base = '.'           # Root directory to save processed outputs

# Mapping from wavelength to folder name
folder_map = {
    '460': 'Blue',
    '630': 'Red',
    '850': 'NIR',
    '940': 'NIR_940',
    '700': 'DeepRed'
}

if not os.path.exists(output_base):
    os.makedirs(output_base)

print("Processing and renaming started...")

for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.png', '.bmp')):
        # Parse filename: e.g., 001_l_460_01.jpg
        # parts[0]='001', parts[1]='l', parts[2]='460', parts[3]='01.jpg'
        parts = file.split('_')
        if len(parts) < 4:
            continue
        
        user_id = parts[0]       # '001'
        wavelength = parts[2]    # '460'
        
        # Extract sequence number and remove file extension: '01.jpg' -> '01' -> '1'
        sample_index = parts[3].split('.')[0]
        sample_index_int = str(int(sample_index)) # Convert '01' to '1'
        
        # --- Core: Construct new filename (e.g., '001' + '1' = '0011.jpg') ---
        new_filename = f"{user_id}{sample_index_int}.jpg"
        
        # Determine target folder
        target_folder = folder_map.get(wavelength, 'Others')
        save_path = os.path.join(output_base, target_folder)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Extract and save ROI
        roi_img = extract_palm_roi(os.path.join(input_dir, file))
        
        if roi_img is not None:
            final_save_path = os.path.join(save_path, new_filename)
            cv2.imwrite(final_save_path, roi_img)
            print(f"Success: {file} -> {target_folder}/{new_filename}")
        else:
            print(f"Failed (Unable to extract ROI): {file}")

print("\nAll operations completed successfully!")