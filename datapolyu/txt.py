import os

def generate_strict_train_txt(img_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mapping = {
        'R': 'RED',
        'G': 'Green',
        'B': 'BLUE',
        'I': 'NIR'
    }

    files_data = {k: [] for k in mapping.keys()}
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    for img_name in all_images:
        parts = img_name.split('_')
        if len(parts) != 3:
            continue
            
        spectrum = parts[0]      # R / G / B / I
        try:
            person_id = int(parts[1]) # 1 to 500
            img_idx = int(parts[2].split('.')[0]) # 1 到 12
        except ValueError:
            continue
        if 1 <= person_id <= 500: 
            if 1 <= img_idx <= 12:
                if spectrum in mapping:
                    label = person_id - 1
                    line = f"datapom/po/{img_name} {label}"
                    files_data[spectrum].append(line)

    for spectrum, lines in files_data.items():
        file_name = f"train_{mapping[spectrum]}.txt"
        save_path = os.path.join(save_dir, file_name)
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
        
       
        print(f" {file_name}: {len(lines)} ")

generate_strict_train_txt(img_dir='./po', save_dir='./')