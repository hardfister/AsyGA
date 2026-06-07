import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_base = current_script_dir

specs = {
    'RED': 'red', 
    'Green': 'green', 
    'BLUE': 'blue', 
    'NIR': 'nir'
}


os.makedirs(output_base, exist_ok=True)

for s_name_key, s_folder_value in specs.items():
    train_lines = []
    test_lines = []
    
    # ( Green -> GREEN)
    s_name_upper = s_name_key.upper()
    
    #  (ID  1 to 100)
    for i in range(1, 101):
        person_id_label = i - 1  
        for img_idx in range(1, 7):
            # 0011.jpg to 1006.jpg
            file_name = f"{i:03d}{img_idx}.jpg"
            # datacasia/red/0011.jpg
            img_path = f"datacasia/{s_folder_value}/{file_name}"
            
            line = f"{img_path} {person_id_label}\n"
            train_lines.append(line)
            
    train_file = os.path.join(output_base, f'train_{s_name_upper}.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)


