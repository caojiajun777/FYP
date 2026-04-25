import json
import os

silver_file = 'task2_function_prediction/v3_silver_labels/large_silver_labels_v3.json'
gold_file = 'task2_function_prediction/gold_standard/gold_val_test.json'
out_dir = 'task2_function_prediction/train_sets'
os.makedirs(out_dir, exist_ok=True)

with open(silver_file, 'r', encoding='utf-8') as f:
    silver_data = json.load(f)

with open(gold_file, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

gold_filenames = set(item['filename'] for item in gold_data)

# Filter out gold set, and ensure it's not 'unknown'
train_pool = []
for item in silver_data:
    if item['filename'] in gold_filenames:
        continue
    labels = item.get('gold_labels', [])
    if 'unknown' in labels or not labels:
        continue
    
    # Optional strict constraint: keeping only the 5 valid classes
    valid_classes = {'power', 'interface', 'communication', 'signal', 'control'}
    filtered_labels = [l for l in labels if l in valid_classes]
    
    if filtered_labels:
        # copy item to avoid mutating original
        new_item = item.copy()
        new_item['gold_labels'] = filtered_labels
        train_pool.append(new_item)

# 1. High Only Set
train_high = [item for item in train_pool if item.get('confidence') == 'high']

# 2. High + Medium Set
train_high_med = [item for item in train_pool if item.get('confidence') in ['high', 'medium']]

print(f"Total training pool (excluding gold test+val): {len(train_pool)}")
print(f"Length of Train High-Only: {len(train_high)}")
print(f"Length of Train High+Medium: {len(train_high_med)}")

with open(f'{out_dir}/train_high.json', 'w', encoding='utf-8') as f:
    json.dump(train_high, f, indent=2, ensure_ascii=False)
with open(f'{out_dir}/train_high_medium.json', 'w', encoding='utf-8') as f:
    json.dump(train_high_med, f, indent=2, ensure_ascii=False)
