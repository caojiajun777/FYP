import json
import os
import argparse

# The order strictly matches the target output. 
# Prompt teaches the model to answer purely with comma-separated values.
CLASSES = ["power", "interface", "communication", "signal", "control"]

def build_qwen_dataset(json_input, output_json, img_base_dir):
    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    llama_factory_list = []
    
    # Instruction optimized for LoRA (classification task strictly)
    instruction = (
        "Analyze the schematic diagram and identify its functional categories. "
        "The categories must be chosen from the following list ONLY: "
        f"[{', '.join(CLASSES)}]. "
        "Output the identified categories separated by commas, with no other text."
    )
        
    for item in data:
        # Resolve image path
        win_path = item.get("image_path", "")
        filename = item.get("filename", "")
        
        # Estimate Linux path similarly
        parts = win_path.replace('\\', '/').split('/')
        if len(parts) >= 2:
            rel_path = f"{parts[-2]}/{parts[-1]}"
            linux_path = os.path.join(img_base_dir, rel_path)
        else:
            linux_path = os.path.join(img_base_dir, filename)
            
        labels = item.get("gold_labels", [])
        # Strict filter to 5 classes and sorted
        valid_labels = [l for l in labels if l in CLASSES]
        valid_labels.sort()
        
        if not valid_labels:
            continue
            
        label_str = ", ".join(valid_labels)
        
        # Standard ShareGPT + Images format for LLaMA-Factory / Qwen
        record = {
            "messages": [
                {
                    "content": f"<image>{instruction}",
                    "role": "user"
                },
                {
                    "content": label_str,
                    "role": "assistant"
                }
            ],
            "images": [
                linux_path
            ]
        }
        llama_factory_list.append(record)
        
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(llama_factory_list, f, indent=2, ensure_ascii=False)
        
    print(f"Processed {len(llama_factory_list)} samples. Output written to {output_json}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True, help="Input silver/gold json")
    parser.add_argument('--output_json', type=str, required=True, help="Output format for LoRA Train")
    parser.add_argument('--img_dir', type=str, required=True, help="Path to EDA_cls_dataset_full on cloud")
    args = parser.parse_args()
    
    build_qwen_dataset(args.input_json, args.output_json, args.img_dir)
