
import json
from pathlib import Path

# Path relative to paper_results
json_path = Path('task1_source_classification/analysis/data_quality_verification.json')

if json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    internal = data.get('internal_duplicates', {})
    cross = data.get('cross_split_leaks', {})
    
    internal_pairs = sum(len(v) for v in internal.values())
    
    cross_pairs = 0
    for k, v in cross.items():
        cross_pairs += len(v)
        
    print(f"Internal Pairs: {internal_pairs}")
    print(f"Cross Pairs: {cross_pairs}")
    print(f"Total Candidates: {internal_pairs + cross_pairs}")
    print(f"Unique Internal Keys: {len(internal)}")

else:
    print(f"JSON not found at {json_path.absolute()}")
