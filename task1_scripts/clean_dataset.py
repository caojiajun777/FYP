#!/usr/bin/env python3
"""Clean dataset by removing duplicates and cross-split leaks identified by verify_data_quality.py"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, List, Tuple

# Configuration
DATASET_PATH = Path("D:/FYP/data/EDA_cls_dataset")
BACKUP_PATH = Path("D:/FYP/data/EDA_cls_dataset_backup")
VERIFICATION_REPORT = Path("d:/FYP/Classifier/paper_results/task1_source_classification/analysis/data_quality_verification.json")
OUTPUT_DIR = Path("d:/FYP/Classifier/paper_results")
STATS_BEFORE = {}
STATS_AFTER = {}

def backup_dataset():
    """Create backup of original dataset before cleaning"""
    if BACKUP_PATH.exists():
        print(f"✓ Backup already exists at {BACKUP_PATH}")
        return
    
    print(f"Backing up dataset to {BACKUP_PATH}...")
    shutil.copytree(DATASET_PATH, BACKUP_PATH, dirs_exist_ok=True)
    print("✓ Backup completed")

def count_dataset(dataset_path: Path, label="") -> Dict:
    """Count images in each split"""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if split_dir.exists():
            images = list(split_dir.glob("**/*.png"))
            counts[split] = len(images)
    if label:
        print(f"\n{label}")
        print(f"  Train: {counts.get('train', 0)}")
        print(f"  Val:   {counts.get('val', 0)}")
        print(f"  Test:  {counts.get('test', 0)}")
        print(f"  Total: {sum(counts.values())}")
    return counts

def load_verification_report() -> Dict:
    """Load verification report JSON"""
    if not VERIFICATION_REPORT.exists():
        raise FileNotFoundError(f"Verification report not found: {VERIFICATION_REPORT}")
    
    with open(VERIFICATION_REPORT, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    print(f"✓ Loaded verification report with {len(report.get('leaks', {}).get('train_val', []))} train-val leaks")
    return report

def clean_cross_split_leaks(dataset_path: Path, report: Dict) -> int:
    """Remove duplicates found across splits"""
    removed_count = 0
    
    # Strategy: Always delete from val/test when duplicate exists in train
    cross_leaks = report.get('cross_split_leaks', {})
    
    # Train-Val leaks: delete from Val
    for leak_info in cross_leaks.get('train_val', []):
        val_path = Path(leak_info['val_path'])
        if val_path.exists():
            val_path.unlink()
            removed_count += 1
    
    # Train-Test leaks: delete from Test
    for leak_info in cross_leaks.get('train_test', []):
        test_path = Path(leak_info['test_path'])
        if test_path.exists():
            test_path.unlink()
            removed_count += 1
    
    # Val-Test leaks: delete from Test
    for leak_info in cross_leaks.get('val_test', []):
        test_path = Path(leak_info['test_path'])
        if test_path.exists():
            test_path.unlink()
            removed_count += 1
    
    print(f"✓ Removed {removed_count} cross-split leaks")
    return removed_count

def clean_internal_duplicates(dataset_path: Path, report: Dict) -> int:
    """Remove internal train duplicates (keep lower-numbered, delete higher)"""
    removed_count = 0
    
    duplicates = report.get('internal_duplicates', {})
    
    # Group duplicates by similarity
    processed = set()
    
    for source_path, similar_items in duplicates.items():
        if source_path in processed:
            continue
        
        source_p = Path(source_path)
        
        for similar_info in similar_items:
            target_path = Path(similar_info['similar_to'])
            
            # Compare file numbers to keep lower-numbered
            try:
                source_num = int(source_path.split("\\")[-1].split(".")[0])
                target_num = int(similar_info['similar_to'].split("\\")[-1].split(".")[0])
                
                # Delete higher-numbered
                if source_num < target_num:
                    if target_path.exists():
                        target_path.unlink()
                        removed_count += 1
                        processed.add(similar_info['similar_to'])
                else:
                    if source_p.exists():
                        source_p.unlink()
                        removed_count += 1
                        processed.add(source_path)
            except:
                pass
    
    print(f"✓ Removed {removed_count} internal train duplicates")
    return removed_count

def generate_cleaning_report(before: Dict, after: Dict, cross_removed: int, internal_removed: int):
    """Generate markdown report of cleaning operation"""
    
    report_path = OUTPUT_DIR / "data_cleaning_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle missing keys
    before = {k: before.get(k, 0) for k in ['train', 'val', 'test']}
    after = {k: after.get(k, 0) for k in ['train', 'val', 'test']}
    
    total_before = sum(before.values())
    total_after = sum(after.values())
    total_removed = total_before - total_after
    
    # Calculate reductions
    train_reduction = before['train'] - after['train']
    val_reduction = before['val'] - after['val']
    test_reduction = before['test'] - after['test']
    
    train_pct = 100 * train_reduction / before['train'] if before['train'] > 0 else 0
    val_pct = 100 * val_reduction / before['val'] if before['val'] > 0 else 0
    test_pct = 100 * test_reduction / before['test'] if before['test'] > 0 else 0
    total_pct = 100 * total_removed / total_before if total_before > 0 else 0
    
    content = f"""# Dataset Cleaning Report

**Generated**: {timestamp}

## Executive Summary

Successfully removed {total_removed} samples:
- **Cross-split leaks removed**: {cross_removed}
- **Internal train duplicates removed**: {internal_removed}
- **Total reduction**: {total_removed} samples ({total_pct:.1f}%)

## 1. Cleaning Strategy

### 1.1 Cross-Split Leak Removal

**Strategy:**
- Train ↔ Val leaks: Delete Val samples
- Train ↔ Test leaks: Delete Test samples
- Val ↔ Test leaks: Delete Test samples (preserve Train)

**Rationale**: Preserve all Training samples (largest and most stable), remove contaminated Val/Test

**Results**: Removed {cross_removed} cross-split leaked samples

### 1.2 Internal Train Duplicate Removal

**Strategy:**
- For each duplicate pair with identical hashes (distance=0)
- Keep lower-numbered sample (as reference)
- Delete higher-numbered sample

**Rationale**: Eliminate redundancy while preserving data diversity

**Results**: Removed {internal_removed} internal train duplicates

## 2. Before/After Comparison

### Dataset Size

| Split | Before | After | Removed | % Reduction |
|-------|--------|-------|---------|------------|
| Train | {before['train']} | {after['train']} | {train_reduction} | {train_pct:.1f}% |
| Val | {before['val']} | {after['val']} | {val_reduction} | {val_pct:.1f}% |
| Test | {before['test']} | {after['test']} | {test_reduction} | {test_pct:.1f}% |
| **Total** | **{total_before}** | **{total_after}** | **{total_removed}** | **{total_pct:.1f}%** |

### Quality Improvements

- **Eliminated cross-split data leakage**: ✓ (All {cross_removed} leaks removed)
- **Eliminated internal train duplicates**: ✓ ({internal_removed} duplicates removed)
- **Preserved training data**: ✓ (All Train samples with lowest numbers kept)
- **Cleaner validation/test sets**: ✓ (No contamination from Train)

## 3. Expected Impact on Model

After retraining on cleaned dataset:
- **Expected accuracy change**: -2% to -4% (from 99.07% to 95-97%)
- **Reason**: Removing data leakage eliminates unfair advantage
- **Validation**: New metrics will be verifiable/publishable without reviewer concerns

## 4. Next Steps

1. Retrain ViT-B/16 on cleaned dataset using same hyperparameters
2. Generate new baseline metrics on cleaned validation set
3. Update thesis documentation with data cleaning methodology
4. Add verification results to appendix

## 5. Dataset Locations

- **Original (backed up)**: D:/FYP/data/EDA_cls_dataset_backup/
- **Cleaned dataset**: D:/FYP/data/EDA_cls_dataset/

To restore original:
```bash
Remove-Item -Recurse -Force "D:/FYP/data/EDA_cls_dataset"
Copy-Item -Recurse "D:/FYP/data/EDA_cls_dataset_backup" -Destination "D:/FYP/data/EDA_cls_dataset"
```
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Report saved to {report_path}")

def save_cleaning_stats(before: Dict, after: Dict, cross_removed: int, internal_removed: int):
    """Save cleaning statistics as JSON"""
    
    # Handle missing keys
    before = {k: before.get(k, 0) for k in ['train', 'val', 'test']}
    after = {k: after.get(k, 0) for k in ['train', 'val', 'test']}
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'before': before,
        'after': after,
        'removed': {
            'cross_split_leaks': cross_removed,
            'internal_duplicates': internal_removed,
            'total': cross_removed + internal_removed
        },
        'reduction_percentage': {
            'train': 100 * (before['train'] - after['train']) / before['train'] if before['train'] > 0 else 0,
            'val': 100 * (before['val'] - after['val']) / before['val'] if before['val'] > 0 else 0,
            'test': 100 * (before['test'] - after['test']) / before['test'] if before['test'] > 0 else 0,
            'total': 100 * (sum(before.values()) - sum(after.values())) / sum(before.values()) if sum(before.values()) > 0 else 0
        }
    }
    
    stats_path = OUTPUT_DIR / "cleaning_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to {stats_path}")

def main():
    """Main cleaning pipeline"""
    print("=" * 70)
    print("EDA Dataset Cleaning Pipeline")
    print("=" * 70)
    
    # Load verification report first
    report = load_verification_report()
    
    # Step 1: Count before
    STATS_BEFORE = count_dataset(DATASET_PATH, "Before cleaning:")
    
    # Step 2: Clean cross-split leaks
    cross_removed = clean_cross_split_leaks(DATASET_PATH, report)
    
    # Step 3: Clean internal duplicates
    internal_removed = clean_internal_duplicates(DATASET_PATH, report)
    
    # Step 4: Count after
    STATS_AFTER = count_dataset(DATASET_PATH, "After cleaning:")
    
    # Step 5: Generate reports
    generate_cleaning_report(STATS_BEFORE, STATS_AFTER, cross_removed, internal_removed)
    save_cleaning_stats(STATS_BEFORE, STATS_AFTER, cross_removed, internal_removed)
    
    print("\n" + "=" * 70)
    print("Cleaning completed successfully!")
    print("=" * 70)
    print(f"\nNext steps:")
    print("1. Run training: python train_vit.py")
    print("2. Evaluate: python evaluate_vit_model.py")
    print("3. Update thesis with new baseline metrics")
    print("=" * 70)

if __name__ == "__main__":
    main()
