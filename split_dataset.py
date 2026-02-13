"""
Split train_labels.csv into train and validation by identity.
This prevents data leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def split_by_identity(csv_path, train_ratio=0.8, seed=42):
    """
    Split dataset by person_id (identity) to prevent leakage.
    """
    df = pd.read_csv(csv_path)
    
    # Get unique person IDs
    unique_ids = df['person_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    
    # Split IDs
    n_train = int(len(unique_ids) * train_ratio)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:]
    
    # Split dataframe
    train_df = df[df['person_id'].isin(train_ids)]
    val_df = df[df['person_id'].isin(val_ids)]
    
    print(f"Total identities: {len(unique_ids)}")
    print(f"Train identities: {len(train_ids)} ({len(train_df)} samples)")
    print(f"Val identities: {len(val_ids)} ({len(val_df)} samples)")
    
    # Save splits
    output_dir = Path(csv_path).parent
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'valid_split.csv', index=False)
    
    print(f"\nSaved to:")
    print(f"  {output_dir / 'train_split.csv'}")
    print(f"  {output_dir / 'valid_split.csv'}")

if __name__ == '__main__':
    split_by_identity(
        'data/DB_extracted/train_labels.csv',
        train_ratio=0.8,
        seed=42
    )