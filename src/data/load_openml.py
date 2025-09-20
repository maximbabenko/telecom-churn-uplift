import argparse              
from pathlib import Path      
import pandas as pd
import openml                 

def main(dataset_id: int, name: str):
    print(f"[OpenML] downloading dataset id={dataset_id} ...")
    ds = openml.datasets.get_dataset(dataset_id) 

    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

    df = X.copy()
    if y is not None and ds.default_target_attribute not in df.columns:
        df[ds.default_target_attribute] = y

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / f"{name}.csv"        
    pq_path  = raw_dir / f"{name}.parquet"   

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        print("[warn] parquet save failed (optional):", e)

    print("[done] saved:", csv_path)
    if pq_path.exists():
        print("[done] saved:", pq_path)
    print("shape:", df.shape)                          
    print("columns (first 15):", list(df.columns)[:15])
    print("default target:", ds.default_target_attribute)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=45580)
    ap.add_argument("--name", type=str, default="orange_belgium")
    args = ap.parse_args()

    main(args.dataset_id, args.name)