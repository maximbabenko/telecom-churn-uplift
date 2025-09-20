from pathlib import Path
import json
import pandas as pd

def find_binary_candidates(df: pd.DataFrame, exclude_cols=None):
    """
    Ищем все строго бинарные колонки (только 0/1, без других значений).
    Возвращаем список кортежей: (имя_колонки, доля_единиц, расстояние_до_0.5).
    Сортируем так, чтобы самые близкие к 0.5 были первыми.
    """
    if exclude_cols is None: 
        exclude_cols = []
    bins = []
    for c in df.columns:
        if c in exclude_cols: 
            continue
        s = df[c].dropna()
        # строго бинарная: каждое значение либо 0, либо 1
        if s.isin([0, 1]).all() and s.size > 0:
            p = float(s.mean())              
            dist = abs(0.5 - p)              
            bins.append((c, p, dist))
    bins.sort(key=lambda x: x[2])
    return bins

def main(name="orange_belgium", target_current="y"):
    raw_dir = Path("data/raw")
    p_parquet = raw_dir / f"{name}.parquet"
    p_csv     = raw_dir / f"{name}.csv"

    # 1) читаем сырые данные
    
    if p_parquet.exists():
        df = pd.read_parquet(p_parquet)
    else:
        df = pd.read_csv(p_csv)

    # 2) y -> churn
    
    assert target_current in df.columns, (
        f"Не нашёл таргет-колонку '{target_current}'. "
        f"Первые колонки: {list(df.columns)[:20]}"
    )
    if "churn" not in df.columns:
        df = df.rename(columns={target_current: "churn"})
    df["churn"] = df["churn"].astype(int)  # на всякий: приводим к 0/1 int

    # 3) ищем treatment среди бинарных (кроме 'churn')
    bin_cands = find_binary_candidates(df, exclude_cols=["churn"])

    print("[info] TOP-5 бинарных кандидатов (ближе к 0.5 — вероятнее treatment):")
    top5 = []
    for c, p, d in bin_cands[:5]:
        print(f"  {c}: p(1)={p:.3f}, |p-0.5|={d:.3f}")
        top5.append({"col": c, "p_ones": p, "dist_to_0_5": d})

    # правило: берём самый близкий к 0.5
    tcol = None
    if bin_cands:
        best_col, p1, dist = bin_cands[0]
        # если рассылка была не 99/1, то p1 обычно в коридоре [0.2..0.8]
        if 0.2 <= p1 <= 0.8:
            tcol = best_col

    assert tcol is not None, (
        "Не удалось надёжно определить колонку treatment автоматически.\n"
        "Посмотри TOP-5 кандидатов выше и зафиксируй вручную.\n"
    )

    if tcol != "treatment":
        df = df.rename(columns={tcol: "treatment"})
    df["treatment"] = df["treatment"].astype(int)

    # 4) сохраняем стандартизованные версии
    out_csv = raw_dir / f"{name}_std.csv"
    out_pq  = raw_dir / f"{name}_std.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_pq, index=False)
    except Exception:
        pass

    # 5) сохраняем meta‑лог: что выбрали, какие кандидаты были, форма
    meta = {
        "chosen_treatment_col": tcol,
        "p_ones_chosen": float(df["treatment"].mean()),
        "top5_binary_candidates": top5,
        "shape": list(df.shape),
        "columns_preview": list(df.columns)[:20],
    }
    meta_path = raw_dir / f"{name}_std_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[ok] standardized saved:")
    print(" ", out_csv)
    if out_pq.exists(): 
        print(" ", out_pq)
    print("[ok] churn/treatment head():")
    print(df[["churn", "treatment"]].head())
    print("[ok] meta path:", meta_path)

if __name__ == "__main__":
    main(name="orange_belgium", target_current="y")
