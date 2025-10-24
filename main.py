# main.py — CIFAR-10 KNN on TRAIN(50k) only
# 1) train/test split
# 2) train/val/test split
# 3) 5-fold cross-validation

import os, glob, warnings
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# -----------------------------
# Config 
# -----------------------------
class CFG:
    # 경로 (Windows에서도 안전하게 raw string 사용)
    TRAIN_DIR    = r"C:\Users\82106\downloads\cifar-10\train\train"   # train 이미지 폴더
    LABELS_PATH  = r"C:\Users\82106\downloads\cifar-10\trainLabels.csv"  # trainLabels.csv

    TARGET_SIZE = (24, 24)      # 빠른 실험용 (32,32로 바꿔도 됨)
    GRAY = False
    USE_PCA = True
    PCA_N = 128
    K_LIST = [1, 3, 5, 7, 9, 11, 15]
    AVERAGE = "macro"
    MAX_TRAIN_SAMPLES = None    # 개발 빠른 테스트용: 예) 20000
    OUT_DIR = "./outputs"

# -----------------------------
# I/O helpers
# -----------------------------
def read_table_auto(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl")
    
    try:
        return pd.read_csv(path)
    except:
        return pd.read_excel(path, engine="openpyxl")

def list_images_by_stem(folder: str) -> Dict[str, str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, e))
    return {Path(p).stem: p for p in paths}

def load_labels_table(labels_path: str) -> pd.DataFrame:
    df = read_table_auto(labels_path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "id" not in cols or "label" not in cols:
        raise ValueError("trainLabels 파일에는 'id'와 'label' 컬럼이 필요합니다.")
    df = df[[cols["id"], cols["label"]]].rename(columns={cols["id"]: "id", cols["label"]: "label"})
    df["id"] = df["id"].astype(str)
    return df

def read_preprocess(path: str, size: Tuple[int, int], gray: bool) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # (H,W)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    # (H,W,3)
    return im

def imgs_to_X(imgs: List[np.ndarray], gray: bool) -> np.ndarray:
    feats = [im.reshape(-1) for im in imgs]
    X = np.stack(feats, 0).astype(np.float32) / 255.0
    return X

def fit_pca(X: np.ndarray, n: int):
    ipca = IncrementalPCA(n_components=n, batch_size=2048)
    ipca.fit(X)
    return ipca, ipca.transform(X)

def metrics(y_true, y_pred, avg="macro"):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average=avg, zero_division=0),
        recall=recall_score(y_true, y_pred, average=avg, zero_division=0),
        f1=f1_score(y_true, y_pred, average=avg, zero_division=0),
    )

def plot_k(xs, means, stds, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")

# -----------------------------
# Prepare TRAIN (50k) only
# -----------------------------
def prepare_train(cfg: CFG):
    train_dir = cfg.TRAIN_DIR
    labels_path = cfg.LABELS_PATH

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train 폴더가 없습니다: {train_dir}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"trainLabels 파일이 없습니다: {labels_path}")

    print(f"[Path] TRAIN_DIR   = {train_dir}")
    print(f"[Path] LABELS_PATH = {labels_path}")

    stem2path = list_images_by_stem(train_dir)
    df = load_labels_table(labels_path)
    df["path"] = df["id"].map(stem2path)
    if df["path"].isna().any():
        miss = df[df["path"].isna()]["id"].head(10).tolist()
        raise FileNotFoundError(f"이미지 경로 매칭 실패 예시(최대 10개): {miss}")

    if cfg.MAX_TRAIN_SAMPLES:
        df = df.sample(cfg.MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)

    imgs = [read_preprocess(p, cfg.TARGET_SIZE, cfg.GRAY) for p in tqdm(df["path"], desc="Load train images")]
    X = imgs_to_X(imgs, cfg.GRAY)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str).values)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if cfg.USE_PCA:
        pca, Xf = fit_pca(Xs, cfg.PCA_N)
    else:
        pca, Xf = None, Xs

    return dict(df=df, X=Xf, y=y, le=le, scaler=scaler, pca=pca)

# -----------------------------
# 1) train/test split
# -----------------------------
def exp_train_test(X, y, cfg: CFG):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rows = []
    for k in cfg.K_LIST:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        m = metrics(yte, pred, cfg.AVERAGE)
        rows.append(dict(k=k, **m))
        print(f"[Train/Test] k={k:>2}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(cfg.OUT_DIR, "train_test_results.csv"), index=False)
    return df

# -----------------------------
# 2) train/val/test split
# -----------------------------
def exp_train_val_test(X, y, cfg: CFG):
    # 기본 70/15/15
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp)

    # k 탐색은 validation으로
    rec = []
    for k in cfg.K_LIST:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        clf.fit(Xtr, ytr)
        pva = clf.predict(Xva)
        mva = metrics(yva, pva, cfg.AVERAGE)
        rec.append((k, mva["accuracy"]))
        print(f"[Val] k={k:>2}  acc={mva['accuracy']:.4f}")

    best_k = sorted(rec, key=lambda t: t[1], reverse=True)[0][0]
    print(f"[Selected k] {best_k}")

    # 선택된 k로 train+val 재학습 후 test 평가
    clf = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
    clf.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))
    pred = clf.predict(Xte)
    m = metrics(yte, pred, cfg.AVERAGE)

    out = dict(selected_k=best_k, **{f"test_{k}": v for k, v in m.items()})
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    pd.DataFrame([out]).to_csv(os.path.join(cfg.OUT_DIR, "train_val_test_result.csv"), index=False)
    return out

# -----------------------------
# 3) 5-fold Cross-Validation
# -----------------------------
def exp_cross_val(X, y, cfg: CFG):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k2accs, k2prec, k2rec, k2f1 = {k: [] for k in cfg.K_LIST}, {k: [] for k in cfg.K_LIST}, {k: [] for k in cfg.K_LIST}, {k: [] for k in cfg.K_LIST}

    for k in cfg.K_LIST:
        print(f"\n[CV] k={k}")
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[va])
            m = metrics(y[va], p, cfg.AVERAGE)
            k2accs[k].append(m["accuracy"])
            k2prec[k].append(m["precision"])
            k2rec[k].append(m["recall"])
            k2f1[k].append(m["f1"])
            print(f"  Fold{fold}: acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    rows = []
    for k in cfg.K_LIST:
        rows.append(dict(
            k=k,
            acc_mean=np.mean(k2accs[k]), acc_std=np.std(k2accs[k]),
            prec_mean=np.mean(k2prec[k]),
            rec_mean=np.mean(k2rec[k]),
            f1_mean=np.mean(k2f1[k])
        ))
    df = pd.DataFrame(rows).sort_values("acc_mean", ascending=False)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(cfg.OUT_DIR, "cv_summary.csv"), index=False)
    plot_k(df["k"].tolist(), df["acc_mean"].values, df["acc_std"].values,
           "5-fold Accuracy vs k", os.path.join(cfg.OUT_DIR, "cv_accuracy_vs_k.png"))
    return df

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("CIFAR-10 KNN (train-only)")
    parser.add_argument("--train_dir", type=str, default=CFG.TRAIN_DIR, help="Path to train image folder")
    parser.add_argument("--labels_path", type=str, default=CFG.LABELS_PATH, help="Path to trainLabels file")
    parser.add_argument("--mode", type=str, default="all", choices=["simple", "valtest", "cv", "all"])
    args = parser.parse_args()

    # 경로 덮어쓰기 가능
    CFG.TRAIN_DIR = args.train_dir
    CFG.LABELS_PATH = args.labels_path

    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    print("== Load TRAIN(50k) & build features ==")
    assets = prepare_train(CFG)
    X, y = assets["X"], assets["y"]

    if args.mode in ["simple", "all"]:
        print("\n== 1) Train/Test split ==")
        df_tt = exp_train_test(X, y, CFG)
        print(df_tt.head())

    if args.mode in ["valtest", "all"]:
        print("\n== 2) Train/Validation/Test split ==")
        out = exp_train_val_test(X, y, CFG)
        print(out)

    if args.mode in ["cv", "all"]:
        print("\n== 3) 5-fold Cross-Validation ==")
        df_cv = exp_cross_val(X, y, CFG)
        print(df_cv.head())
