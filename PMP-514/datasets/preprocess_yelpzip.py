"""
yelpzip_2014_eligible.csv → DGL heterograph → datasets/yelpzip* (binary)

Features:
  24d (default) : hand-crafted only
  56d (--emb)   : hand-crafted (24) + text embedding PCA (32)

  - User (6): review_count, avg_rating, rating_std, unique_prod_count,
              same_day_review_count, review_day_gap
  - Product (4): review_count, avg_rating, rating_std, unique_user_count
  - Rating deviation (3): from_prod, from_user, is_extreme
  - Text stat (7): char_len, word_count, unique_word_ratio,
                   exclamation_count, question_count, uppercase_ratio, digit_ratio
  - Temporal (4): year, month, dayofweek, prod_same_day_review_count
  - Text embedding (32, --emb only): SentenceTransformer('all-MiniLM-L6-v2') -> StandardScaler -> PCA(32)

실행:
    python datasets/preprocess_yelpzip.py          # 24d → datasets/yelpzip
    python datasets/preprocess_yelpzip.py --emb    # 40d → datasets/yelpzip_emb
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

CSV_PATH  = 'datasets/final_B_bridge_sample_with_text.csv'
SEED      = 2

HAND_COLS = [
    'user_review_count', 'user_avg_rating', 'user_rating_std',
    'user_unique_prod_count', 'user_same_day_review_count', 'user_review_day_gap',
    'is_user_first_review',
    'prod_review_count', 'prod_avg_rating', 'prod_rating_std', 'prod_unique_user_count',
    'rating_deviation_from_prod', 'rating_deviation_from_user', 'rating_is_extreme',
    'review_char_len', 'review_word_count', 'unique_word_ratio',
    'exclamation_count', 'question_count', 'uppercase_ratio', 'digit_ratio',
    'review_month', 'review_dayofweek', 'prod_same_day_review_count',
]


def make_edges_all_pairs(col):
    srcs, dsts = [], []
    for _, grp in col.groupby(col):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        i, j = np.meshgrid(idx, idx)
        mask = i != j
        srcs.append(i[mask])
        dsts.append(j[mask])
    if not srcs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(srcs).astype(np.int64), np.concatenate(dsts).astype(np.int64)


def make_edges_sampled(col, k=50, seed=42):
    rng = np.random.default_rng(seed)
    srcs, dsts = [], []
    for _, grp in col.groupby(col):
        idx = grp.index.to_numpy()
        n = len(idx)
        if n < 2:
            continue
        k_ = min(k, n - 1)
        for i in range(n):
            cands = rng.choice(n - 1, k_, replace=False)
            cands[cands >= i] += 1
            srcs.append(np.full(k_, idx[i], dtype=np.int64))
            dsts.append(idx[cands])
    if not srcs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(srcs), np.concatenate(dsts)



def make_edges_tfidf(texts, max_features=30000, threshold=0.4, batch_size=512):
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    print("    Vectorizing...", flush=True)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                          min_df=3, max_features=max_features, dtype=np.float32)
    X = vec.fit_transform(texts)
    X = normalize(X, norm='l2', copy=False).astype(np.float32).tocsr()
    n = X.shape[0]
    print(f"    TF-IDF shape={X.shape}, nnz={X.nnz:,}", flush=True)

    # sparse CSR → GPU (~400MB). dense 변환 없음
    X_gpu = cpsp.csr_matrix(X)

    all_srcs, all_dsts = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_len = end - start

        if start % (batch_size * 20) == 0:
            print(f"    GPU sparse batch {start}/{n}", flush=True)

        # 배치만 dense로 GPU에 올림: (batch, 30000) ≈ 61MB
        batch_dense = cp.array(X[start:end].toarray(), dtype=cp.float32)

        # SpMM: X_gpu(n,30000) @ batch_dense.T(30000,batch) → (n,batch) → T → (batch,n)
        # spgemm 대신 SpMM(sparse×dense) 사용 → 버퍼 폭발 없음
        sim = X_gpu.dot(batch_dense.T).T  # (batch, n) dense on GPU ≈ 90MB

        # self-loop 제거
        sim[cp.arange(batch_len), cp.array(range(start, end))] = 0.0

        rows, cols = cp.where(sim >= threshold)

        all_srcs.append((cp.asnumpy(rows) + start).astype(np.int64))
        all_dsts.append(cp.asnumpy(cols).astype(np.int64))

        del batch_dense, sim, rows, cols
        cp.get_default_memory_pool().free_all_blocks()

    if not all_srcs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(all_srcs), np.concatenate(all_dsts)

def build_features(df, use_emb=False):
    df = df.copy()
    df['text'] = df['text'].fillna('')
    df['date'] = pd.to_datetime(df['date'])

    # ── User-level ──────────────────────────────────────────────────────────
    user_stats = df.groupby('user_id').agg(
        user_review_count=('rating', 'count'),
        user_avg_rating=('rating', 'mean'),
        user_rating_std=('rating', 'std'),
        user_unique_prod_count=('prod_id', 'nunique'),
    ).fillna(0)
    df = df.join(user_stats, on='user_id')

    df['user_same_day_review_count'] = (
        df.groupby(['user_id', 'date'])['rating'].transform('count')
    )

    df_s = df.sort_values(['user_id', 'date']).copy()
    df_s['_prev_date'] = df_s.groupby('user_id')['date'].shift(1)
    df_s['_raw_gap']   = (df_s['date'] - df_s['_prev_date']).dt.days
    df_s['is_user_first_review'] = df_s['_prev_date'].isna().astype(np.float32)
    df_s['user_review_day_gap']  = df_s['_raw_gap'].fillna(0.0)
    df = df_s.sort_index()

    # ── Product-level ────────────────────────────────────────────────────────
    prod_stats = df.groupby('prod_id').agg(
        prod_review_count=('rating', 'count'),
        prod_avg_rating=('rating', 'mean'),
        prod_rating_std=('rating', 'std'),
        prod_unique_user_count=('user_id', 'nunique'),
    ).fillna(0)
    df = df.join(prod_stats, on='prod_id')

    # ── Rating deviation ─────────────────────────────────────────────────────
    df['rating_deviation_from_prod'] = (df['rating'] - df['prod_avg_rating']).abs()
    df['rating_deviation_from_user'] = (df['rating'] - df['user_avg_rating']).abs()
    df['rating_is_extreme'] = df['rating'].isin([1.0, 5.0]).astype(np.float32)

    # ── Text statistics ───────────────────────────────────────────────────────
    t = df['text']
    df['review_char_len']   = t.str.len()
    df['review_word_count'] = t.str.split().str.len()
    df['unique_word_ratio'] = t.apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1)
    )
    df['exclamation_count'] = t.str.count('!')
    df['question_count']    = t.str.count(r'\?')
    df['uppercase_ratio']   = t.apply(
        lambda x: sum(c.isupper() for c in x) / max(len(x), 1)
    )
    df['digit_ratio']       = t.apply(
        lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
    )

    # ── Temporal ─────────────────────────────────────────────────────────────
    df['review_month']     = df['date'].dt.month
    df['review_dayofweek'] = df['date'].dt.dayofweek
    df['prod_same_day_review_count'] = (
        df.groupby(['prod_id', 'date'])['rating'].transform('count')
    )

    feats = df[HAND_COLS].values.astype(np.float32)
    feats = StandardScaler().fit_transform(feats).astype(np.float32)

    if use_emb:
        texts = df['text'].fillna('').tolist()
        model_st = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model_st.encode(texts, batch_size=256, show_progress_bar=True,
                              convert_to_numpy=True)
        emb = StandardScaler().fit_transform(emb).astype(np.float32)
        emb = PCA(n_components=32, random_state=SEED).fit_transform(emb).astype(np.float32)
        feats = np.concatenate([feats, emb], axis=1)

    print(f"  feature dim: {feats.shape[1]}")
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', action='store_true', help='append 16d text embedding (40d total)')
    args = parser.parse_args()

    save_path = 'datasets/bridge_emb' if args.emb else 'datasets/bridge'

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index(drop=True)
    n  = len(df)
    print(f"  Rows: {n:,}  |  Fraud: {(df['label']==1).sum():,}  |  Benign: {(df['label']==0).sum():,}")

    labels = df['label'].astype(np.int64).values

    print("Building features...")
    feats = build_features(df, use_emb=args.emb)

    print("Building edges...")
    # E1 UPU: 같은 유저가 쓴 리뷰끼리
    upu_src, upu_dst = make_edges_all_pairs(df['user_id'])
    print(f"  net_upu (same user)              : {len(upu_src):>10,} edges")

    # E2 USU: 같은 상품 + 같은 별점
    usu_key = df['prod_id'].astype(str) + '_' + df['rating'].astype(str)
    usu_src, usu_dst = make_edges_all_pairs(usu_key)
    print(f"  net_usu (same product+star)      : {len(usu_src):>10,} edges")

    # E3 UVU: 같은 상품 + 같은 월
    df['_month'] = pd.to_datetime(df['date']).dt.month
    uvu_key = df['prod_id'].astype(str) + '_' + df['_month'].astype(str)
    uvu_src, uvu_dst = make_edges_all_pairs(uvu_key)
    print(f"  net_uvu (same product+month)     : {len(uvu_src):>10,} edges")

    # E4 UTU: 텍스트 유사도 기반
    print('Building text-similarity edges (TF-IDF)...')
    utu_src, utu_dst = make_edges_tfidf(df['text'].fillna('').tolist())
    print(f"  net_utu (TF-IDF thr= 0.4)        : {len(utu_src):>10,} edges")

    print("Building DGL heterograph...")
    graph = dgl.heterograph({
        ('review', 'net_upu', 'review'): (upu_src, upu_dst),
        ('review', 'net_usu', 'review'): (usu_src, usu_dst),
        ('review', 'net_uvu', 'review'): (uvu_src, uvu_dst),
        ('review', 'net_utu', 'review'): (utu_src, utu_dst),
    }, num_nodes_dict={'review': n})

    graph.ndata['feature'] = torch.tensor(feats)
    graph.ndata['label']   = torch.tensor(labels)

    print("Splitting dataset (60/20/20 stratified)...")
    idx = np.arange(n)
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx, labels, stratify=labels, train_size=0.6, random_state=SEED, shuffle=True)
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest, y_rest, stratify=y_rest, test_size=0.5, random_state=SEED, shuffle=True)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val]     = True
    test_mask[idx_test]   = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask']   = val_mask
    graph.ndata['test_mask']  = test_mask

    fraud_labels = torch.tensor(labels)
    print(f"  Train (fraud/total): {fraud_labels[train_mask].sum():>5} / {train_mask.sum():<6}")
    print(f"  Val   (fraud/total): {fraud_labels[val_mask].sum():>5} / {val_mask.sum():<6}")
    print(f"  Test  (fraud/total): {fraud_labels[test_mask].sum():>5} / {test_mask.sum():<6}")

    dgl.save_graphs(save_path, [graph])
    print(f"\nSaved to '{save_path}'")
    print("Done.")


if __name__ == '__main__':
    main()
