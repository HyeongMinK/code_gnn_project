# PMP-514: YelpZip 기반 가짜 리뷰 탐지를 위한 Multi-relation GNN

YelpZip 리뷰 데이터셋을 기반으로, 각 리뷰가 가짜(fake)인지 정상(real)인지를 판별하는 이진 분류 모델입니다.  
GraphSAGE 계열의 **LA-SAGE-S** 백본 위에 fraud-aware partition aggregation, Focal Loss, Pairwise Ranking Loss를 결합하였습니다.

---

## 모델링 목표

| 목표 | 내용 |
|---|---|
| 목표 1 | 24차원 hand-crafted 피처 + 4종 multi-relation graph로 가짜 리뷰 식별 |
| 목표 2 | 클래스 불균형에 적합한 PR-AUC와 Macro-F1을 기준으로 모델 평가 |
| 목표 3 | 이웃을 fraud / benign / unknown 3그룹으로 분리해 집계하여 camouflage 문제 해소 |

---

## 데이터셋: YelpZip

| 항목 | 값 |
|---|---|
| 전체 리뷰 수 | 608,458 건 |
| 전체 사용자 수 | 260,239 명 |
| 전체 상품(레스토랑) 수 | 5,044 개 |
| 수집 기간 | 2004-10-20 ~ 2015-01-10 |
| 원본 가짜 리뷰 비율 | 약 13.22% |

원본 YelpZip에서 **2014년 데이터만** 추출하여 사용합니다. 2014년은 수집 기간 내에서 리뷰 수가 가장 많고 사용자 활동이 가장 활발한 해입니다.

| 항목 | 값 |
|---|---|
| 2014년 리뷰 수 | 180,659 건 |
| 2014년 사용자 수 | 101,729 명 |
| 2014년 상품 수 | 4,529 개 |
| 2014년 가짜 리뷰 비율 | 약 12.66% |

---

## 전처리 및 샘플링

### Product-month 블록 단위 샘플링

원시 데이터를 그대로 사용하면 노드 수가 학술제 권장 상한(약 5만 건)을 초과하고, 특정 상품에 노드가 편중됩니다. 이를 해소하기 위해 `block_id = prod_id + "_" + year_month` 단위의 두 단계 샘플링을 수행합니다.

- **방법 A**: `n_reviews ≥ 20`, `n_users ≥ 5`, `단일 상품 점유율 ≤ 20%` 조건을 만족하는 블록 전체 포함 → 1,420개 블록, 코어 노드 48,071건
- **방법 B**: 방법 A 결과를 입력으로 받아 `(year_month, block_size_group)` 25개 층에 대해 구조 점수 상위 블록만 균등 선택 → 846개 블록, 코어 노드 35,418건

### User Bridge 보강

방법 B만으로는 R-U-R(Review-User-Review) coverage가 39% 수준에 머뭅니다. 코어 블록에 등장한 사용자에 한해, 동일 연도 내 코어 블록 외 리뷰를 **사용자당 최대 3건(날짜 최근순)** 추가합니다. bridge 노드 비율이 전체의 25%를 넘지 않도록 자동 조정합니다.

### 최종 샘플 통계

| 지표 | 값 |
|---|---|
| 전체 노드 수 | 46,547 건 |
| └ 코어 노드 | 35,418 건 |
| └ User Bridge 노드 | 11,129 건 (23.91%) |
| 포함 상품 수 | 2,002 개 |
| 포함 사용자 수 | 26,760 명 |
| 포함 월 수 | 12개월 (2014년 전체) |
| 가짜 리뷰 비율 | 12.87% |
| R-T-R coverage | 90.26% |
| R-U-R coverage | 69.46% (Bridge 전 39.5%) |
| R-S-R coverage | 82.63% |

---

## 입력 피처: 24차원 Hand-crafted Feature

고차원 텍스트 피처(TF-IDF 등) 대신, 행동·별점·텍스트 통계 기반의 해석 가능한 피처를 사용합니다.

| 그룹 | 피처 (개수) | 설명 |
|---|---|---|
| User-level (7) | `user_review_count`, `user_avg_rating`, `user_rating_std`, `user_unique_prod_count`, `user_same_day_review_count`, `user_review_day_gap`, `is_user_first_review` | 작성자의 반복·이상 행동 패턴 |
| Product-level (4) | `prod_review_count`, `prod_avg_rating`, `prod_rating_std`, `prod_unique_user_count` | 특정 상품에 축적된 리뷰 분포 |
| Rating Deviation (3) | `rating_deviation_from_prod`, `rating_deviation_from_user`, `rating_is_extreme` | 별점의 이상 편차 |
| Text Statistic (7) | `review_char_len`, `review_word_count`, `unique_word_ratio`, `exclamation_count`, `question_count`, `uppercase_ratio`, `digit_ratio` | 리뷰 본문 표면 통계 |
| Temporal (3) | `review_month`, `review_dayofweek`, `prod_same_day_review_count` | 작성 시점 및 burst 탐지 |

최종 피처 행렬에는 `StandardScaler`를 적용하며, 결측치는 0으로 처리합니다.

---

## 4종 관계 그래프

리뷰를 노드, 리뷰 간 관계를 엣지로 정의하는 multi-relation graph를 구성합니다.

| 관계명 | 연결 기준 | 포착하는 fraud signal |
|---|---|---|
| `net_upu` | 같은 `user_id` | 동일 작성자의 반복 이상 행동 |
| `net_usu` | 같은 `prod_id` + 같은 `rating` | 평점 몰아주기·공격 (rating homophily) |
| `net_uvu` | 같은 `prod_id` + 같은 `review_month` | 시간적으로 집중된 burst 리뷰 |
| `net_utu` | char n-gram TF-IDF cosine ≥ 0.4 | 템플릿 재사용·복사붙여넣기 |

모든 엣지는 `i→j`와 `j→i`를 함께 생성하는 유향 엣지로 구성합니다. 전체 그래프 엣지 수는 약 **9,548,722개**입니다.

---

## 모델 구조: LA-SAGE-S (PMP-514)

```
입력: 24차원 hand-crafted feature × 46,547 노드
↓ 관계별 LASAGESConv (net_upu / net_usu / net_uvu / net_utu)
    - 이웃을 fraud / benign / unknown 3그룹으로 partition
    - 각 그룹에 별도 MLP (LI-Linear) transformation 적용
    - h_neigh = fraud_emb + benign_emb + unknown_emb
    - h_out = fc_self(h_self) + h_neigh
↓ 관계별 출력 4개를 concat (40 × 4 = 160d)
↓ Linear: 160d → 40d
↓ Dropout (p=0.2)
↓ Conv1d projection head: 40d → 2d
↓ class logits → softmax
```

### 핵심 설계 원칙

**Partition Aggregation (이웃 분리 집계)**  
가짜 리뷰가 정상 이웃과 연결되는 camouflage 상황에서, 일반 GNN은 이웃 평균 시 fraud signal이 희석됩니다. PMP-514는 이웃을 제거하지 않고 fraud / benign / unknown 3그룹으로 분리하여 각각 다른 변환을 적용합니다. 레이블 미확인 이웃은 학습된 balance gate를 통해 두 그룹의 가중 혼합으로 처리됩니다.

**LI-Linear (Label-Informed Linear)**  
이웃 집계 시 중심 노드의 원본 피처(`h_self`)를 조건으로 삼아 변환을 조정하는 커스텀 선형 레이어입니다.

**얕은 GNN (n_layer = 1)**  
깊은 propagation일수록 멀리 있는 정상 노드의 신호가 fraud node에 섞여 signal이 희석됩니다. multi-relation + partition aggregation + ranking-aware loss로 표현력을 확보합니다.

**손실 함수**  
- **Focal Loss** (γ=2.0): 쉬운 샘플의 영향력을 낮추고 어려운 fraud sample 학습에 집중
- **Pairwise Ranking Loss** (α=0.5, margin=1.0): fraud score가 benign score보다 margin 이상 높도록 직접 압박

---

## 프로젝트 구조

```
PMP-514/
├── main.py                         # 학습/평가 진입점
├── config/
│   └── bridge.yml                  # 하이퍼파라미터 설정
├── model/
│   ├── base.py                     # MLP, LILinear, LIMLP 기반 모듈
│   ├── LASAGE_S.py                 # LA-SAGE-S 모델 (LASAGESConv + LASAGE_S)
│   └── SAGE.py                     # 베이스라인 GraphSAGE
├── DataHelper/
│   ├── dataset.py                  # 기본 데이터셋 클래스
│   ├── datasetHelper.py            # DGL 그래프 로딩 및 DataLoader 구성
│   └── sampler.py                  # 샘플링 유틸리티
├── training_procedure/
│   ├── __init__.py                 # Trainer 클래스
│   ├── prepare.py                  # 모델/옵티마이저/손실함수 초기화
│   ├── train.py                    # 학습 루프 (에폭 단위)
│   └── evaluate.py                 # 추론 및 지표 계산
├── datasets/
│   ├── final_B_bridge_sample_with_text.csv   # 전처리된 최종 학습 샘플
│   └── preprocess_yelpzip.py       # YelpZip 전처리 스크립트
├── utils/
│   ├── utils.py                    # 설정 로딩, 체크포인트 입출력
│   ├── logger.py                   # 로깅 유틸리티
│   ├── plot_tools.py               # 시각화 도구
│   ├── random_seeder.py            # 재현성을 위한 시드 설정
│   └── constants.py                # 공통 상수
├── checkpoints/                    # 저장된 모델 가중치 및 결과 파일
├── logs/                           # 학습 로그 파일
├── dashboard_data/                 # Streamlit 대시보드용 내보낸 데이터
├── pic/                            # 아키텍처 다이어그램
└── requirements.txt
```

---

## 설치

```bash
# 1. PyTorch (CUDA 11.8)
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. PyTorch-Geometric 희소 연산
pip install torch-scatter==2.1.1+pt20cu118 torch-sparse==0.6.17+pt20cu118 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 3. DGL (CUDA 12.1 휠)
pip install dgl==1.1.3+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

# 4. CuPy
pip install cupy-cuda12x==13.3.0

# 5. Sentence Transformers
pip install sentence-transformers==2.7.0 transformers==4.41.0

# 6. 나머지 의존성
pip install -r requirements.txt
```

> **주의**: DGL과 PyTorch의 CUDA 버전이 호환되어야 합니다. 위 휠은 CUDA 12.1 런타임 / PyTorch CUDA 11.8 바이너리 조합으로 빌드되었습니다.

---

## 데이터 분할

`dataset_seed=717` 기준 stratified split으로 각 분할의 가짜 리뷰 비율을 동일하게 유지합니다.

| 분할 | 가짜 / 전체 | 비율 |
|---|---|---|
| Train | 3,593 / 27,928 | 60% |
| Valid | 1,198 / 9,309 | 20% |
| Test | 1,198 / 9,310 | 20% |
| 전체 | 5,989 / 46,547 | 100% |

---

## 학습 실행

```bash
python main.py \
    --dataset bridge \
    --model LA-SAGE-S \
    --hyper_file config/ \
    --data_dir datasets/ \
    --log_dir logs/ \
    --best_model_path checkpoints/
```

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `n_layer` | 1 | 1-hop, over-smoothing 회피 |
| `hid_dim` | 40 | 관계별 hidden 차원 |
| `full_neighbors` | True | 1-hop 이웃 누락 없음 |
| `relation_agg` | cat | concat → Linear 결합 |
| `proj` | True | Conv1d projection head 사용 |
| `dropout` | 0.2 | projection 직전 적용 |
| `focal_gamma` | 2.0 | hard sample 집중 강도 |
| `ranking_alpha` | 0.5 | ranking loss 가중치 |
| `ranking_margin` | 1.0 | fraud-vs-benign 마진 |
| `optimizer` | Adam | 옵티마이저 |
| `lr` | 0.01 | 초기 학습률 |
| `weight_decay` | 0.001 | L2 정규화 |
| `batch_size` | 512 | 노드 배치 크기 |
| `epochs / patience` | 120 / 25 | AP 기준 early stopping |
| `monitor` | ap_gnn | best model 선택 기준 |
| `seed` | 1234 | 재현성 고정 |

### 저장된 모델 평가

```bash
python main.py --dataset bridge --model LA-SAGE-S --run_best
```

---

## 평가 지표

| 지표 | 설명 | 비고 |
|---|---|---|
| PR-AUC (ap_gnn) | Precision-Recall 곡선 아래 면적 | **1차 모니터링 지표** |
| ROC-AUC (auc_gnn) | ROC 곡선 아래 면적 | 불균형 환경에서 낙관적일 수 있음 |
| Macro-F1 (f1_macro) | 클래스 평균 F1 | threshold moving과 함께 보고 |
| G-Mean (gmean_gnn) | 민감도·특이도의 기하 평균 | 클래스 간 균형 요약 |
| F1 (fraud=1) | 가짜 클래스 F1 | 실제 탐지 성능의 직접 지표 |
| Recall / Precision (fraud=1) | 가짜 클래스 재현율·정밀도 | threshold moving 후 보고 |

최종 테스트 threshold는 Validation set의 Precision-Recall 곡선 최적점(`best_pr_thres`)에서 결정합니다.

---

## 실험 결과

**PMP-514 (YelpZip bridge 샘플, seed=1234, best epoch=30)**

| 지표 | Validation (E30) | Test (E30) |
|---|---|---|
| ROC-AUC | 0.9067 | **0.9049** |
| PR-AUC / AP | 0.6789 | **0.6848** |
| Macro-F1 | 0.7653 | **0.7697** |
| Macro-Recall | 0.7650 | 0.7743 |
| G-Mean | 0.7447 | 0.7571 |
| F1 (fraud=1) | 0.5909 | 0.5998 |
| F1 (benign=0) | 0.9397 | 0.9395 |
| Recall (fraud=1) | 0.5901 | 0.6118 |
| Precision (fraud=1) | 0.5916 | 0.5883 |
| best_pr_thres | 0.5022 | 0.5417 |

결과는 `checkpoints/val_<seed>/LA-SAGE-S/<dataset>/<timestamp>/results.txt`에 저장됩니다.

---

## 결과 해석

- **ROC-AUC 0.905 vs PR-AUC 0.685**: 클래스 불균형(가짜 약 13%) 환경에서 ROC-AUC는 낙관적으로 보일 수 있습니다. 상위 fraud score 구간의 정밀도는 PR-AUC 0.685가 더 정확히 반영합니다.
- **Validation ↔ Test 일관성**: 두 분할 간 모든 지표 차이가 0.01 이내로, 모델이 valid set에 과적합되지 않았음을 시사합니다.
- **Fraud F1 (0.60) vs Benign F1 (0.94)**: fraud는 빈도가 낮고 위장 패턴이 다양해 더 어렵습니다. "가짜 10개 중 약 6개 탐지, 가짜 판정 중 약 6개가 실제 가짜"로 해석됩니다.

---

## 재현성

```bash
python main.py --dataset bridge --model LA-SAGE-S --seed 1234
```

`utils/random_seeder.py`를 통해 Python `random`, NumPy, PyTorch 시드가 모두 고정됩니다.

---

## 참고 자료

- Mukherjee et al. (2012). *Spotting Fake Reviewer Groups in Consumer Reviews.* WWW. — YelpZip 데이터셋 원천
- Rayana & Akoglu (2015). *Collective Opinion Spam Detection.* KDD. — R-T-R / R-U-R / R-S-R 관계 정의의 출발점
- Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS. — LA-SAGE-S 백본의 기반
- Dou et al. (2020). *Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters (CARE-GNN).* CIKM. — multi-relation graph 기반 가짜 리뷰 탐지 대표 연구
