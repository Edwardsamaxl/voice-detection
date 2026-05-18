# 语音处理与声纹识别系统 — 实现计划（V2.1 封装版）

> **目标读者**：实现 Agent。本文档基于与负责人的需求对齐结果编写，实现时请严格遵循本文档，如有歧义优先以本文档为准。
> **版本**：V2.3（2026-05-18）— 输出字段扩展（segment_id, file, local_speaker, duration, sr, score, precision, embedding）；speaker_db / vector_db 持久化拆分；识别流程升级为两层比对 + FAISS 全局一致性校验（center 粗排 → FAISS Top-K 检索 → `topk_consistency` 5 选 4）；聚类取消 `min_cluster_size` 限制，由两层聚类自然消解碎片；**识别阈值经三组 leave-out 测试校准：`T_LOW=0.50`（快速拒识）、`T_HIGH=0.65`（最终确认）**。

---

## 一、系统定位与输入输出

### 1.1 系统目标

构建一套**多人对话语音处理系统**。

- **输入**：音频文件（可能包含多人对话，也可能是单说话人音频）。
- **输出**：按时间段、按说话人分段的结构化结果，包含：
  - **时间段**：start, end
  - **Speaker ID**：全局唯一标识（如 SPK_0）
  - **Display Name**：人名（如"张三"，可初始为 UNKNOWN）
  - **缅甸语原文**：Whisper ASR 识别出的缅甸语文本
  - **中文翻译**：缅甸语 → 中文的翻译结果

最终形态：输入一段新音频，系统能输出**谁**在**什么时间**说了**什么**（缅甸语），以及**中文翻译**是什么。

### 1.2 输出示例（V2.2 扩展字段）

```json
[
  {
    "segment_id": "meeting_0012_0000",
    "file": "meeting_0012.wav",
    "start": 0.0,
    "end": 3.2,
    "duration": 3.2,
    "sr": 16000,
    "local_speaker": "SPEAKER_00",
    "global_speaker": "SPK_0",
    "display_name": "张三",
    "score": 0.823,
    "text": "မင်္ဂလာပါ",
    "translation": "你好",
    "precision": null,
    "embedding": [0.0123, -0.0456, ...]
  }
]
```

**字段说明（ recognize 输出）**：

| 字段 | 来源 | 说明 |
|------|------|------|
| `segment_id` | `SpeakerSegment.segment_id` | 加工后的唯一标识，格式 `{basename}_{seq:04d}` |
| `file` | `SpeakerSegment.file` | 来源音频文件名 |
| `start` / `end` | diarization | 起止时间（秒） |
| `duration` | derived | `end - start` |
| `sr` | preprocess | 采样率，固定 16000（ffmpeg 已标准化） |
| `local_speaker` | pyannote | 原始 diarization 标签，如 `SPEAKER_00` |
| `global_speaker` | clustering / identify | 全局 speaker ID，如 `SPK_0`、`BUR_0366`、`UNKNOWN` |
| `display_name` | `SpeakerProfile.name` | 人名，未标注时为 `null` |
| `score` | `IdentificationResult.score` | 说话人识别置信度（余弦相似度，范围 [-1, 1]）；`UNKNOWN` 时为 `null` |
| `text` | ASR | 缅甸语识别文本 |
| `translation` | NLLB | 中文翻译，未启用时为 `null` |
| `precision` | eval | 文本准确度（如 WER/CER），需有参考文本时计算；无参考时为 `null` |
| `embedding` | extractor | 当前语段的 192 维声纹向量（float32 list）|

### 1.3 核心能力清单

1. **音频标准化**：任意格式 → 16kHz/16bit/单声道 wav
2. **说话人分段（Diarization）**：Pyannote，输出时间切片 + local_speaker 标签
3. **语音片段预处理**：过滤 <1s、平滑碎片（min_duration_on/off）
4. **声纹提取**：SpeechBrain ECAPA-TDNN，192 维向量
5. **全局聚类**：跨文件聚合同一说话人，生成全局 SPK_ID
6. **声纹库管理**：加权 center + FAISS 多向量索引 + speaker_profile 人名元数据
7. **语音识别（ASR）**：Whisper，输出缅甸语原文
8. **翻译**：缅甸语 → 中文（机器翻译模型，离线优先）
9. **新音频识别**：提取 embedding → FAISS TopK 检索 → 双阈值判断 → 识别或新建 Speaker
10. **动态更新**：高置信度结果自动增量更新声纹库

---

## 二、系统架构与数据流

### 2.1 最优架构（三层 + Core）

```
src/core/
  ├── types.py          → SpeakerSegment, SpeakerProfile, SpeakerData (不可变 dataclass)
  ├── pool.py           → EmbeddingPool (封装 segments + matrix + meta_index)
  ├── storage.py        → Storage Protocol + JsonStorage / NpzStorage / MemoryStorage
  ├── repository.py     → SpeakerRepository (统一 seam: speaker_db (含精选向量) + vector_index + profile)
  └── pipeline.py       → 【已清理】原 Stage Protocol + Pipeline 死代码已删除

数据流：

EmbeddingPool（全量原始数据，封装后）
        ↓
SpeakerRepository.build_from_pool()
        ↓
  ├─→ speaker_db（识别库）   → 每人一条记录：weighted center + 精选 embeddings（≤20 个）
  ├─→ vector_db（全量库）    → 聚类前/后的全部原始向量（中间数据，供重新聚类或追溯）
  ├─→ speaker_profile（人名表）→ 显示层（SPK_ID → 姓名）
  └─→ data_store（结果缓存）  → 业务数据（分段 + speaker + text + translation）
```

### 2.2 核心数据结构（封装版）

**（1）SpeakerSegment（最小不可变单元）**

```python
@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    segment_id: str       # 唯一标识，如 "A_0001"
    file: str             # 来源文件
    start: float
    end: float
    local_speaker: str    # pyannote 内部分段标签
    global_speaker: str | None = None   # 聚类后统一 ID
    display_name: str | None = None
    embedding: np.ndarray | None = None
    score: float | None = None          # 说话人识别置信度（余弦相似度）
    text: str | None = None
    translation: str | None = None
    sr: int = 16000                     # 采样率，ffmpeg 标准化后固定 16000

    @property
    def duration(self) -> float: ...
    def with_embedding(self, emb) -> SpeakerSegment: ...
    def with_global_speaker(self, spk_id) -> SpeakerSegment: ...
    def with_score(self, score: float | None) -> SpeakerSegment: ...
    def with_text(self, text, translation=None) -> SpeakerSegment: ...
    def with_sr(self, sr: int) -> SpeakerSegment: ...
```

> **设计原则**：`frozen=True` 保证不可变，下游模块通过 `with_*` 方法派生新版本，杜绝偷偷改字段。

**（2）EmbeddingPool（替代裸 dict embedding_pool）**

```python
class EmbeddingPool:
    """封装 segments 列表、meta_index 和 matrix 缓存。"""

    def add(self, segment: SpeakerSegment) -> None: ...
    def get_by_id(self, segment_id: str) -> SpeakerSegment | None: ...
    def to_matrix(self) -> np.ndarray: ...          # 懒构建 + 缓存
    def apply_labels(self, labels: list[int], prefix="SPK_") -> EmbeddingPool: ...
    def filter_by_speaker(self, spk_id: str) -> list[SpeakerSegment]: ...
```

> **关键差异**：旧设计中 `embedding_pool` 是裸 dict，调用者需要手动维护 `matrix` 和 `meta_index` 的一致性。新设计中 `EmbeddingPool` 内部管理这些一致性，`to_matrix()` 只在 dirty 时重建。

**（3）SpeakerData（speaker_db 识别库记录）**

```python
@dataclass(frozen=True, slots=True)
class SpeakerData:
    spk_id: str
    center: np.ndarray          # 加权平均 + L2 归一化后的中心向量（用于第一层粗排）
    embedding_count: int        # 该 speaker 当前保留的向量数
    embeddings: list[np.ndarray] | None = None   # 精选向量（≤MAX_EMB 个），用于第二层精排和 FAISS 建索引
    durations: list[float] | None = None         # 精选向量对应的原始片段时长
    profile: SpeakerProfile | None = None
```

> **V2.3 架构修正**：精选向量保留在 `SpeakerData` 内，随 `speaker_db` 一并序列化。单个 speaker 最多 20 个精选向量（20 × 192 维 float 数组），JSON 体积完全可控；识别时只需加载 `speaker_db` 即可获得全部识别所需数据（center + 精选向量），无需额外查询全量库。
>
> **识别时职责**：`speaker_db` 里的 `center` 用于第一层快速缩小候选范围；`embeddings` 用于构建 FAISS 索引并做 Stage 2 精排和一致性校验。

**（4）SpeakerProfile（人名元数据表）**

```python
@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    name: str
    alias: list[str] | None = None
    gender: str | None = None
    notes: str | None = None
    created_at: str | None = None
```

**（5）VectorDb（全量原始向量库，V2.3 修正）**

```python
@dataclass(frozen=True, slots=True)
class VectorEntry:
    spk_id: str
    embedding: np.ndarray
    duration: float

class VectorDb:
    """管理聚类前/后的全部原始向量（中间数据）。

    V2.2 错误地把精选向量放到 VectorDb，导致识别时必须加载 vector_db，
    而 speaker_db 沦为纯元数据壳子。V2.3 修正：
    - speaker_db: 存 center、embedding_count、embeddings（精选向量）、profile（识别库）
    - vector_db:  存全部原始向量（聚类前后所有 segments），供重新聚类、追溯或离线分析使用

    在线识别流程只依赖 speaker_db；vector_db 是拓展性质的中间数据，不参与识别。
    """
    def add(self, spk_id: str, embedding: np.ndarray, duration: float) -> None: ...
    def get_entries(self, spk_id: str) -> list[VectorEntry]: ...
    def all_vectors(self) -> tuple[np.ndarray, list[str]]: ...  # (matrix, labels)
```

**（6）SpeakerRepository（统一 seam，V2.2 调整）**

```python
class SpeakerRepository:
    def build_from_pool(self, pool: EmbeddingPool, max_emb: int = MAX_EMB) -> None: ...
    def identify(self, query_emb: np.ndarray) -> IdentificationResult: ...
    def add_speaker(self, spk_id: str, embeddings: list[np.ndarray], durations: list[float], profile: SpeakerProfile | None = None) -> None: ...
    def update_speaker(self, spk_id: str, new_emb: np.ndarray, duration: float) -> bool: ...
    def assign_name(self, spk_id: str, name: str) -> None: ...
    def get_speaker(self, spk_id: str) -> SpeakerData | None: ...
    def all_speakers(self) -> list[str]: ...
    def rebuild(self) -> None: ...
    def save(self, speaker_key: str = "speaker_db:main") -> None: ...
    def load(self, speaker_key: str = "speaker_db:main") -> None: ...
```

> **关键设计**：`SpeakerRepository` 是**唯一对外 seam**。内部持有 `speaker_db`（含精选向量）和 `_vector_index`（FAISS），在线识别只需这两者；`vector_db`（全量原始向量）是可选的中间数据，不挂在 `SpeakerRepository` 内。
>
> **V2.3 持久化**：
> - `speaker_db:main` → JSON：`{spk_id: {"center": [...], "embedding_count": 20, "embeddings": [...], "durations": [...], "profile": {...}}}`（精选向量每人≤20个，JSON 体积可控）
> - `vector_db:main` → NPZ：`{"spk_ids": [...], "embeddings": (N, 192), "durations": (N,)}`（全量原始向量，中间数据，不参与识别）
> - `vector_index:main` → Pickle：直接序列化 FAISS `IndexFlatIP` 对象，加载时无需重建即可直接搜索
>
> **识别流程（两层比对 + FAISS 一致性校验）**：
> 1. `identify(query_emb)` 先用 `speaker_db` 中所有人的 `center` 做余弦相似度排序，选出 Top-N 候选；
> 2. 若最像的候选 center 分数过低（如低于 `T_LOW`），直接返回 `UNKNOWN`，避免无意义的精排；
> 3. **Stage 2 精排**：调用 `_vector_index.search(query_emb, TOPK)`，从**全部精选向量**中全局检索 Top-K 最近邻；
> 4. **一致性校验**：用 `topk_consistency(topk_results, k=TOPK, min_consistency=TOPK_CONSISTENCY_MIN)` 判断 Top-K 结果中是否有足够多数（默认 5 选 4）属于**同一个 speaker**；若不一致，返回 `UNKNOWN`；
> 5. 取一致性指向的 speaker，计算其精选向量与 query 的平均相似度作为最终分数；若 ≥ `T_HIGH` 且该 speaker 在 center 粗排候选中，返回 `KNOWN`；否则返回 `UNKNOWN`。
>
> **关键变更**：V2.2 之前 Stage 2 是逐个候选 speaker 取精选向量做内积，"Top-K 一致"天然成立，没有发挥 `topk_consistency` 的作用。新流程改为**先全局 FAISS 检索、再一致性过滤**，真正利用多向量索引的交叉验证能力。

**（7）VectorIndex（FAISS 适配 seam）**

```python
class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray, labels: list[str]) -> None: ...
    def search(self, query: np.ndarray, topk: int = 5) -> list[SearchResult]: ...
    def add(self, vector: np.ndarray, label: str) -> None: ...
    def rebuild(self) -> None: ...
```

> 具体实现为 `FaissVectorIndex`（`src/speaker_db/vector_index.py`）。`repository.py` 中保留 `VectorIndex` 作为 Protocol 别名，便于后续替换为 `IndexIVFFlat` 或其他库（Milvus / Qdrant）而无需修改 `SpeakerRepository`。

**（8）Storage（统一持久化 seam）**

```python
class Storage(Protocol):
    def save(self, key: str, data: Any) -> None: ...
    def load(self, key: str) -> Any: ...
    def exists(self, key: str) -> bool: ...
```

实现：
- `JsonStorage` — metadata、segments、profiles（替换 `diarization/cache.py`, `speaker_profile` 存储）
- `NpzStorage` — embedding matrices（替换 `embedding/cache.py`）
- `PickleStorage` — FAISS indices、复杂对象
- `MemoryStorage` — 测试专用

> **统一 key 命名**：`"segments:bur_3260"`、`"embeddings:bur_3260"`、`"speaker_db:main"`、`"vector_db:main"`、`"vector_index:main"`、`"profiles:main"`

---

## 2.3 V2.3 架构修正：speaker_db 含精选向量，vector_db 为全量原始库

### 问题

V2.2 把精选向量从 `SpeakerData` 剥离到 `VectorDb`，导致：
1. 在线识别时必须同时加载 `speaker_db` JSON 和 `vector_db` NPZ，识别库被拆成两份；
2. `SpeakerRepository` 内部需要同时维护 `_speakers` 和 `_vector_db` 两个存储，职责混乱；
3. `vector_db` 名字暗示它是向量主库，实际却成了精选向量的附属容器，而真正的全量原始向量反而散落在 `NpzStorage` 的各个 key 里。

### 目标架构（V2.3）

```
SpeakerRepository（在线识别）
  ├─ _speakers: dict[str, SpeakerData]      # 识别库：center + 精选向量 + profile
  └─ _vector_index: FaissVectorIndex        # FAISS 检索索引（由 speaker_db.embeddings 重建）

VectorDb（离线中间数据，不由 SpeakerRepository 持有）
  └─ 全量原始向量（聚类前/后所有 segments，供重新聚类、追溯、分析）
```

### 持久化

| 存储 | key | 格式 | 内容 |
|------|-----|------|------|
| speaker_db | `speaker_db:main` | JSON | `{spk_id: {"center": [...], "embedding_count": 20, "embeddings": [...], "durations": [...], "profile": {...}}}` |
| vector_db | `vector_db:main` | NPZ | `{"spk_ids": [...], "embeddings": (N, 192), "durations": (N,)}`（全量原始向量，中间数据）|
| vector_index | `vector_index:main` | Pickle | FAISS `IndexFlatIP` 缓存（可由 speaker_db 重建）|

### 职责边界

- **speaker_db**：在线识别的唯一数据源。`SpeakerRepository.save/load` 只需读写 `speaker_db:main` 一个文件即可恢复完整的识别能力（center + 精选向量 + profile）。FAISS 索引可由 `speaker_db.embeddings` 重建，因此 `vector_index:main` 只是加速缓存，丢失后可自动重建。
- **vector_db**：全量原始向量库，由聚类/建库脚本（如 `scripts/build_embedding_pool.py`）写入，供重新聚类或离线分析。`SpeakerRepository` 不持有它，也不依赖它做识别。

---

## 三、分阶段实现计划

### Phase 1: 项目骨架与音频预处理

**目标**：能跑通音频标准化，输出 16kHz/16bit/单声道 wav。

| 文件 | 核心内容 |
|------|---------|
| `config.py` | 所有常量（见第 8 节参数表） |
| `src/core/types.py` | **【新增】**核心 dataclass：SpeakerSegment, SpeakerProfile, SpeakerData 等 |
| `src/core/pool.py` | **【新增】**EmbeddingPool 封装 |
| `src/core/storage.py` | **【新增】**Storage Protocol + Adapter |
| `src/core/repository.py` | **【新增】**SpeakerRepository + VectorIndex Protocol（骨架） |
| `src/core/pipeline.py` | 【已清理】原 Stage Protocol + Pipeline 死代码已删除 |
| `src/audio/preprocess.py` | `convert_to_wav()`, `load_wav()`, `crop_segment()`, `rms_normalize()` |
| `requirements.txt` | `torch`, `pyannote.audio`, `speechbrain`, `scikit-learn`, `faiss-cpu`, `openai-whisper`, `numpy`, `scipy`, `soundfile`, `pytest`, `pytest-cov` + 翻译模型依赖 |

**验证标准**：输入任意格式音频，输出符合规范的 wav；`SpeakerSegment` 和 `EmbeddingPool` 单元测试通过。

---

### Phase 2: 说话人分段

**目标**：输入 wav，输出带时间戳的说话人片段（`list[SpeakerSegment]`）。

| 文件 | 核心内容 |
|------|---------|
| `src/diarization/segment.py` | `run_diarization(wav_path) -> Annotation`；封装 pyannote |
| `src/diarization/postprocess.py` | `merge_short_segments(annotation)`；`annotation_to_segments(annotation, min_duration=1.0) -> list[SpeakerSegment]` |

**关键变更**：
- `annotation_to_segments()` 返回 `list[SpeakerSegment]`（不再是 `list[dict]`）
- `SpeakerSegment` 的 `global_speaker`, `display_name`, `embedding`, `text`, `translation` 初始为 `None`
- `segment_id` 和 `file` 由调用方（如 `scripts/build_embedding_pool.py`）在拿到 `list[SpeakerSegment]` 后统一注入

**验证标准**：一段多人对话音频，能正确切分出不同说话人的 `SpeakerSegment`。

---

### Phase 3: 声纹向量提取与归一化

**目标**：从每个 `SpeakerSegment` 提取 192 维 embedding。

| 文件 | 核心内容 |
|------|---------|
| `src/embedding/extractor.py` | `class EmbeddingExtractor`：`extract(wav_array) -> np.ndarray`；`extract_segments(wav_path, segments: list[SpeakerSegment]) -> list[SpeakerSegment]`（返回新实例，不修改输入） |
| `src/embedding/normalize.py` | `l2_normalize(emb) -> emb / np.linalg.norm(emb)` |

**关键变更**：
- `extract_segments()` 输入 `list[SpeakerSegment]`，返回 `list[SpeakerSegment]`（通过 `seg.with_embedding(emb)` 生成不可变副本）
- 不再需要独立的 `embedding/cache.py` —— 使用 `src/core/storage.py` 中的 `NpzStorage`

**验证标准**：同一人的两个片段，cosine similarity > 0.85；不同人 < 0.5。

---

### Phase 4: 全局聚类、去重与 Speaker ID 回写

**目标**：跨文件聚合同一说话人，生成全局 SPK_ID，并回写到 `EmbeddingPool`。

| 文件 | 核心内容 |
|------|---------|
| `scripts/build_embedding_pool.py` | 批量：wav → diarization → embedding → `EmbeddingPool` → `.npz` 缓存（通过 `NpzStorage`） |
| `src/clustering/pool.py` | `build_embedding_pool(npz_dir, segment_dir) -> EmbeddingPool`：从所有 `.npz` + `.json` 文件加载，构建 `EmbeddingPool` |
| `src/clustering/cluster.py` | **两层聚类**：`agg_clustering(embeddings, distance_threshold=0.3, centroid_threshold=0.35) -> labels`。第一层用保守阈值做 `AgglomerativeClustering`（保证纯度），第二层按类 centroid 合并相近的簇（减少碎片化）；不设置 `min_cluster_size`，避免新 speaker 因片段不足被丢弃；去重用矩阵向量化 dot product；`pool.apply_labels(labels)` 回写 |

**关键变更**：
- 批量提取后的中间数据通过 `NpzStorage` 持久化，key 如 `"embeddings:bur_3260"`
- `cluster.py` 操作的是 `EmbeddingPool` 对象，聚类后通过 `pool.apply_labels(labels)` 返回**新 pool**
- 不再直接操作裸 `embedding_pool["segments"][i]["global_speaker"]`

**验证标准**：
1. 同一人在 A.wav 和 B.wav 中的片段，被分配到同一个 cluster label；两层聚类不要求所有片段被第一层直接合并，允许经第二层 centroid 合并后归入同一 speaker。
2. 回写后，`EmbeddingPool.filter_by_speaker("SPK_0")` 能正确返回该 speaker 的所有 segments。

**2026-05-16 聚类策略与实测结论**：
- SLR80 train（2330 条，20 位女性说话人）真实测试：`distance_threshold=0.30` 单层聚类产生 562 个类（100% 纯度），`0.55` 产生 24 个类但纯度降至 97.2%。
- 采用两层聚类 `thr=0.30, cthr=0.35` 后（不设置 `min_cluster_size`，避免丢弃新 speaker 的少量片段），类数降至 21，纯度 98.8%。这是当前数据上**泛化与精度最平衡的参数**。
- **关键发现**：该数据集 `intra-speaker max distance=0.854`，`inter-speaker min distance=0.323`，说明 ECAPA-TDNN 在缅甸语女性说话人上类内/类间分布存在显著重叠；单层高阈值容易过拟合到该统计特性，两层策略用保守第一层 + centroid 第二层更稳健。
- 策略原则：**不追求 100% 纯度**（会过拟合到当前数据集），优先保证未知数据上的稳定性。

---

### Phase 5: Speaker 库与向量索引

**目标**：建立可快速检索的声纹库，支持多向量表示。

| 文件 | 核心内容 |
|------|---------|
| `src/speaker_db/vector_index.py` | `class FaissVectorIndex`：实现 `build()`, `search()`, `add()`, `rebuild()`；`VectorIndex` 保留为 Protocol 别名 |
| `src/speaker_db/builder.py` | **【废弃】**功能已并入 `src/core/repository.py` |
| `src/speaker_db/profile.py` | **【废弃】**功能已并入 `src/core/repository.py` |
| `src/speaker_db/updater.py` | **【废弃】**功能已并入 `src/core/repository.py` |
| `src/speaker_db/storage.py` | **【废弃】**功能已并入 `src/core/storage.py` |

**关键设计**：
- FAISS 索引中每个 speaker 存入最多 `MAX_EMB=20` 个精选 embedding（不是只存 1 个 center）。
- 小规模（<10 万向量）使用 `IndexFlatIP`。
- `SpeakerRepository` 是统一对外接口，内部持有 `FaissVectorIndex` 实例。

**验证标准**：100 个 speaker 的多向量索引，FAISS 检索耗时 < 10ms。

---

**关于 "vector_db" 与全量原始向量的说明**：

文档中 `speaker_db:main` 存储的是**识别库**，包含每人的中心向量和精选向量（≤20 个 / speaker），用于在线识别的两层比对。

`vector_db:main` 存储的是**全量原始向量**（聚类前/后所有 segments 的 embedding），作为中间数据供重新聚类、追溯或离线分析使用，不参与在线识别。此外，全量原始向量也可由 `NpzStorage` 按文件单独管理（key 如 `"embeddings:bur_3260"`）。

---

### Phase 6: 语音识别（ASR）与翻译

**目标**：Whisper 识别缅甸语原文，再翻译成中文。

| 文件 | 核心内容 |
|------|---------|
| `src/asr/whisper_asr.py` | `transcribe(wav_path, language="my") -> list[SpeakerSegment]`（返回带 `text` 字段的 SpeakerSegment 列表） |
| `src/translation/translator.py` | 缅甸语 → 中文翻译模块。加载本地 NMT 模型，输入缅甸语句子，输出中文翻译。预留 API fallback 接口。 |

**注意**：翻译模块需离线可用，优先加载本地模型。如本地模型效果不佳，可预留 API 接口（Google Translate / 百度翻译）作为 fallback。

**2026-05-16 真实测试发现**：
- Whisper `tiny` 在真实 SLR80 dev wav 上可以跑通，但缅甸语识别质量不可用，输出出现中英混杂与乱码式幻觉；因此 Whisper 只能作为工程连通性验证，不能作为缅甸语 ASR 主模型。
- 当前翻译实现默认使用 `facebook/nllb-200-distilled-600M`（`mya_Mymr -> zho_Hans`），方案方向可保留，但本机缓存不完整，且当前依赖组合加载 NLLB 时出现 `scale_embedding expected int, got bool`。翻译真实质量尚未验证。
- Phase 6 的优先级应调整为：先替换/接入专用缅甸语 ASR（优先试 ModelScope `damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch` 或 SLR80/XLS-R fine-tuned 模型），再修 NLLB/翻译 fallback。若 ASR 文本错误，后续翻译质量没有评价意义。

---

### Phase 7: 结果整合与说话人对齐

**目标**：将 Whisper 的 ASR 结果、翻译结果，与 diarization 的说话人时间段对齐。

| 文件 | 核心内容 |
|------|---------|
| `src/asr/align.py` | `align_segments(whisper_segments: list[SpeakerSegment], diarization_annotation) -> list[SpeakerSegment]` |

**对齐逻辑**：
1. 对每条 Whisper segment，用 `annotation.crop(Segment(start, end)).argmax()` 找占比最大的 speaker。
2. **边界处理**：
   - 如果 `crop` 结果为空 → `speaker = "UNKNOWN"`
   - 如果 `end - start < 0.5` → `speaker = "IGNORE"`
3. 将对应的中文翻译挂在 segment 上（通过 `seg.with_text(text, translation)`）。

**输出数据结构**：`dict[str, list[SpeakerSegment]]`（文件 → segments）。

**验证标准**：识别的文本能正确归属到对应的 global_speaker，翻译结果语义通顺。

---

### Phase 8: 新音频识别流程

**目标**：输入新音频，识别其中说话人是谁，或判断为未知。

| 文件 | 核心内容 |
|------|---------|
| `src/recognition/identify.py` | **【废弃】**已并入 `SpeakerRepository.identify()` |
| `src/recognition/verify.py` | `topk_consistency(topk_results, k=5, min_consistency=4) -> bool` |

**两层比对 + FAISS 一致性校验**（在 `SpeakerRepository.identify()` 内部实现）：
```python
# Stage 1：与所有 speaker 的 center 向量比对，快速缩小范围
candidates = compare_with_centers(query_emb, topn=N_CANDIDATES)

# 若最像的候选 center 分数过低，直接拒识
if max(candidates, key=lambda c: c.score).score < T_LOW:
    return IdentificationResult(spk_id=None, score=None, status="UNKNOWN")

# Stage 2：用 FAISS 从全部精选向量中全局检索 Top-K
topk_results = vector_index.search(query_emb, topk=TOPK)

# 一致性校验：5 个中至少 4 个属于同一 speaker
if not topk_consistency(topk_results, k=TOPK, min_consistency=TOPK_CONSISTENCY_MIN):
    return IdentificationResult(spk_id=None, score=None, status="UNKNOWN")

# 取一致性 speaker，计算其精选向量的平均相似度作为最终分数
best_speaker = most_common_speaker(topk_results)
final_score = mean_similarity(query_emb, speaker_db.get_entries(best_speaker), k=TOPK)

if final_score >= T_HIGH and best_speaker in candidates:
    return IdentificationResult(spk_id=best_speaker, score=final_score, status="KNOWN")

return IdentificationResult(spk_id=None, score=None, status="UNKNOWN")
```

**设计要点**：
- 中心向量只负责**快速缩小范围**（粗排），并做第一层快速拒识（低于 `T_LOW` 直接返回 `UNKNOWN`）；
- **FAISS 全局检索**：Stage 2 不再逐个候选 speaker 分桶计算，而是直接调用 `_vector_index.search()` 在全部精选向量中做全局 Top-K 检索；
- **一致性校验是核心判决条件**：`topk_consistency()` 确保 Top-K 检索结果中多数（默认 5 选 4）指向**同一个 speaker**，防止因个别离群向量导致误判；
- 最终分数由一致性 speaker 的精选向量平均分决定，只有 `≥ T_HIGH` 才返回 `KNOWN`；
- `UNKNOWN` 结果由调用方决定是丢弃、暂存 pending_db，还是立即新建 speaker。

**验证标准**：已知说话人的新音频，识别准确率 > 90%；未知说话人，拒绝识别（不强行归类）。

---

### Phase 9: 动态更新与主流程封装（V2.2 调整）

**目标**：高置信度识别结果，自动更新声纹库；V2.2 新增输出字段传递。

| 文件 | 核心内容 |
|------|---------|
| `src/pipeline.py` | `build_pipeline(...)` 和 `recognize_pipeline(...)`；V2.2 调整：识别时把 `IdentificationResult.score` 回填到 `SpeakerSegment.score`；ffmpeg 预处理后的 `sr=16000` 回填到 `SpeakerSegment.sr` |

**`recognize_pipeline()` V2.2 变更点**：
1. **diarization 后**：`replace(seg, segment_id=..., file=..., sr=16000)` 注入 segment_id、file、sr；
2. **identify 后**：`seg.with_score(result.score)` 把余弦相似度回填；`UNKNOWN` 时 score 为 `null`；
3. **ASR 对齐后**：`precision` 字段暂留 `null`，待有参考文本时注入（见 Phase 11）。

**`SpeakerRepository` V2.3 内部变更**：
1. `_speakers` 存完整 `SpeakerData`（center, embedding_count, **embeddings**, **durations**, profile），精选向量回到 speaker_db；
2. **移除** `_vector_db` 精选向量存储，`VectorDb` 改为独立的全量原始向量库（中间数据），不由 `SpeakerRepository` 持有；
3. `build_from_pool()` → 精选向量直接写入 `SpeakerData`，再由 `speaker_db.embeddings` 重建 FAISS；
4. `save()` / `load()` → 只读写 `speaker_db:main`（JSON），识别库自包含，无需额外 vector_db 文件。

**验证标准**：多次识别同一人新音频后，该中心向量稳定性提升（同类样本 similarity 升高）；输出 JSON 包含全部 V2.2 字段。

---

### Phase 10: CLI 入口与测试（V2.2 调整）

**目标**：能用命令行跑通整个流程，输出包含全部 V2.2 字段。

| 文件 | 核心内容 |
|------|---------|
| `main.py` | `argparse` 支持两个子命令：`build`（建库）、`recognize`（识别）；V2.2 `recognize` 输出 JSON 包含 `segment_id`, `file`, `duration`, `sr`, `local_speaker`, `global_speaker`, `display_name`, `score`, `text`, `translation`, `precision`, `embedding` |
| `tests/` | 每个模块至少一个单元测试；`test_pipeline.py` 端到端测试；`tests/test_core_*.py` 覆盖 core 模块 |

**验证标准**：
- `python main.py build --embedding_dir ./embeddings/` 成功建库，且 `data/processed/speaker_db/` 下存在自包含的 `speaker_db/main.json`（含 center + 精选 embeddings），以及可选的全量原始向量 `vector_db/main.npz`（中间数据）
- `python main.py recognize --input ./new_audio.wav` 输出包含全部 V2.2 字段的 JSON

---

### Phase 11: 文本精度评估（precision 字段，已实现）

**目标**：为 `precision` 字段提供 CER-based 精度计算。

**实现方式**：
- `precision` 采用 **CER**（Character Error Rate）计算，而非 ASR 内部 confidence。
- 参考文本来源：自动从 SLR80 数据集的三个 CSV（train/dev/test）中按音频文件名匹配查找。当前仅限 SLR80 数据，未来可扩展 `--reference` 参数支持手动传入参考文本。
- 无匹配参考文本时，`precision` 留 `null`。
- 计算逻辑：
  1. 合并该音频所有 segment 的 `text` 得到完整识别文本；
  2. 用 Levenshtein 距离（Wagner-Fischer）计算与参考文本的 CER；
  3. `precision = round(max(0.0, 1.0 - cer), 4)`。

**实现位置**：`main.py` 中 `cmd_recognize()` 的收尾阶段（`_load_ground_truth_text` + `_compute_cer`）。

---

## 四、模块依赖关系

```
src/core/
  ├── types.py        <- 所有模块依赖（最底层）
  ├── pool.py         <- types
  ├── storage.py      <- types
  └── repository.py   <- types, pool, storage

业务模块依赖：
  src/audio/preprocess.py
         |
         v
  src/diarization/segment.py -> src/diarization/postprocess.py
         |                            |
         |                            v
         |                      list[SpeakerSegment]
         |                            |
         v                            v
  src/embedding/extractor.py -----> SpeakerSegment.with_embedding()
         |
         v
  EmbeddingPool (src/core/pool.py)
         |
         v
  src/clustering/cluster.py
         |
         v
  SpeakerRepository.build_from_pool() (src/core/repository.py)
         |
         +---> src/speaker_db/vector_index.py (FaissVectorIndex)
         |
         +---> src/asr/whisper_asr.py
         |           |
         |           v
         +---> src/asr/align.py <---------+
         |                                 |
         +---> src/pipeline.py <-----------+
         |                                 |
         +---> SpeakerRepository.identify()  (Phase 8)
         |                                 |
         +---> SpeakerRepository.update_speaker() (Phase 9)
```

**关键规则**：
- `src/core/` 不依赖任何业务模块（audio, diarization, embedding, asr 等）。
- 业务模块可以依赖 `src/core/`，但不可以互相交叉依赖（如 asr 不应直接 import diarization）。

---

## 五、数据集与验证策略

### 5.1 当前数据集（SLR80）

- **来源**：OpenSLR SLR80 缅甸语 ASR 数据集
- **性质**：单说话人朗读数据（20 位女性说话人，2528 条公开语音，约 4 小时）
- **当前位置**：`data/raw/burmese_asr/`（根目录）
- **真实 speaker 标识**：文件名形如 `bur_0366_5281755035.wav`，其中 `0366` 是 speaker id；构建 SLR80 基准声纹库时必须使用该前缀作为 `global_speaker`。
- **用途**：
  - **ASR 准确率测试**（WER/CER）
  - **翻译质量验证**（缅甸语 → 中文，人工抽检）
  - **声纹提取通路验证**（代码能跑通，但单说话人文件 trivial）
  - **无法验证 diarization**（没有多人对话场景）
  - **可验证“已知说话人识别”基准**：用 train 按文件名前缀建 20 人库，用 dev/test 做闭集识别评估；但仍无法验证真实多人对话 diarization。

**2026-05-16 真实测试发现**：
- raw split 数量与 CSV 对齐：train/dev/test 分别为 2330/100/100 个 wav。
- `missing_train.txt`、`missing_dev.txt`、`missing_test.txt` 当前列出的文件实际存在，这些文件是错误或过期诊断产物，不能再作为缺失文件依据。
- 使用默认无监督聚类从 train embeddings 建库时，2528 个 segments 被聚成 681 个 speaker；这与 SLR80 真实 20 个 speaker 不符，该库应判定为无效。
- 使用文件名前缀建 20 人库后，dev/test 闭集识别结果为：dev 88/102（86.27%），test 92/105（87.62%）；高置信结果未出现误识，失败主要是低置信拒识。
- 已新增 `slr80_filename` 标签策略：SLR80 建库应使用 `--label_strategy slr80_filename`，通用未知数据才使用 `cluster`。

### 5.2 多人对话数据（后续准备）

- **需求**：真实的多说话人对话/会议录音
- **用途**：验证完整 Pipeline（diarization + speaker ID + ASR + 翻译）
- **存放位置**：`data/raw/multi_speaker_dialogue/`（目录已预留）

---

## 六、风险与难点

| 风险 | 级别 | 应对 |
|------|------|------|
| Pyannote / SpeechBrain 模型下载慢或需 huggingface token | 中 | 提前配置 token，使用国内镜像或本地缓存 |
| FAISS 在 Windows 上安装困难 | 中 | 用 `faiss-cpu`，如仍失败换预编译包 |
| 聚类 threshold 调参困难 | 高 | **已解决**：采用两层聚类策略，第一层保守阈值（0.30）保纯度，第二层 centroid 合并（0.35）减碎片；不设置 `min_cluster_size`，避免新 speaker 因片段不足被丢弃；SLR80 实测最佳参数 `thr=0.30, cthr=0.35` |
| Whisper 缅甸语 ASR 准确率不足 | 高 | Whisper 对缅甸语支持较弱，可能需要 fine-tune 或使用专用缅甸语 ASR 模型 |
| 缅甸语 → 中文翻译模型稀缺 | 中 | 优先尝试 `Helsinki-NLP/opus-mt-my-zh` 或 mBART，如效果差再调 API |
| 长音频（>1小时）内存爆炸 | 中 | 分段处理，流式读取 |
| Whisper + Pyannote 同时加载，GPU 显存不足 | 中 | 支持 CPU fallback，或分批加载模型 |

---

## 七、关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLE_RATE` | 16000 | 音频采样率 |
| `SAMPLE_FMT` | s16 | 16bit 位宽 |
| `CHANNELS` | 1 | 单声道 |
| `MIN_DURATION` | 1.0 | 最小语音片段长度（秒） |
| `MIN_DURATION_ON` | 0.3 | 最短语音段，低于则删除 |
| `MIN_DURATION_OFF` | 0.2 | 最短静音段，低于则合并 |
| `EMBEDDING_DIM` | 192 | 声纹向量维度（ECAPA-TDNN） |
| `DISTANCE_THRESHOLD` | 0.30 | 第一层聚类距离阈值（cosine distance），保守值，保证纯度优先 |
| `CENTROID_THRESHOLD` | 0.35 | 第二层 centroid 合并阈值（cosine distance）；同一 speaker 的多个子簇 centroid 距离低于此值时合并 |
| `MIN_CLUSTER_SIZE` | — | **已废弃**。不设置最小聚类大小，第一层严格聚类产生的碎片由第二层 centroid 合并自然消解，避免新 speaker 因片段不足被丢弃 |
| `DEDUP_THRESHOLD` | 0.95 | 去重相似度阈值 |
| `MAX_EMB` | 20 | 每个 speaker 最大保留 embedding 数 |
| `TOPK` | 5 | FAISS 检索 TopK |
| `TOPK_CONSISTENCY_MIN` | 4 | TopK 一致性最小通过数（5 个中至少 4 个一致） |
| `T_LOW` | **0.50** | 第一层 center 粗排拒识阈值；若最像的 center 相似度低于此值，直接判定为 UNKNOWN，不进入 FAISS 精排；经 leave-out 测试，此值可挡掉 69-97% 陌生人查询，同时库内误拒率 < 0.6% |
| `T_HIGH` | **0.65** | 第二层精排高置信度确认阈值；FAISS Top-K 平均相似度 ≥ 此值且 consistency 通过才返回 KNOWN；原值 0.70 经测试会误拒 7-15% 已知人，0.65 误拒率降至 3-6%，同时陌生人误识率接近 0% |
| `TARGET_RMS` | 0.1 | RMS 归一化目标值 |
| `MIN_WHISPER_SEGMENT` | 0.5 | Whisper 片段 <0.5s 标记为 IGNORE |
| `UPDATE_MIN_DURATION` | 2.0 | 动态更新时，<2s 的片段不用于更新声纹库 |
| `UPDATE_DEDUP_THRESHOLD` | 0.95 | 动态更新去重阈值 |

---

## 八、工程目录结构

```
voice-detection/
|-- config.py                          # 全局参数
|-- requirements.txt
|-- pytest.ini
|-- main.py                            # CLI 入口（build / recognize）
|-- pipeline.py                        # 主流程封装
|
|-- scripts/                           # 批量工具脚本
|   |-- build_embedding_pool.py       # 批量：wav → diarization → embedding → NpzStorage
|
|-- data/
|   |-- raw/
|   |   |-- burmese_asr/              # SLR80 数据集
|   |   |-- multi_speaker_dialogue/    # 多说话人数据（后续添加）
|   |-- processed/                     # 处理后的中间数据
|   |   |-- burmese_asr/
|   |   |   |-- wav/                  # 标准化后的 wav
|   |   |   |-- segments/             # diarization 结果缓存（JSON，通过 JsonStorage）
|   |   |   |-- embeddings/           # 提取的 embedding（NPZ，通过 NpzStorage）
|   |-- output/                        # 最终输出
|   |   |-- results/                  # 识别结果 JSON
|
|-- models/                            # 预训练模型本地缓存
|   |-- pyannote/
|   |-- speechbrain/
|   |-- whisper/
|   |-- translation/
|
|-- src/
|   |-- core/                          # 【新增】核心数据结构与 seams
|   |   |-- __init__.py
|   |   |-- types.py                   # SpeakerSegment, SpeakerProfile, SpeakerData 等 dataclass
|   |   |-- pool.py                    # EmbeddingPool（替代裸 dict embedding_pool）
|   |   |-- storage.py                 # Storage Protocol + Json/Npz/Pickle/Memory Storage
|   |   |-- repository.py              # SpeakerRepository + VectorIndex Protocol
|   |   |-- pipeline.py                # 【已清理】原 Stage Protocol + Pipeline 死代码已删除
|   |-- audio/
|   |   |-- preprocess.py
|   |   |-- __init__.py
|   |-- diarization/
|   |   |-- segment.py
|   |   |-- postprocess.py
|   |   |-- __init__.py
|   |   |-- cache.py                   # 【废弃，逐步替换为 core/storage.py】
|   |-- embedding/
|   |   |-- extractor.py
|   |   |-- normalize.py
|   |   |-- __init__.py
|   |   |-- cache.py                   # 【废弃，逐步替换为 core/storage.py】
|   |-- clustering/
|   |   |-- pool.py                    # 【新增】从 NpzStorage + JsonStorage 加载 EmbeddingPool
|   |   |-- cluster.py                 # 聚类 + 去重 + 分配 SPK_ID
|   |   |-- __init__.py
|   |-- speaker_db/
|   |   |-- vector_index.py            # FaissVectorIndex 实现；VectorIndex 保留为 Protocol 别名
|   |   |-- __init__.py
|   |   |-- builder.py                 # 【废弃，功能并入 repository.py】
|   |   |-- profile.py                 # 【废弃，功能并入 repository.py】
|   |   |-- updater.py                 # 【废弃，功能并入 repository.py】
|   |   |-- storage.py                 # 【废弃，功能并入 core/storage.py】
|   |-- asr/
|   |   |-- whisper_asr.py
|   |   |-- align.py
|   |   |-- __init__.py
|   |-- translation/
|   |   |-- translator.py
|   |   |-- __init__.py
|   |-- recognition/
|   |   |-- identify.py                # 【废弃，功能并入 repository.py】
|   |   |-- verify.py                  # TopK consistency 判断
|   |   |-- __init__.py
|
|-- tests/
|   |-- test_core_types.py             # 【新增】SpeakerSegment, SpeakerProfile 不可变性测试
|   |-- test_core_pool.py              # 【新增】EmbeddingPool 懒构建、apply_labels 测试
|   |-- test_core_storage.py           # 【新增】Json/Npz/Memory Storage 测试
|   |-- test_phase4_10_compat.py       # 【新增】Phase 4-10 兼容性与集成测试
|   |-- test_audio.py
|   |-- test_diarization.py
|   |-- test_embedding.py
|   |-- test_clustering.py
|   |-- test_speaker_db.py
|   |-- test_asr.py
|   |-- test_translation.py
|   |-- test_recognition.py
|   |-- test_pipeline.py
|
|-- IMPLEMENTATION_PLAN.md
```

### 废弃文件说明

| 旧文件 | 替代方案 | 迁移策略 |
|--------|---------|---------|
| `diarization/cache.py` | `core/storage.py` 的 `JsonStorage` | 功能已迁移至 `JsonStorage`；旧文件保留仅作向后兼容参考 |
| `embedding/cache.py` | `core/storage.py` 的 `NpzStorage` | 功能已迁移至 `NpzStorage`；旧文件保留仅作向后兼容参考 |
| `speaker_db/storage.py` | `core/storage.py` 的 `PickleStorage` / `JsonStorage` | 功能已迁移至 `core/storage.py` |
| `speaker_db/builder.py` | `core/repository.py` 的 `SpeakerRepository.build_from_pool()` | 功能已迁移至 Repository |
| `speaker_db/profile.py` | `core/repository.py` 的 `SpeakerRepository.assign_name()` | 功能已迁移至 Repository |
| `speaker_db/updater.py` | `core/repository.py` 的 `SpeakerRepository.update_speaker()` | 功能已迁移至 Repository |
| `recognition/identify.py` | `core/repository.py` 的 `SpeakerRepository.identify()` | 功能已迁移至 Repository |
| `src/core/pipeline.py` 的 `Stage`/`Pipeline` | 已删除 | 死代码，无替代；业务流直接由 `pipeline.py` 中的 `build_pipeline` / `recognize_pipeline` 函数承担 |

---

## 九、模型与依赖

| 组件 | 模型/库 | 来源 |
|------|---------|------|
| 说话人分段 | pyannote/speaker-diarization | HuggingFace |
| 声纹提取 | speechbrain/spkrec-ecapa-voxceleb | HuggingFace |
| 语音识别 | openai-whisper (base/small) | OpenAI |
| 翻译 | Helsinki-NLP/opus-mt-my-zh 或 mBART | HuggingFace |
| 向量检索 | faiss-cpu | Facebook Research |
| 聚类 | sklearn AgglomerativeClustering | scikit-learn |

---

*计划版本：V2.3*
*重构日期：2026-05-18*
*变更：所有裸 dict 数据结构封装为不可变 dataclass；新增 src/core/ 统一 seams；全局聚类升级为两层策略（保守阈值 + centroid 合并，取消 min_cluster_size）；识别流程升级为两层比对 + FAISS 全局一致性校验（center 粗排 → FAISS Top-K 检索 → `topk_consistency` 5 选 4）；新增 `T_LOW` 第一层拒识阈值；speaker_db / vector_db 持久化拆分。*
