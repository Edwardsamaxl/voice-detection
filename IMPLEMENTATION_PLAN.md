# 语音处理与声纹识别系统 — 实现计划（V2.1 封装版）

> **目标读者**：实现 Agent。本文档基于与负责人的需求对齐结果编写，实现时请严格遵循本文档，如有歧义优先以本文档为准。
> **版本**：V2.1（2026-05-16）— 将所有裸 dict 数据结构封装为不可变 dataclass 与类接口。

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

### 1.2 输出示例

```json
[
  {
    "start": 0.0,
    "end": 3.2,
    "speaker": "SPK_0",
    "display_name": "张三",
    "text": "မင်္ဂလာပါ",
    "translation": "你好"
  },
  {
    "start": 3.5,
    "end": 7.1,
    "speaker": "SPK_1",
    "display_name": "UNKNOWN",
    "text": "ခင်ဗျား နာမည် ဘယ်လို ခေါ် လဲ",
    "translation": "你叫什么名字"
  }
]
```

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
  ├── repository.py     → SpeakerRepository (统一 seam: speaker_db + vector_db + profile)
  └── pipeline.py       → 【已清理】原 Stage Protocol + Pipeline 死代码已删除

数据流：

EmbeddingPool（全量原始数据，封装后）
        ↓
SpeakerRepository.build_from_pool()
        ↓
  ├─→ speaker_db（聚类后）   → 每人一个 weighted center
  ├─→ vector_db（FAISS）     → 每人多个精选 embedding（5~20 个）
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
    text: str | None = None
    translation: str | None = None

    @property
    def duration(self) -> float: ...
    def with_embedding(self, emb) -> SpeakerSegment: ...
    def with_global_speaker(self, spk_id) -> SpeakerSegment: ...
    def with_text(self, text, translation=None) -> SpeakerSegment: ...
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

**（3）SpeakerData（聚类后中心向量）**

```python
@dataclass(frozen=True, slots=True)
class SpeakerData:
    spk_id: str
    center: np.ndarray          # 加权平均 + L2 归一化后的中心向量
    embeddings: list[np.ndarray] # 全量原始 embeddings（用于动态更新时重新计算 center）
    durations: list[float] | None = None  # 各 embedding 对应的片段时长（用于加权 center）
    profile: SpeakerProfile | None = None
```

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

**（5）SpeakerRepository（统一 seam）**

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
    def save(self, key: str = "speaker_db:main") -> None: ...
    def load(self, key: str = "speaker_db:main") -> None: ...
```

> **关键设计**：`SpeakerRepository` 是**唯一对外 seam**。内部协调 `speaker_db`（dict）、`vector_db`（FAISS）、`speaker_profile`（dict）三者的一致性。调用者无需知道内部有三个存储。

**（6）VectorIndex（FAISS 适配 seam）**

```python
class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray, labels: list[str]) -> None: ...
    def search(self, query: np.ndarray, topk: int = 5) -> list[SearchResult]: ...
    def add(self, vector: np.ndarray, label: str) -> None: ...
    def rebuild(self) -> None: ...
```

> 具体实现为 `FaissVectorIndex`（`src/speaker_db/vector_index.py`）。`repository.py` 中保留 `VectorIndex` 作为 Protocol 别名，便于后续替换为 `IndexIVFFlat` 或其他库（Milvus / Qdrant）而无需修改 `SpeakerRepository`。

**（7）Storage（统一持久化 seam）**

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

> **统一 key 命名**：`"segments:bur_3260"`、`"embeddings:bur_3260"`、`"speaker_db:main"`、`"profiles:main"`

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
| `src/clustering/cluster.py` | **两层聚类**：`agg_clustering(embeddings, distance_threshold=0.3, centroid_threshold=0.35, min_cluster_size=3) -> labels`。第一层用保守阈值做 `AgglomerativeClustering`（保证纯度），第二层按类 centroid 合并相近的簇（减少碎片化）；去重用矩阵向量化 dot product；`pool.apply_labels(labels)` 回写 |

**关键变更**：
- 批量提取后的中间数据通过 `NpzStorage` 持久化，key 如 `"embeddings:bur_3260"`
- `cluster.py` 操作的是 `EmbeddingPool` 对象，聚类后通过 `pool.apply_labels(labels)` 返回**新 pool**
- 不再直接操作裸 `embedding_pool["segments"][i]["global_speaker"]`

**验证标准**：
1. 同一人在 A.wav 和 B.wav 中的片段，被分配到同一个 cluster label；两层聚类不要求所有片段被第一层直接合并，允许经第二层 centroid 合并后归入同一 speaker。
2. 回写后，`EmbeddingPool.filter_by_speaker("SPK_0")` 能正确返回该 speaker 的所有 segments。

**2026-05-16 聚类策略与实测结论**：
- SLR80 train（2330 条，20 位女性说话人）真实测试：`distance_threshold=0.30` 单层聚类产生 562 个类（100% 纯度），`0.55` 产生 24 个类但纯度降至 97.2%。
- 采用两层聚类 `thr=0.30, cthr=0.35, min_cluster_size=3` 后，类数降至 21，纯度 98.8%。这是当前数据上**泛化与精度最平衡的参数**。
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

**双阈值判断逻辑**（在 `SpeakerRepository.identify()` 内部实现）：
```python
results = self._vector_index.search(query_emb, topk=TOPK)
if topk_consistent(results) and results[0].score >= T_HIGH:
    -> 高置信，返回现有 speaker
else:
    -> 低置信 / 未知，由调用方决定新建 Speaker
```

**验证标准**：已知说话人的新音频，识别准确率 > 90%；未知说话人，拒绝识别（不强行归类）。

---

### Phase 9: 动态更新与主流程封装

**目标**：高置信度识别结果，自动更新声纹库。

| 文件 | 核心内容 |
|------|---------|
| `src/pipeline.py` | `build_pipeline(embedding_dir, label_strategy) -> SpeakerRepository`：建库主流程；`recognize_pipeline(new_audio, repo, extractor, ...) -> list[SpeakerSegment]`：识别主流程（已过滤 IGNORE 并回填 display_name） |

**`SpeakerRepository.update_speaker()` 逻辑**：
1. **质量过滤**：`duration < UPDATE_MIN_DURATION` → skip；`cosine_similarity(new_emb, center) > UPDATE_DEDUP_THRESHOLD` → skip
2. **更新 `speaker_db`**：新 embedding 加入全量原始数据，重新计算加权 center
3. **更新 `vector_db`**：新 embedding 加入 FAISS 精选向量池（每人最多 `MAX_EMB` 个），超限时删除离 center 最远的 embedding
4. **重建索引**：调用 `VectorIndex.rebuild()`

**验证标准**：多次识别同一人新音频后，该中心向量稳定性提升（同类样本 similarity 升高）。

---

### Phase 10: CLI 入口与测试

**目标**：能用命令行跑通整个流程。

| 文件 | 核心内容 |
|------|---------|
| `main.py` | `argparse` 支持两个子命令：`build`（建库）、`recognize`（识别） |
| `tests/` | 每个模块至少一个单元测试；`test_pipeline.py` 端到端测试；`tests/test_core_*.py` 覆盖 core 模块 |

**验证标准**：
- `python main.py build --input_dir ./audio_samples/` 成功建库
- `python main.py recognize --input ./new_audio.wav` 输出带说话人标签、缅甸语原文、中文翻译的 JSON

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
| 聚类 threshold 调参困难 | 高 | **已解决**：采用两层聚类策略，第一层保守阈值（0.30）保纯度，第二层 centroid 合并（0.35）减碎片；不追求 100% 纯度以避免过拟合到当前数据集统计特性；SLR80 实测最佳参数 `thr=0.30, cthr=0.35, min=3` |
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
| `MIN_CLUSTER_SIZE` | 3 | 第二层最小类大小，低于此值的类被视为噪声，合并到最近的合法类 |
| `DEDUP_THRESHOLD` | 0.95 | 去重相似度阈值 |
| `MAX_EMB` | 20 | 每个 speaker 最大保留 embedding 数 |
| `TOPK` | 5 | FAISS 检索 TopK |
| `TOPK_CONSISTENCY_MIN` | 4 | TopK 一致性最小通过数（5 个中至少 4 个一致） |
| `T_HIGH` | 0.7 | 高置信度阈值 |
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

*计划版本：V2.1*
*重构日期：2026-05-16*
*变更：所有裸 dict 数据结构封装为不可变 dataclass；新增 src/core/ 统一 seams；全局聚类升级为两层策略（保守阈值 + centroid 合并）。*
