# 语音处理与声纹识别系统 — 实现计划（Prompt 版）

> **目标读者**：实现 Agent。本文档基于与负责人的需求对齐结果编写，实现时请严格遵循本文档，如有歧义优先以本文档为准。

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

### 2.1 最优架构（三层）

```
embedding_pool（全量原始数据）    → 原始数据，不进 FAISS
        ↓
speaker_db（聚类后）             → 每人一个 weighted center
        ↓
vector_db（FAISS）               → 每人多个精选 embedding（5~20 个）
        ↓
speaker_profile（人名表）         → 显示层（SPK_ID → 姓名）
        ↓
data_store（结果缓存）           → 业务数据（分段 + speaker + text + translation）
```

### 2.2 关键数据结构

**（1）embedding_pool（分层结构）**

```python
embedding_pool = {
    "segments": [
        {
            "embedding": np.array([...]),   # 192 维声纹向量
            "file": "A.wav",
            "segment_id": "A_0001",         # 唯一标识
            "start": 0.0,
            "end": 3.2,
            "duration": 3.2,
            "local_speaker": "speaker_0",   # pyannote 内部分段标签
            "global_speaker": "SPK_0",      # 聚类后统一 ID
            "quality": 0.92                 # 可选质量评分
        }
    ],
    "matrix": np.ndarray,                   # (N, 192) 向量矩阵，用于向量化计算
    "meta_index": {"A_0001": 0, ...}       # segment_id → index，快速定位
}
```

**（2）speaker_db（聚类后中心向量）**

```python
speaker_db = {
    "SPK_0": {
        "center": np.array([...]),          # 加权平均 + L2 归一化后的中心向量
        "count": 12,                        # 该 speaker 的 embedding 数量
        "embeddings": [emb1, emb2, ...]     # 该 speaker 的全量原始 embeddings（用于动态更新时重新计算 center）
    }
}
```

**（3）speaker_profile（人名元数据表）**

```python
speaker_profile = {
    "SPK_0": {
        "name": "张三",
        "alias": ["老张", "Zhang San"],
        "gender": "male",
        "notes": "客户经理",
        "created_at": "2026-04-01"
    },
    "SPK_1": {
        "name": "UNKNOWN",
        "alias": []
    }
}
```

**（4）vector_db（FAISS 索引）**

```python
vector_db = {
    "index": faiss.IndexFlatIP(dim),        # 内积索引（归一化后等价于 cosine）
    "labels": ["SPK_0", "SPK_0", "SPK_1", ...]  # 每个向量对应的 speaker ID
}
```

> **关键设计**：FAISS 中采用**多向量表示**，每个 speaker 保留最多 `MAX_EMB=20` 个精选 embedding，而不是只存 1 个 center。这是为了支撑 TopK 一致性判断。

**（5）data_store（业务结果缓存）**

```python
data_store = {
    "file1.wav": [
        {
            "start": 0.0,
            "end": 3.2,
            "duration": 3.2,
            "local_speaker": "A",
            "global_speaker": "SPK_0",
            "display_name": "张三",
            "embedding": [...],
            "text": "မင်္ဂလာပါ",
            "translation": "你好"
        }
    ]
}
```

---

## 三、分阶段实现计划

### Phase 1: 项目骨架与音频预处理

**目标**：能跑通音频标准化，输出 16kHz/16bit/单声道 wav。

| 文件 | 核心内容 |
|------|---------|
| `config.py` | 所有常量（见第 8 节参数表） |
| `audio/preprocess.py` | `convert_to_wav(input_path, output_path)` FFmpeg 封装；`load_wav(wav_path) -> tuple[np.ndarray, int]`；`crop_segment(wav_array, sr, start, end) -> np.ndarray`；`rms_normalize(wav_array, target_rms=0.1)` |
| `requirements.txt` | `torch`, `pyannote.audio`, `speechbrain`, `scikit-learn`, `faiss-cpu`, `openai-whisper`, `numpy`, `scipy`, `soundfile`, `pytest`, `pytest-cov` + 翻译模型依赖 |

**验证标准**：输入任意格式音频，输出符合规范的 wav。

---

### Phase 2: 说话人分段

**目标**：输入 wav，输出带时间戳的说话人片段。

| 文件 | 核心内容 |
|------|---------|
| `diarization/segment.py` | `run_diarization(wav_path, model_name="pyannote/speaker-diarization-3.1", token=None, device=None) -> Annotation`；封装 `pyannote/speaker-diarization` |
| `diarization/postprocess.py` | `merge_short_segments(annotation, min_duration_on=0.3, min_duration_off=0.2)`；`annotation_to_segments(annotation, min_duration=1.0) -> list[dict]`（Annotation 转 segments，含预留字段）；`rename_labels(annotation, label_mapping) -> Annotation` |
| `diarization/cache.py` | `save_segments(file_path, segments)` / `load_segments(file_path)`，JSON 格式缓存 |

**数据结构**：
```python
segments = [
    {
        "start": 0.0,
        "end": 3.2,
        "duration": 3.2,
        "local_speaker": "A",
        "global_speaker": None,
        "display_name": None,
        "embedding": None,
        "text": None,
        "translation": None,
    }
]
```
> 字段说明：`global_speaker`, `display_name`, `embedding`, `text`, `translation` 为下游模块预留，初始为 `None`。

**验证标准**：一段多人对话音频，能正确切分出不同说话人的时间段。

---

### Phase 3: 声纹向量提取与归一化

**目标**：从每个语音片段提取 192 维 embedding。

| 文件 | 核心内容 |
|------|---------|
| `embedding/extractor.py` | `class EmbeddingExtractor`：初始化 SpeechBrain 预训练模型；`extract(wav_array, sr=16000) -> np.ndarray(192,)`；`extract_segments(wav_path, segments) -> list[dict]`（批量提取） |
| `embedding/normalize.py` | `l2_normalize(emb) -> emb / np.linalg.norm(emb)` |
| `embedding/cache.py` | `save_embeddings(file_path, segments)` / `load_embeddings(file_path)`：将带 embedding 的 segments 存为 `.npz` + `.json`；`list_embedding_files(dir_path)` |

**验证标准**：同一人的两个片段，cosine similarity > 0.85；不同人 < 0.5。

---

### Phase 4: 全局聚类、去重与 Speaker ID 回写

**目标**：跨文件聚合同一说话人，生成全局 SPK_ID，并将结果回写到数据结构。

**前置步骤**：先运行 `scripts/build_embedding_pool.py` 批量提取 embedding，得到按文件保存的 `.npz` 缓存。

| 文件 | 核心内容 |
|------|---------|
| `clustering/pool.py` | `build_embedding_pool(npz_dir) -> embedding_pool`：从所有 `.npz` 文件加载，构建全局分层结构（`segments` + `matrix` + `meta_index`） |
| `clustering/cluster.py` | 聚类 + 去重 + SPK_ID 回写。`agg_clustering(embeddings, distance_threshold=0.3) -> labels`；去重（cosine > 0.95）用矩阵向量化 dot product；将 `labels` 映射为 `SPK_{label}` 回写到 `embedding_pool.segments[i].global_speaker` |

**验证标准**：
1. 同一人在 A.wav 和 B.wav 中的片段，被分配到同一个 cluster label。
2. 回写后，`embedding_pool` 中的 `global_speaker` 字段已填充。

---

### Phase 5: Speaker 库与向量索引

**目标**：建立可快速检索的声纹库，支持多向量表示。

| 文件 | 核心内容 |
|------|---------|
| `speaker_db/builder.py` | `build_speaker_db(embedding_pool) -> speaker_db`：按 global_speaker 分组，加权平均（时长做权重）计算 center，L2 归一化；去异常 embedding；**保留全量原始 embeddings**（用于后续动态更新重新计算 center） |
| `speaker_db/profile.py` | 管理 `speaker_profile`；`assign_name(spk_id, name)` |
| `speaker_db/storage.py` | `save_speaker_db(db, path)` / `load_speaker_db(path)`；pickle / numpy 格式 |
| `speaker_db/vector_index.py` | `class VectorIndex`：`build(vectors, spk_ids) -> faiss.IndexFlatIP`；`search(query_emb, topk=5) -> results`；`add(new_vector, spk_id)`；`rebuild()` |

**关键设计**：
- FAISS 索引中每个 speaker 存入最多 `MAX_EMB=20` 个精选 embedding（不是只存 1 个 center）。
- 小规模（<10 万向量）使用 `IndexFlatIP`。

**验证标准**：100 个 speaker 的多向量索引，FAISS 检索耗时 < 10ms。

---

### Phase 6: 语音识别（ASR）与翻译

**目标**：Whisper 识别缅甸语原文，再翻译成中文。

| 文件 | 核心内容 |
|------|---------|
| `asr/whisper_asr.py` | `transcribe(wav_path, language="my") -> {"segments": [{"start", "end", "text"}]}` |
| `translation/translator.py` | **【新增】**缅甸语 → 中文翻译模块。加载本地 NMT 模型（如 `Helsinki-NLP/opus-mt-my-zh` 或 mBART），输入缅甸语句子，输出中文翻译。预留 API fallback 接口。 |

**注意**：翻译模块需离线可用，优先加载本地模型。如本地模型效果不佳，可预留 API 接口（Google Translate / 百度翻译）作为 fallback。

---

### Phase 7: 结果整合与说话人对齐

**目标**：将 Whisper 的 ASR 结果、翻译结果，与 diarization 的说话人时间段对齐。

| 文件 | 核心内容 |
|------|---------|
| `asr/align.py` | `assign_speaker_to_segments(whisper_segments, diarization_annotation, translations) -> list[dict]` |

**对齐逻辑**：
1. 对每条 Whisper segment，用 `annotation.crop(Segment(start, end)).argmax()` 找占比最大的 speaker。
2. **【新增】边界处理**：
   - 如果 `crop` 结果为空 → `speaker = "UNKNOWN"`
   - 如果 `end - start < 0.5` → `speaker = "IGNORE"`（片段太短，speaker 归属不可靠）
3. 将对应的中文翻译挂在该 segment 上。

**输出数据结构**：
```python
result = {
  "file1.wav": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "SPK_0",
      "display_name": "张三",
      "text": "မင်္ဂလာပါ",
      "translation": "你好"
    }
  ]
}
```

**验证标准**：识别的文本能正确归属到对应的 global_speaker，翻译结果语义通顺。

---

### Phase 8: 新音频识别流程

**目标**：输入新音频，识别其中说话人是谁，或判断为未知。

| 文件 | 核心内容 |
|------|---------|
| `recognition/identify.py` | `identify_speaker(segment_emb, vector_index, speaker_db) -> {"speaker", "score", "confidence"}` |
| `recognition/verify.py` | `topk_consistency(topk_results, k=5, min_consistency=4) -> bool`：**5 个中至少 4 个一致**才算通过；支持占比判断 |

**双阈值判断逻辑**：
```python
if score >= T_high(0.7) and topk_consistent:
    → 高置信，更新现有 speaker
else:
    → 低置信 / 未知，新建 Speaker（分配新 SPK_ID，加入向量库）
```

**验证标准**：已知说话人的新音频，识别准确率 > 90%；未知说话人，拒绝识别（不强行归类）。

---

### Phase 9: 动态更新与主流程封装

**目标**：高置信度识别结果，自动更新声纹库。

| 文件 | 核心内容 |
|------|---------|
| `speaker_db/updater.py` | `update_speaker_embedding(spk_id, new_emb, duration, max_emb=20)`：
1. **质量过滤**：`duration < 2.0` → skip；`cosine_similarity(new_emb, mean_emb) > 0.95` → skip（去重）
2. **更新 `speaker_db`**：新 embedding 加入 `speaker_db[spk_id]["embeddings"]`（全量原始数据），用全量 embeddings 重新计算加权 center
3. **更新 `vector_db`**：新 embedding 加入 FAISS 精选向量池（每人最多 `MAX_EMB=20` 个），超限时删除与 `center` cosine similarity 最小（离中心最远）的 embedding
4. **重建索引**：重新构建 FAISS `IndexFlatIP` 索引
- **【预留】**`quality_score` 接口（暂不实现，预留参数位） |
| `pipeline.py` | `build_pipeline(audio_files) -> data_store`：建库主流程；`recognize_pipeline(new_audio) -> results`：识别主流程 |

**验证标准**：多次识别同一人新音频后，该中心向量稳定性提升（同类样本 similarity 升高）。

---

### Phase 10: CLI 入口与测试

**目标**：能用命令行跑通整个流程。

| 文件 | 核心内容 |
|------|---------|
| `main.py` | `argparse` 支持两个子命令：`build`（建库）、`recognize`（识别） |
| `tests/` | 每个模块至少一个单元测试；`test_pipeline.py` 端到端测试 |

**验证标准**：
- `python main.py build --input_dir ./audio_samples/` 成功建库
- `python main.py recognize --input ./new_audio.wav` 输出带说话人标签、缅甸语原文、中文翻译的 JSON

---

## 四、模块依赖关系

```
audio/preprocess.py
       |
       v
diarization/segment.py -> diarization/postprocess.py
       |
       v
embedding/extractor.py -> embedding/normalize.py
       |
       v
clustering/pool.py -> clustering/cluster.py
       |
       v
speaker_db/builder.py -> speaker_db/vector_index.py
       |                     |
       v                     v
asr/whisper_asr.py    translation/translator.py
       |                     |
       +-------> asr/align.py <---------+
       |                               |
       +-------> pipeline.py <---------+
       |                               |
       +-------> recognition/identify.py -> recognition/verify.py
       |                               |
       +-------> speaker_db/updater.py  |
```

---

## 五、数据集与验证策略

### 5.1 当前数据集（SLR80）

- **来源**：OpenSLR SLR80 缅甸语 ASR 数据集
- **性质**：单说话人朗读数据（20 位女性说话人，~2530 条，4 小时）
- **当前位置**：`speech_asr_slr80_my_trainsets/`（根目录）
- **用途**：
  - ✅ **ASR 准确率测试**（WER/CER）
  - ✅ **翻译质量验证**（缅甸语 → 中文，人工抽检）
  - ⚠️ **声纹提取通路验证**（代码能跑通，但单说话人文件 trivial）
  - ❌ **无法验证 diarization**（没有多人对话场景）
  - ❌ **无法真实验证声纹识别准确率**（缺乏多人对话测试数据）

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
| 聚类 threshold 调参困难 | 高 | 先用 0.3 默认值，预留配置文件可调，后期用标注数据校准 |
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
| `DISTANCE_THRESHOLD` | 0.3 | 聚类距离阈值（cosine distance），对应 similarity 0.7 |
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
|   |-- build_embedding_pool.py       # 批量：wav → diarization → embedding
|
|-- data/
|   |-- raw/
|   |   |-- burmese_asr/              # SLR80 数据集
|   |   |-- multi_speaker_dialogue/    # 多说话人数据（后续添加）
|   |-- processed/                     # 处理后的中间数据
|   |   |-- burmese_asr/
|   |   |   |-- wav/                  # 标准化后的 wav
|   |   |   |-- segments/             # diarization 结果缓存（JSON）
|   |   |   |-- embeddings/           # 提取的 embedding（npz + json）
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
|   |-- audio/
|   |   |-- preprocess.py
|   |   |-- __init__.py
|   |-- diarization/
|   |   |-- segment.py
|   |   |-- postprocess.py
|   |   |-- cache.py                   # save_segments / load_segments
|   |   |-- __init__.py
|   |-- embedding/
|   |   |-- extractor.py
|   |   |-- normalize.py
|   |   |-- cache.py                   # save_embeddings / load_embeddings
|   |   |-- __init__.py
|   |-- clustering/
|   |   |-- pool.py                    # 从 npz 构建全局 embedding_pool
|   |   |-- cluster.py                 # 聚类 + 去重 + 分配 SPK_ID
|   |   |-- __init__.py
|   |-- speaker_db/
|   |   |-- builder.py
|   |   |-- profile.py
|   |   |-- storage.py
|   |   |-- vector_index.py
|   |   |-- updater.py
|   |   |-- __init__.py
|   |-- asr/
|   |   |-- whisper_asr.py
|   |   |-- align.py
|   |   |-- __init__.py
|   |-- translation/
|   |   |-- translator.py              # 缅甸语 → 中文翻译
|   |   |-- __init__.py
|   |-- recognition/
|   |   |-- identify.py
|   |   |-- verify.py
|   |   |-- __init__.py
|
|-- tests/
|   |-- test_audio.py
|   |-- test_diarization.py
|   |-- test_embedding.py
|   |-- test_embedding_cache.py
|   |-- test_clustering.py
|   |-- test_speaker_db.py
|   |-- test_asr.py
|   |-- test_translation.py
|   |-- test_recognition.py
|   |-- test_pipeline.py
|
|-- IMPLEMENTATION_PLAN.md
```

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

*计划重构日期：2026-05-14*
*基于 grill-me 需求对齐结果*
