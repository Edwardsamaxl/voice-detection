# Voice Detection

说话人分割与声纹识别流水线，支持缅甸语语音处理。

- **说话人分离**（Diarization）—— pyannote.audio
- **声纹 Embedding 提取** —— SpeechBrain ECAPA-TDNN
- **说话人聚类与识别** —— FAISS + 自定义阈值策略
- **语音识别**（ASR）—— UniASR（缅甸语专用）
- **文本对齐与翻译** —— NLLB-200（缅甸语 → 简体中文）

## 快速开始

请使用项目级虚拟环境，**不要**安装到全局 Python 环境。

### 1. 环境准备

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchaudio==2.4.1
python -m pip install -r requirements.txt -c constraints-win-cpu.txt
```

### 2. 环境检查

```bash
python scripts/doctor.py
```

关键依赖：
- Python 3.11
- torch 2.4.1 + torchaudio 2.4.1（CPU 版）
- ffmpeg 在 PATH 中
- HF_TOKEN 已配置（`.env` 文件）

> **注意**：`doctor` 可能提示 `pyannote.core <6` 和 `pyannote.metrics <4` 检查失败，这是脚本版本约束过严导致的，不影响实际运行。

### 3. 模型准备

模型自动缓存在 `models/` 目录：

| 模型 | 路径 | 说明 |
|------|------|------|
| pyannote 说话人分离 | `models/pyannote/` | 需 HF_TOKEN |
| UniASR | `models/uni_asr/` | 缅甸语专用 ASR（当前默认后端） |
| 翻译模型 | `models/translation/nllb-fixed/` | NLLB-200 本地修复版，缅甸语 → 简体中文 |

若模型缺失，运行：
```bash
python setup_models.py
```

### 4. HF_TOKEN 配置

若 `doctor.py` 提示 `HF_TOKEN` 缺失，在运行 diarization 前设置：

```powershell
$env:HF_TOKEN = "hf_..."
```

### 5. 运行测试

```bash
pytest tests/ -q
```

## 项目结构

```
voice-detection/
├── config.py                          # 全局参数与常量
├── main.py                            # CLI 入口（build / recognize 子命令）
├── pipeline.py                        # 高层流程编排（建库、识别流水线）
├── setup_models.py                    # 模型下载与初始化脚本
├── requirements.txt                   # Python 依赖
├── constraints-win-cpu.txt            # Windows CPU 环境约束
├── pytest.ini                         # 测试配置
│
├── scripts/                           # 工具脚本
│   ├── doctor.py                      # 环境依赖检查
│   ├── build_embedding_pool.py        # 批量：wav → diarization → embedding → 缓存
│   ├── build_speaker_repository.py    # 从 embedding 缓存构建声纹库
│   ├── eval_clustering_real.py        # 真实数据聚类效果评估
│   └── eval_phase6_asr.py             # ASR 准确率评估
│
├── src/                               # 源码
│   ├── core/                          # 核心数据结构与统一接口
│   │   ├── types.py                   # SpeakerSegment、SpeakerProfile、SpeakerData 等 dataclass
│   │   ├── pool.py                    # EmbeddingPool（封装 segments + matrix + meta_index）
│   │   ├── storage.py                 # Storage Protocol + Json/Npz/Pickle/Memory 实现
│   │   ├── repository.py              # SpeakerRepository（统一对外接口：speaker_db + vector_db + profile）
│   │   └── normalize.py               # Embedding L2 归一化工具
│   │
│   ├── audio/                         # 音频预处理
│   │   └── preprocess.py              # 格式转换、加载、切片、RMS 归一化
│   │
│   ├── diarization/                   # 说话人分割
│   │   ├── segment.py                 # pyannote 封装（run_diarization）
│   │   └── postprocess.py             # 片段合并、过滤、Annotation → SpeakerSegment
│   │
│   ├── embedding/                     # 声纹提取
│   │   ├── extractor.py               # SpeechBrain ECAPA-TDNN 封装
│   │   └── normalize.py               # 声纹向量归一化
│   │
│   ├── clustering/                    # 聚类与池构建
│   │   ├── cluster.py                 # 两层聚类（Agglomerative + centroid 合并）与去重
│   │   └── pool.py                    # 从 Npz/Json 缓存加载并构建 EmbeddingPool
│   │
│   ├── speaker_db/                    # 声纹库与向量检索
│   │   └── vector_index.py            # FaissVectorIndex 实现（build / search / add / rebuild）
│   │
│   ├── asr/                           # 语音识别
│   │   ├── whisper_asr.py             # Whisper 封装（遗留，模型已移除）
│   │   ├── uniasr_asr.py              # UniASR 封装（缅甸语专用，当前默认）
│   │   └── align.py                   # ASR 结果与 diarization 时间段对齐
│   │
│   ├── translation/                   # 翻译
│   │   └── translator.py              # NLLB-200 缅甸语 → 简体中文
│   │
│   └── recognition/                   # 识别辅助
│       └── verify.py                  # TopK 一致性判断（遗留，当前识别流程未使用）
│
├── tests/                             # 测试套件
│   ├── test_core_types.py             # core/types 不可变性测试
│   ├── test_core_pool.py              # EmbeddingPool 懒构建、apply_labels 测试
│   ├── test_core_storage.py           # Json/Npz/Memory Storage 测试
│   ├── test_audio.py                  # 音频预处理测试
│   ├── test_diarization.py            # 说话人分割与后处理测试
│   ├── test_embedding.py              # 声纹提取测试
│   ├── test_clustering.py             # 聚类逻辑测试
│   ├── test_speaker_db.py             # 向量索引测试
│   ├── test_asr.py                    # ASR 测试
│   ├── test_translation.py            # 翻译测试
│   ├── test_recognition.py            # 识别逻辑测试
│   ├── test_pipeline.py               # 端到端流水线测试
│   └── test_phase4_10_compat.py       # Phase 4-10 模块边界兼容性测试
│
├── models/                            # 预训练模型本地缓存（被 git 忽略）
│   ├── pyannote/                      # 说话人分割模型
│   ├── uni_asr/                       # UniASR 缅甸语 ASR 模型（当前默认）
│   └── translation/                   # NLLB-200 翻译模型
│
└── data/                              # 数据目录（被 git 忽略）
    ├── raw/                           # 原始音频
    ├── processed/                     # 中间处理结果（wav、embeddings、segments）
    └── output/                        # 最终识别结果 JSON
```

## 使用指南

### 数据目录结构

```
data/
├── raw/burmese_asr/audio/          # 原始音频（.wav）
├── processed/burmese_asr/train/    # 训练集 embeddings + segments
├── processed/burmese_asr/dev/      # 验证集 embeddings + segments
├── processed/burmese_asr/test/     # 测试集 embeddings + segments
├── processed/burmese_asr/wav/      # 预处理后的 16kHz mono wav
├── processed/speaker_db/           # 构建好的说话人库
└── output/                         # 识别结果输出
```

### 构建说话人库（Speaker DB）

使用训练集 embeddings 构建声纹识别库：

```bash
python main.py build \
  --embedding_dir data/processed/burmese_asr/train/embeddings \
  --output_dir data/processed/speaker_db \
  --label_strategy cluster
```

参数说明：
- `--label_strategy cluster`：基于聚类自动分配全局 speaker ID（默认，适用于未知数据集）
- `--label_strategy slr80_filename`：按 SLR80 文件名前缀分配（如 `BUR_0366`，适用于 SLR80 等有文件名标识的数据集）

构建完成后，`data/processed/speaker_db/` 会生成：
- `speaker_db/main.json` —— speaker 识别库（含中心向量、精选向量、profile 元数据）
- `vector_index/main.pkl` —— FAISS 向量索引缓存（可由 `speaker_db` 重建，仅作加速）

> **识别库自包含**：`speaker_db/main.json` 已包含在线识别所需的全部数据（center + 精选向量）。FAISS 缓存丢失后会自动从 JSON 重建。

### 单文件识别

#### 默认方式：UniASR（缅甸语推荐）

```bash
python main.py recognize \
  data/processed/burmese_asr/wav/bur_0366_5281755035.wav \
  --repo_dir data/processed/speaker_db \
  --no_translate \
  --output data/output/result.json
```

#### 编程调用

```python
from src.core.repository import SpeakerRepository
from src.core.storage import JsonStorage, PickleStorage
from src.speaker_db.vector_index import FaissVectorIndex
from src.embedding.extractor import EmbeddingExtractor
from src.translation.translator import Translator
from pipeline import recognize_pipeline

repo_dir = "data/processed/speaker_db"
vector_index = FaissVectorIndex()
repo = SpeakerRepository(vector_index=vector_index, storage=JsonStorage(repo_dir))
repo.load("speaker_db:main")

# 加载 FAISS 缓存
ps = PickleStorage(repo_dir)
if ps.exists("vector_index:main"):
    data = ps.load("vector_index:main")
    vector_index.build(data["vectors"], data["labels"])

extractor = EmbeddingExtractor(device="cpu")
translator = Translator(device="cpu")   # 传入 None 则跳过翻译

segments = recognize_pipeline(
    wav_path="data/processed/burmese_asr/wav/bur_0366_5281755035.wav",
    repo=repo,
    extractor=extractor,
    translator=translator,
    device="cpu",
)

for seg in segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.global_speaker}: {seg.text}")
    print(f"  翻译: {seg.translation}")
```

#### 输出格式（JSON）

```json
[
  {
    "segment_id": "bur_0366_5281755035_0000",
    "file": "bur_0366_5281755035.wav",
    "start": 0.706,
    "end": 4.587,
    "duration": 3.881,
    "sr": 16000,
    "local_speaker": "SPEAKER_00",
    "global_speaker": "BUR_0366",
    "display_name": null,
    "score": 0.823,
    "text": "ဒီကိစ္စ နဲ့လို့ခံတစ်ဦးတဲ့န်ပတ်သက်ဒေသဖြစ်ကိုဖြိုးဟက လိုပါတယ်လာ",
    "translation": null,
    "precision": null,
    "embedding": [0.0123, -0.0456, ...]
  }
]
```

**字段详解：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `segment_id` | string | 全局唯一标识，格式为 `{basename}_{seq:04d}` |
| `file` | string | 来源音频文件名 |
| `start` / `end` | float | 该语音片段在音频中的起止时间（秒） |
| `duration` | float | 片段时长，`end - start` |
| `sr` | int | 采样率，经 ffmpeg 标准化后固定为 16000 |
| `local_speaker` | string | pyannote 原始分割标签，如 `SPEAKER_00`；单文件内唯一，跨文件不对应同一说话人 |
| `global_speaker` | string | 全局说话人 ID，如 `SPK_0`、`BUR_0366` 或 `UNKNOWN`。由声纹聚类或 FAISS 检索分配，跨文件标识同一说话人 |
| `display_name` | string | 说话人姓名，需手动在 `speaker_db` 中标注；未标注时为 `null` |
| `score` | float | 说话人识别置信度（余弦相似度，范围约 `[-1, 1]`）。`UNKNOWN` 或识别失败时为 `null` |
| `text` | string | ASR 识别的缅甸语文本 |
| `translation` | string | 中文翻译结果；未启用翻译或失败时为 `null` |
| `precision` | float | 文本准确度（CER-based），需有参考文本时才计算，`precision = max(0, 1 - CER)`；无参考时为 `null` |
| `embedding` | list[float] | 当前语段的 192 维声纹向量（float32）。可用于离线分析或重新聚类 |

### 批量验证

端到端测试单条音频：

```bash
python main.py recognize \
  data/processed/burmese_asr/wav/bur_0366_5281755035.wav \
  --output data/output/result.json
```

也可批量处理整个目录：

```bash
python scripts/test_pipeline_on_dev.py \
  --wav_dir data/raw/burmese_asr/dev \
  --repo_dir data/processed/speaker_db_full \
  --output_dir data/processed/eval_results
```

### 预构建声纹库（开箱即用）

项目已包含基于 SLR80 训练集构建好的声纹库，位于 `data/processed/speaker_db_full/`：

- `speaker_db/main.json` —— 20 位说话人的识别库（center + 精选向量）
- `vector_index/main.pkl` —— FAISS 索引缓存

识别时直接加载，无需重新跑训练流程：

```bash
python main.py recognize \
  data/raw/burmese_asr/dev/bur_0366_5281755035.wav \
  --repo_dir data/processed/speaker_db_full \
  --output data/output/result.json
```

### 增量更新声纹库

识别过程中，**高置信度结果会自动增量更新**说话人向量（`repo.update_speaker()`，阈值见 `config.py` 中的 `T_HIGH`）。

如需**手动添加新说话人**或**批量注册语音**：

```python
from src.core.repository import SpeakerRepository
from src.core.storage import JsonStorage
from src.speaker_db.vector_index import FaissVectorIndex
import numpy as np

repo = SpeakerRepository(
    vector_index=FaissVectorIndex(),
    storage=JsonStorage("data/processed/speaker_db_full"),
)
repo.load("speaker_db:main")

# 添加全新说话人（传入多条语音的 embeddings）
repo.add_speaker(
    spk_id="SPK_NEW",
    embeddings=[emb1, emb2, emb3],   # list[np.ndarray]
    durations=[1.2, 3.5, 2.0],       # 对应时长，用于加权 center
)

# 给说话人命名
repo.assign_name("SPK_NEW", "张三")

# 保存更新后的库
repo.save("speaker_db:main")
```

> 注：`add_speaker` 会自动计算加权 center、去重、裁剪到 `MAX_EMB` 个精选向量，并重建 FAISS 索引。

## 系统架构与识别流程

### 识别流程（两层比对）

新音频进入系统后，说话人识别经历以下阶段：

1. **Stage 1（粗排）**：用 `speaker_db` 中所有 speaker 的 `center` 向量做余弦相似度排序，选出 Top-5 候选
2. **快速拒识**：若最像的候选 center 分数低于 `T_LOW = 0.50`，直接返回 `UNKNOWN`，不进入精排
3. **Stage 2（精排）**：对每个候选 speaker，取其保留的精选 embeddings 与 query 做内积，取 Top-5 相似度做平均
4. **最终判决**：取平均相似度最高的 speaker，若分数 ≥ `T_HIGH = 0.65` 则返回 `KNOWN`，否则返回 `UNKNOWN`

> 该流程经 leave-out 测试校准：陌生人拒识率 69-97%，库内误拒率 3-6%，陌生人误识率接近 0%。

### 数据流

```
输入音频
  → 音频标准化（ffmpeg → 16kHz mono wav）
  → 说话人分割（pyannote）
  → 声纹提取（ECAPA-TDNN，192 维）
  → 说话人识别（两层比对）
  → ASR（UniASR）
  → 翻译（NLLB-200，可选）
  → 结构化 JSON 输出
```

## ASR 后端

当前仅支持 **UniASR**（缅甸语专用模型），作为项目唯一后端。当前 CER 约 62.5%，仍有显著提升空间（见下方测试结果）。

> 历史版本曾支持 Whisper，但相关模型文件已移除，如需恢复需重新下载 `models/whisper/`。

### 全链路测试结果（SLR80 dev，100 条，CPU）

| 指标 | 结果 |
|------|------|
| 说话人识别准确率 | **94.12%**（102 个片段中 96 个正确） |
| ASR CER | **62.50%** |
| ASR Precision | **37.50%** |
| 单文件平均处理时间 | ~3.5s - 4.5s |
| 总评估时间 | ~12 分钟（100 文件，CPU） |

**结论**：说话人识别子系统已具备生产可用水平；ASR 是当前主要瓶颈，建议优先优化（如 fine-tune XLS-R CTC 或更换更大模型）。

## 性能参考（CPU）

以 5 秒音频为例：

| 阶段 | 耗时 |
|------|------|
| Diarization (pyannote) | ~25s（首次加载模型，同进程缓存后大幅缩短） |
| Embedding (SpeechBrain) | ~5s |
| ASR UniASR | ~3s - 5s |
| 说话人识别（向量比对） | < 10ms |

> 模型加载为惰性 + 缓存，同进程内第二次调用会快很多。当前 UniASR 模型每文件重新初始化，优化为单实例复用后可再节省 ~1-2s/文件。

## 关键配置参数

如需调整识别严格度或聚类行为，修改 `config.py` 中的以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `T_LOW` | 0.50 | 第一层 center 粗排拒识阈值；低于此值直接判定 `UNKNOWN` |
| `T_HIGH` | 0.65 | 第二层精排高置信确认阈值；FAISS Top-K 平均分 ≥ 此值才返回 `KNOWN` |
| `MAX_EMB` | 20 | 每个 speaker 在库中最大保留的精选向量数 |
| `DISTANCE_THRESHOLD` | 0.30 | 聚类第一层距离阈值（cosine distance），保守值保纯度 |
| `CENTROID_THRESHOLD` | 0.35 | 聚类第二层 centroid 合并阈值 |

## 常见问题

**Q: 终端显示缅甸语乱码？**
A: Windows CMD 默认 GBK 编码不支持缅甸字符。将结果输出到 JSON 文件后用 UTF-8 编辑器打开，或换用 Windows Terminal / PowerShell 7+。

**Q: pyannote 模型下载失败？**
A: 确保 `.env` 中 `HF_TOKEN` 有效，且已在 HuggingFace 上接受 `pyannote/speaker-diarization-3.1` 的使用许可。

**Q: 翻译模型第一次使用很慢？**
A: 本项目已预置本地修复版 `models/translation/nllb-fixed/`，无需联网下载。若该目录缺失，才会回退到 HuggingFace 在线下载（约 2.3GB）。

**Q: 如何跳过翻译？**
A: 命令行加 `--no_translate`，或 API 调用时 `translator=None`。

**Q: Speaker DB 需要重建吗？**
A: 仅在训练集 embeddings 有变更或聚类参数调整后才需要重建。识别阶段直接加载已有缓存即可。
