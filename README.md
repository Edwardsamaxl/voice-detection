# Voice Detection

说话人分割与声纹识别流水线，支持缅甸语语音处理。

- **说话人分离**（Diarization）—— pyannote.audio
- **声纹 Embedding 提取** —— SpeechBrain ECAPA-TDNN
- **说话人聚类与识别** —— FAISS + 自定义阈值策略
- **语音识别**（ASR）—— Whisper（通用）或 UniASR（缅甸语专用）
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
| Whisper | `models/whisper/base.pt` | 通用多语言 ASR |
| UniASR | `models/uni_asr/` | 缅甸语专用 ASR |
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
│   │   ├── whisper_asr.py             # Whisper 封装（多语言通用）
│   │   ├── uniasr_asr.py              # UniASR 封装（缅甸语专用）
│   │   └── align.py                   # ASR 结果与 diarization 时间段对齐
│   │
│   ├── translation/                   # 翻译
│   │   └── translator.py              # NLLB-200 缅甸语 → 简体中文
│   │
│   └── recognition/                   # 识别辅助
│       └── verify.py                  # TopK 一致性判断
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
│   ├── whisper/                       # Whisper ASR 模型
│   ├── uni_asr/                       # UniASR 缅甸语 ASR 模型
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
- `--label_strategy cluster`：基于聚类自动分配全局 speaker ID（默认）
- `--label_strategy slr80_filename`：按 SLR80 文件名前缀分配（如 `BUR_0366`）

构建完成后，`data/processed/speaker_db/` 会生成：
- `speaker_db/main.json` —— speaker 元数据
- `vector_index/main.pkl` —— FAISS 向量索引缓存

### 单文件识别

#### 默认方式：UniASR（缅甸语推荐）

```bash
python main.py recognize \
  data/processed/burmese_asr/wav/bur_0366_5281755035.wav \
  --repo_dir data/processed/speaker_db \
  --no_translate \
  --output data/output/result.json
```

#### 切换到 Whisper

```bash
python main.py recognize \
  data/processed/burmese_asr/wav/bur_0366_5281755035.wav \
  --repo_dir data/processed/speaker_db \
  --asr_backend whisper \
  --asr_model base \
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
    asr_backend="uniasr",      # "whisper" 或 "uniasr"
    device="cpu",
)

for seg in segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.global_speaker}: {seg.text}")
    print(f"  翻译: {seg.translation}")
```

#### 输出格式

```json
[
  {
    "start": 0.706,
    "end": 4.587,
    "speaker": "SPK_8",
    "display_name": null,
    "text": "ဒီကိစ္စ နဲ့လို့ခံတစ်ဦးတဲ့န်ပတ်သက်ဒေသဖြစ်ကိုဖြိုးဟက လိုပါတယ်လာ",
    "translation": "需要一个特定的领域来解决这个问题."
  }
]
```

### 批量验证

端到端测试单条音频：

```bash
python main.py recognize \
  data/processed/burmese_asr/wav/bur_0366_5281755035.wav \
  --output data/output/result.json
```

## ASR 后端选择建议

| 场景 | 推荐后端 | 说明 |
|------|----------|------|
| 缅甸语 | **UniASR** | 专用于缅甸语，CER/WER 显著优于 Whisper |
| 多语言/英语 | Whisper | base/large-v3 等模型选择灵活 |
| 快速原型 | Whisper tiny | 速度快，精度低，适合调试 pipeline |

## 性能参考（CPU）

以 5 秒音频为例：

| 阶段 | 耗时 |
|------|------|
| Diarization (pyannote) | ~25s（首次加载模型） |
| Embedding (SpeechBrain) | ~5s |
| ASR Whisper base | ~10s |
| ASR UniASR | ~5s |

> 模型加载为惰性 + 缓存，同进程内第二次调用会快很多。

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
