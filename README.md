# Movie Datasets Generator

从电影长片和长视频中生成高质量训练数据集的两阶段流水线工具，专为 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 风格的训练数据（LoRA、视频生成模型）设计，当前的 caption 流程已针对 Wan2.1 一类视频 LoRA 训练做过强化。

## 概述

从电影中提取短小、画面丰富的片段，并自动生成事实性描述文本（caption），可直接用于微调视频扩散模型。

**阶段一 — 预处理**（`preprocess.py`）：检测场景切换、规避对话片段、评估画面质量，导出短视频片段及元数据。

**阶段二 — 生成描述**（`caption_dataset.py`）：从每个片段中采样多帧画面并附带时间戳，发送至视觉大模型，对返回的描述进行严格校验，最终写入 `metadata.csv`。

### 功能特性

- 基于 ffmpeg 的场景检测，阈值可配置
- 利用内嵌字幕轨自动规避对话片段
- 画面质量评分（亮度、细节、运动量、肤色检测）
- 自动检测并裁剪黑边
- AI 驱动的描述生成，输出更适合视频 LoRA 的长 caption，强调动作、镜头运动、光线、色彩、构图与质感
- 每个采样帧会附带时间戳信息，帮助模型理解片段中的时序变化
- 兼容部分 OpenAI-compatible 后端把文本放在 `reasoning` / `reasoning_content` 字段中的返回格式
- 结构化校验（禁止风格词、角色名、情节推断；要求运动与镜头线索）
- 无效描述自动修复与重试机制
- 生成详细报告，包含置信度评分

## 环境要求

- **Python 3.7+**
- **ffmpeg / ffprobe** — 需在系统 `PATH` 中
- **OpenAI 兼容的视觉 API** — 默认使用[阿里云 DashScope](https://dashscope.aliyuncs.com)（Qwen3-VL）

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg
```

> [!NOTE]
> 无需安装任何第三方 Python 依赖，仅使用标准库。

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd moviedatasets_generater

# 2. 复制配置模板并填入 API Key
cp config.example.json config.json
# 编辑 config.json，将 api_key 设为你的 DashScope API Key

# 3. 提取片段
python preprocess.py --input-video /path/to/movie.mp4

# 4. 生成描述
python caption_dataset.py --metadata dataset_movie/metadata.csv

# 可选：增加采样帧数量以强化时序信息
python caption_dataset.py --metadata dataset_movie/metadata.csv --frame-count 7
```

## 使用方法

### 阶段一 — 提取片段

```bash
python preprocess.py --input-video /path/to/movie.mp4
```

运行后会在输入文件旁创建 `dataset_<movie>/` 目录，包含短视频 `.mp4` 片段和 `metadata.csv` 文件。

#### 预处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-dir` | `dataset_<movie>` | 片段和元数据的输出目录 |
| `--target-min` / `--target-max` | 20 / 35 | 目标生成片段数量范围 |
| `--clip-min` / `--clip-max` | 3.0 / 5.0 | 片段时长范围（秒） |
| `--skip-start` / `--skip-end` | 240 / 300 | 跳过视频开头/结尾的秒数 |
| `--scene-thresh` | 0.2 | 场景切换检测阈值 |
| `--crop-mode` | `auto` | 黑边处理：`auto`、`manual` 或 `none` |
| `--subtitle-stream` | `auto` | 字幕流：索引号、`auto` 或 `none` |
| `--crf` | 17 | 输出质量（值越低质量越高） |

### 阶段二 — 生成描述

```bash
python caption_dataset.py --metadata dataset_movie/metadata.csv
```

更新 `metadata.csv` 中的 `prompt` 列，并生成 `caption_report.csv` 详细报告。默认会均匀采样 5 帧，每帧连同其时间戳一起送入视觉模型。

#### 描述生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--metadata` | `dataset/metadata.csv` | 元数据文件路径 |
| `--overwrite` | `blank` | `blank`（仅填充空白）或 `all`（全部重写） |
| `--only` | — | 指定要处理的视频文件名，逗号分隔 |
| `--dry-run` | — | 预览描述结果，不写入元数据文件 |
| `--frame-count` | `5` | 每个片段采样的帧数，至少为 1 |
| `--api-base-url` | DashScope URL | 覆盖视觉 API 端点 |
| `--api-key` | — | 覆盖 API 密钥 |
| `--model` | `qwen3-vl-plus` | 覆盖视觉模型名称 |
| `--timeout` | `60.0` | 覆盖 API 超时秒数 |

### 配置文件

项目根目录下的 `config.json` 存储 API 配置（已包含默认模板）：

```json
{
    "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-...",
    "model": "qwen3-vl-plus",
    "timeout": 60.0
}
```

配置优先级：**命令行参数 > 环境变量 > config.json > 内置默认值**

环境变量仍然可用：`CAPTION_API_BASE_URL`、`CAPTION_API_KEY`、`CAPTION_MODEL`、`CAPTION_TIMEOUT`。

接口兼容性说明：

- 默认按 OpenAI `chat/completions` 兼容接口调用
- 优先读取返回中的 `message.content`
- 若某些兼容后端将文本放在 `message.reasoning` 或 `message.reasoning_content`，脚本也会自动回退解析

## 输出格式

### metadata.csv

```csv
video,prompt
clip_001.mp4,"CINESTYLE, A woman crosses a dim corridor while the camera follows with a restrained handheld drift. Low-key practical light leaves much of the hallway in shadow, with a narrow rim of warm light tracing her coat as she glances back. The palette stays muted and cool overall, broken by isolated amber highlights from wall fixtures. Shallow depth of field softens the distant background and keeps attention on her movement through the frame."
clip_002.mp4,"CINESTYLE, Hands move across a laptop keyboard as a locked-off close shot holds steady on the desk surface. Cool daylight from tall windows mixes with soft interior fill, creating a clean corporate contrast across the keys and fingers. The image uses a restrained neutral grade with gentle reflections on metal and glass. Fine background blur and subtle screen glow give the workspace a polished contemporary texture."
```

### 描述校验规则

生成的描述会经过严格校验以保证训练数据质量：

- 英文输出，自动加上触发词 `CINESTYLE`
- 最多 6 个句子，鼓励使用 4 到 6 个句子完整描述片段
- 必须包含可见动作或时间推进线索
- 必须包含镜头行为线索（如 `handheld`、`tracking shot`、`push-in`、`pan`、`locked-off`）
- 必须包含电影风格词，并在结构化返回中填充 `style_terms_used`
- 禁止风格词（`cinematic`、`masterpiece`、`4k`、`photorealistic` 等）
- 禁止不确定用语（`maybe`、`possibly`、`appears to be`）
- 禁止出现人名、角色名或电影名称
- 禁止推断人物关系、职业身份或情节动机
- 禁止引用对白、字幕或屏幕文字
- 仅描述画面中直接可见的内容，不使用 “in the first frame” 一类元叙述

未通过校验的描述会进入修复循环（最多 2 次），修复失败则留空以便后续重试。

## 流水线示意图

```
movie.mp4
  │
  ├─ preprocess.py
  │    ├─ 场景检测 (ffmpeg)
  │    ├─ 字幕提取 & 对话规避
  │    ├─ 质量评分 (亮度、细节、运动量)
  │    ├─ 黑边裁剪
  │    └─ 片段导出 → dataset/clip_*.mp4
  │                 → dataset/metadata.csv
  │
  └─ caption_dataset.py
       ├─ 帧采样 (ffmpeg)
       ├─ 视觉 API 调用 (Qwen3-VL)
       ├─ 描述校验 & 修复
       └─ 输出 → dataset/metadata.csv (已更新)
                → dataset/caption_report.csv
```
