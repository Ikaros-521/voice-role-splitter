# voice-role-splitter

把一堆角色语音按**角色**分好：声纹嵌入聚类 → 人在浏览器里确认命名 → 对剩下没认领的做**经过标定的**二次归属 → 导出成 `roles/<角色>/<语气>/<文件>.wav`。

[中文](README.md) | [English](README.en.md)

![tests](https://github.com/Ikaros-521/voice-role-splitter/actions/workflows/tests.yml/badge.svg)

## 为什么不是"跑个聚类就完事"

这类工具常见的做法是：声纹嵌入 → 聚类 → 把每个簇当成一个角色。我在 7,055 条 / 6.7 小时 / 6 个角色的日语动画游戏台词上实测过，这条路是错的：

- **一个角色一个质心 = 瞎猜。** 用人工确认过的簇做留一法验证，角色判对率只有 **15.3%**，而 6 个角色随机猜的基线是 16.7%。
- **簇不等于角色。** 同一个角色被拆成 7 簇、另一个被拆成 5 簇——嵌入空间里占主导的是**说话方式/情绪**（喊叫、低语、哭腔、半秒语气词），不是身份。
- **文件序号不是先验。** 资源包里相邻条目看着像同一人（MFCC 相似度 0.552 vs 随机 0.002），但音高一测就推翻：相邻两条中位 F0 差 **78 Hz**，随机对 **88 Hz**，基本没差别。那是同场景、同录音链造出来的假信号。

所以这个工具的定位是：**模型负责"像不像"，人负责"是谁"**；之后模型只在有实测准确率支撑的边际区间内，去补那些没人认领的文件。每一步的质量数字都来自你自己的数据，不是论文里的平均值。

```
 vrs embed ──► vrs cluster ──► vrs page clusters ──► 人工给 17 个簇命名（约 15 分钟）
                                                              │
 vrs export ◄── vrs refine ◄── vrs apply ◄──── labels_manual.csv
     │            │  用"人亲耳听过的文件"做留一法标定，只补没有角色的
     ▼            ▼
 roles/<角色>/c04_激动喊叫/*.wav      roles_final.csv + review_list.csv
```

## 目录

- [安装](#安装) · [快速开始](#快速开始) · [工作流](#工作流)
- [产出物](#产出物) · [能力边界（实测）](#能力边界实测) · [设计红线](#设计红线)
- [文档](#文档) · [数据与版权](#数据与版权)

## 安装

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[campplus]"                            # 核心依赖 + 一个声纹后端
pip install -e ".[dev]"                                 # 跑测试
```

后端彼此独立、全部可选，装你环境里装得上的那个：

| 后端 | 训练数据 | 安装 | 说明 |
|---|---|---|---|
| `campplus` | 中+英（3D-Speaker） | `[campplus]`（funasr） | 还需 `kaldiio torch_complex omegaconf jaconv jieba` |
| `ecapa` | 英文（VoxCeleb） | `[ecapa]`（`speechbrain<1.0`） | 1.x 需要 torch≥2.4（`torch.amp`） |
| `eres2net` | 中文（3D-Speaker） | `[eres2net]`（modelscope） | 最重，单模型最强 |

**至少装两个后端**才有意义：跨模型一致性是这条链路里少数不依赖任何假设的质量信号。

```bash
vrs backends --run-dir runs/show1                # 哪个能用、为什么不能用、缺哪行 pip
vrs backends --run-dir runs/show1 --fetch ecapa  # 拉权重（镜像常 429 限流，自带重试）
```

GPU 可选（CPU 能跑，慢约 10 倍）。全程本地，不上传、不需要 API key。

**权重存放**：先在机器上已有的缓存里找（`run.yaml` 的 `model_dir` > `$VRS_MODEL_DIR` >
`$MODELSCOPE_CACHE` > `~/.cache/modelscope` > `$HF_HOME`/`$HF_HUB_CACHE` >
`~/.cache/huggingface` > 项目内 `./models`），**找到就直接复用，绝不重复下载**；确实没有时，
默认下载到**项目目录下的 `models/`**（跟着仓库走、看得见、`gitignore` 已排除），而不是丢进 C 盘。
想换地方就用 `--set model_dir=D:/weights` 或设 `VRS_MODEL_DIR`。

## 快速开始

```bash
# 1. 一个数据集一个 run 目录；run.yaml 是唯一需要手改的文件
vrs init --audio-dir ./wav/voice --run-dir runs/show1 \
         --roles-file examples/cast_sora_no_otoshimono.yaml

vrs status  --run-dir runs/show1     # 做到哪一步、还缺什么
vrs probe   --run-dir runs/show1     # 条数、时长、采样率、<1s 有多少
vrs embed   --run-dir runs/show1     # 各后端嵌入 + 一次健康检查
vrs pitch   --run-dir runs/show1     # F0 统计（与模型无关的质检信号）
vrs cluster --run-dir runs/show1 --ks 8,12,16,20,24
vrs page clusters --run-dir runs/show1   # 标注.html：逐个簇听声命名

# …打开页面、听、命名、导出 labels_manual.csv 放进 run 目录

vrs apply  --run-dir runs/show1      # 命名落到每一行 -> roles.csv
vrs refine --run-dir runs/show1      # 只给"还没有角色"的文件定归属，阈值经标定
vrs page review --run-dir runs/show1 # 待人工.html：剩下的按建议分组，可整组接受
vrs export --run-dir runs/show1      # roles/<角色>/c<簇号>_<语气>/*.wav（默认硬链接）
```

### 角色表（`--roles-file`）怎么构建

`examples/cast_sora_no_otoshimono.yaml` 就是这份角色表：`vrs init` 读它、写进 `run.yaml` 的 `roles:`，
它就是标注页面下拉菜单里的全部选项。

```yaml
roles:
  伊卡洛斯: "イカロス（早見沙織）"   # 左＝落盘的角色 id；右＝只给人看的标签
```

- **左边 `value` 是身份标识**：原样进 `roles.csv` 的 `role` 列，并原样当作导出目录名
  `roles/<value>/`。**两个簇起同一个名字 = 同一个角色**（这才是常态：一个角色会被情绪拆成好几簇）。
  所以 value 一次定死：一种语言、不带装饰符，你想在磁盘上看到什么就写什么。
- **右边 `label` 只影响听感效率**：建议「原文名（声优）」，2 秒一条地听时，声优名是最快能把两个
  相似音色区分开的信息。
- **下拉是闭集，页面没有自由输入框**——故意的：手打一个名字就是凭空新增一个角色，
  `伊卡洛斯` / `イカロス` / `伊卡洛斯（天使）` 会静默变成三个文件夹。听到表里没有的角色，
  就加一行 → 重跑 `vrs page clusters` → 重开页面。已命名的结果存在浏览器 `localStorage`
  （键是 run 目录 + 模式，与角色表无关），重新生成页面不会丢；只有重跑 `vrs cluster` 会重排簇号，
  那种情况先 `vrs apply` 把命名落盘。
- **构建顺序**：① 从官方 staff 表 / wiki（Bangumi、VNDB、MyAnimeList）/ 游戏包装把角色名和**声优名**抄全，
  ② 定 value 用哪种语言，③ 一个"声音"一行而不是一个"角色"一行——旁白/系统语音、纯音效各留一行兜底，
  ④ 次要角色也先写上，宁可用不到也别在命名的中途卡住，⑤ `vrs init --roles-file` 会校验并打印
  `roles : N in the page dropdown`，写错当场报第几条、为什么错。
- **校验规则**（因为 value 要当目录名）：不含 `\ / : * ? " < > |`、不以 `.` 开头或结尾、
  去空格后非空且 ≤60 字符、忽略大小写不重复、且不能占用 `未定` / `需复核` / `unresolved`
  （这三个是流水线表示"还没定"的记号，拿去当角色名会让文件全跑进 `_未确定`）。
- 同一份表可以写成 `[[value, label], …]` pairs、纯字符串列表，或者 `.json`；页面里 label 为空时
  自动等于 value。跑完一轮后，`roles.csv` 的 `role` 列去个重就是下一部作品的角色表起点
  （命令见 [docs/USAGE.md](docs/USAGE.md#cast-file-role-vocabulary-of-the-annotation-page)）。

## 工作流

| 步骤 | 产物 | 谁说了算 |
|---|---|---|
| `embed` | `embeddings/*.npy`、`meta.npz`、健康检查结论 | 模型 |
| `pitch` | `pitch.npz`（中位 F0、音高跨度、浊音占比） | 测量 |
| `cluster` | `cluster_sweep.csv`、`labels.csv` | 模型，按 `agree_kappa` / `verify_pass` / `f0_disp` / silhouette |
| `page clusters` | `标注.html` | **人**：一个簇一个角色名 |
| `apply` | `roles.csv`、`review_list.csv` | 人的命名原样传播，不改判 |
| `refine` | `roles_final.csv`、`queue.csv`、`gold_calibration.csv` | 模型，且只在标定过的边际区间内 |
| `page review` | `待人工.html` | **人**：整组接受 + 抽检 25% |
| `export` | `roles/…`、`clusters.csv` | 机械操作 |

每步幂等，只读写 run 目录：任何一步之后都能停下、换参数重来、或直接手改 `run.yaml`。

## 产出物

- **`roles.csv` / `roles_final.csv`**：每行一条语音，含 `role`、`cluster`、`own_pct`（在本簇里的典型程度）、`centroid_score`、`cohesive_margin`、`cross_delta`、`confidence`、`speech_sec`、`f0`；有转写时附 `text`。
- **`review_list.csv`**：该用耳朵过一遍的文件，按风险排序。
- **`clusters.csv`**：每个簇的条数、中位时长、响度分位、F0 中位/跨度，以及生成的语气名（依据实测声学量，不是猜的）。
- **`roles/<角色>/c<簇号>_<语气>/*.wav`**：默认硬链接（不额外占磁盘）；`--copy` 得到可独立编辑的副本，`--flat` 取消语气子层。

## 能力边界（实测）

上面那个语料（7,055 条 / 6 个角色）的数字，方法见 [docs/METHOD.md](docs/METHOD.md)（暂为英文）：

| 结论 | 实测 |
|---|---|
| 单角色单质心 ≈ 随机 | 留一法 top-1 **0.153**（随机基线 0.167） |
| 簇≠角色，情绪主导分簇 | 1 个角色被拆成 7 簇，另一个 5 簇 |
| top-k 判据·跨簇（需该角色≥2 簇） | 整体 **0.571**；边际 ≥0.03 时 **0.84** |
| top-k 判据·同簇可参照（用于无角色文件） | 边际 0.03–0.06 → **0.87**；≥0.06 → **0.93** |
| 无监督切分的质量 | 簇内 ASV 通过率 **0.71**，随机对 0.12 |

由此定下的行为：`refine` 默认 `only_unassigned`（不碰人已确认的），`propose` 永不自动落库，阈值标定直接打印在你亲手听过的文件上。**单簇角色无法跨簇验证**，工具会明说这一点，而不是硬编一个置信度。

短句是真正的硬骨头：那批剩余队列里 466 条的中位长度只有 **0.69 秒**，其中 62 条连 F0 都测不出、23 条完全没有声纹向量。这种别硬分，留在 `_未确定/`。

## 设计红线

1. **喂给后端的音频必须是 [-1, 1] 的 float。** 两个前端内部都会自己缩放；送 int16 量级会让所有余弦虚高（实测均值 0.62 vs 正常 0.40），聚类塌成一个吃掉 94% 数据的大簇——而且各项指标还"看着挺好"。`audio.read_wav` 现在直接抛 `ScalingError`。
2. **不用任何位置先验。** 文件序号反映的是场景/录音链相似，不是同角色；全流程不看序号。
3. **平均连接层次簇不是质心球形区域**，所以"某后端最近质心≠所在簇"是方法学必然，只能当参考列，不能当风险分。
4. **人的判断是终态。** `apply` 原样传播命名；`refine` 默认不碰已命名文件；`propose` 永不自动改判。
5. **质量指标要么无前提，要么标清楚**：跨模型 κ、簇内 ASV 通过率、簇内音高离散度是无前提的；silhouette 只做参考并降权。
6. **给人打开的东西一律 UTF-8 BOM**；带中文的 numpy 数组必须 `dtype=object`（定宽 unicode 会静默截断，我踩过）。
7. **run 目录必须可整目录复制**，所以权重缓存放外面（默认 `~/.cache/vrs/models`）。

## 文档

- [docs/USAGE.md](docs/USAGE.md) — 每条命令、`run.yaml` 每个字段
- [docs/METHOD.md](docs/METHOD.md) — 流程、实测数字、为什么这样选
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — 环境版本矩阵、下载限流、编码、卡死排查
- [CONTRIBUTING.md](CONTRIBUTING.md) — 开发环境、上面这些红线、如何加一个新后端
- [examples/](examples) — 一份角色表、一份 `apply` 期望的标注导出格式样例

```bash
pytest -q     # 85 项，离线 + 纯 CPU，约 1 分钟；上面每个坑都有对应回归用例
```

`docs/` 目前只有英文，欢迎翻译 PR。

## 数据与版权

仓库**不含任何音频**。数据相关的一切（`.wav`、嵌入、run 产物、导出的角色文件夹、生成的页面）都在 `.gitignore` 里。工具和示例配置是 MIT；但动画/游戏台词本身是别人录制的版权内容——把标好角色的音频包发出去属于再分发他人作品，跟工具开源是两回事。要分享配置就提交 `run.yaml` 和角色表，音频留在本地。

许可证见 [LICENSE](LICENSE)。CI 在 Linux/Windows × Python 3.10–3.12 上跑测试套件（[.github/workflows/tests.yml](.github/workflows/tests.yml)）。
