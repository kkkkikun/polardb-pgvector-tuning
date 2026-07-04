# PolarDB pgvector / PASE HNSW 内核级性能优化

> 全国大学生计算机系统能力大赛 · **PolarDB 数据库创新设计赛** 参赛作品
> 队伍 **1321481「静以修身」** · **全国优胜奖** · 决赛第 19 名（3534 支队伍）

在不引入任何新算法的前提下，围绕 HNSW 索引在高维固定维度（200 维）场景下的**内存带宽瓶颈与热路径冗余开销**，通过「压缩存储 + 访存优化 + 并发改进 + 微观精简」的综合策略，在严格保证 `Recall ≥ 0.85` 的前提下，将索引构建时间**降至原来的 45%（加速 2.21×）**，查询吞吐提升 **8.7%**，索引体积缩小 **34.1%**。

---

## 成绩速览

| 指标 | Baseline（原始 pgvector） | 优化后 | 变化 |
|------|---------------------------|--------|------|
| 索引构建时间 | 635,156 ms | **286,979 ms** | **−54.8%（2.21× 加速）** |
| 查询吞吐 QPS（Recall>0.85） | 2,709 | **2,945** | **+8.7%** |
| 索引体积 | 6,010 MB | **3,962 MB** | **−34.1%** |
| 召回率 Recall | — | **≥ 0.85** | 满足赛题约束 |

> 评测环境：官方评测机（支持 AVX-512，32GB 内存）；数据集 1000 万条 200 维向量；`M=12 / ef_construction=60 / ef_search≈81–87`。

---

## 基本信息

- **队伍 ID**：1321481
- **队伍名称**：静以修身
- **成员**：kikun、陈静茹 —— 鄂尔多斯应用技术学院
- **联系方式**：3581957503@qq.com · [blog.kkkkikun.dpdns.org](https://blog.kkkkikun.dpdns.org)
- **原始仓库（gitee）**：https://gitee.com/mikupromax/polardb_competition_2025.git
- **最佳成绩提交**：`17c0ee0c718aa88d767bb148294863b1b087e64e`

```bash
POLAR_REPO=https://gitee.com/mikupromax/polardb_competition_2025.git
POLAR_COMMIT=17c0ee0c718aa88d767bb148294863b1b087e64e
```

---

## 核心优化策略

六项优化逐级叠加，每一项都用 `perf` 量化验证：

| # | 优化项 | 思路 | 构建加速 | QPS 变化 |
|---|--------|------|----------|----------|
| 1 | **L2 距离 SIMD 重写** | 针对 200 维硬编码，AVX-512 16 路手工展开 + 尾部处理 | −6.63% | −2.57% |
| 2 | **激进编译/链接优化** | `-march=native -flto -O3 -funroll-loops -mavx512* -ffast-math -fno-semantic-interposition` | **−32.55%** | **+9.07%** |
| 3 | **针对性硬件预取** | 邻居遍历提前 2 步 `__builtin_prefetch`，掩盖 pointer-chasing 延迟 | −13.84% | −4.02% |
| 4 | **SQ8 标量量化** | FP16 200 维 800B → SQ8 208B（−74%），AVX-512 距离计算 | −12.59% | **+6.96%** |
| 5 | **版本号乐观锁** | `neighborVersion` 原子版本号 + 三阶段无锁读 + 降级加锁 | −1.91% | −1.68% |
| 6 | **热路径微优化** | Early Termination、哈希预分配、内联比较、TOAST 跳过 | −2.89% | +1.34% |

### 1. L2 距离计算的 SIMD 友好重写
索引构建阶段 L2 距离计算是高频操作。pgvector 原生实现虽支持自动向量化，但通用循环存在分支开销与对齐限制。我们针对**固定 200 维**做硬编码：手工 16 路循环展开 + 剩余项处理，使编译器稳定生成 AVX-512 指令（参考 FAISS / ScaNN 的 inner-product kernel 范式）。

### 2. 激进的编译与链接优化
在 `Makefile` 与 `build.sh` 中注入针对评测机的编译选项：**LTO** 跨模块内联与死代码消除、`-mavx512f/vl/bw/dq` 显式激活 AVX-512、`-funroll-loops -ftree-vectorize`、`-fassociative-math` 放宽浮点重排、`-fno-semantic-interposition` 减少 PLT 跳转。这一步带来最大单项收益（构建 −32.55%）。

### 3. 针对性的硬件预取
HNSW 邻居访问具有明显的 **pointer-chasing** 特性。在 `HnswSearchLayer` 邻居遍历中提前 2 步预取下一节点的结构体与向量数据（200 维 ≈ 12.5 个 Cache Line，仅预取前 4 个 CL 平衡带宽与收益），构建阶段因复用搜索逻辑受益明显。

### 4. SQ8 线性量化方案
在 pgvector 的 `halfvec` 路径上拦截向量写入，引入 **SQ8 标量量化**：每向量 `[scale:4B][bias:4B][codes:200×1B]`，体积从 800B 降至 208B（−74%）。
- **零分配两遍扫描**完成 min/max 统计与编码；
- 距离计算提供标量 fallback 与 **AVX-512 双路径**（32 路展开 + FMA + uint8→float 向量化转换）；
- 通过 `VARSIZE` + 魔数 `SQ8_TAG=0x5158` 运行时自动识别 FP32/FP16/SQ8，对上层透明；
- 常数向量（scale=0）启用 O(1) 快速路径。

### 5. 基于版本号的乐观锁机制
借鉴 MVCC 与 RCU，在 HNSW 节点新增 `pg_atomic_uint32 neighborVersion`：**先读版本 → 无锁复制邻接表 → 再读版本验证**，失败重试 3 次后降级加锁，`pg_memory_barrier()` 防重排。在读多写少场景具备高并发扩展潜力（评测低并发下收益有限，但正确性有保障）。

### 6. 工程细节微优化组合拳
- **Early Termination**：邻域搜索连续 32 次拒绝即提前终止；
- **哈希预分配**：构建阶段 `visited_hash` 容量 `2×ef×m → 3×ef×m`，减少 rehash；
- **指针复用**：热路径单次 `DatumGetPointer` 替代多次 `PG_DETOAST_DATUM_PACKED`；
- **内联比较**：`ItemPointerEquals` 改字段直比宏，消除 PLT 跳转（~20ns/次）。

---

## 关键技术细节

**数据表示层** — SQ8 量化格式 `[scale][bias][data:N×1B]`，200 维 800B→208B，魔数 `0x5158` 三格式自动识别：

```c
typedef struct HnswSQ8Payload {
    float   scale;
    float   bias;
    uint8_t data[FLEXIBLE_ARRAY_MEMBER];
} HnswSQ8Payload;
#define SQ8_TAG 0x5158  /* 'QX' 量化标记 */
```

**距离计算层** — 格式感知分发器，最常见路径（SQ8 vs SQ8）优先：

```c
static inline double HnswGetDistance(Datum a, Datum b, HnswSupport *support) {
    Vector *va = (Vector *) DatumGetPointer(a);
    Vector *vb = (Vector *) DatumGetPointer(b);
    bool a_sq8 = (va->unused == SQ8_TAG);
    bool b_sq8 = (vb->unused == SQ8_TAG);
    if (likely(a_sq8 && b_sq8))   /* 索引构建期最常见 */
        return (double) HnswSQ8Distance2_Vector(va, vb);
    /* ... 混合 / 原始类型 fallback ... */
}
```

**搜索层** — Early Termination 动态阈值：

```c
int consecutiveRejections = 0;
for (int i = 0; i < unvisitedLength; i++) {
    if (c->distance > f->distance) {
        if (unlikely(++consecutiveRejections >= 32)) break;  /* 无效遍历已足够多 */
    } else {
        consecutiveRejections = 0;  /* 找到改进，重置 */
    }
}
```

完整设计、perf 火焰图分析、边界情况与改进方向见 [`ROUND2/1321481_静以修身_设计文档.md`](ROUND2/1321481_静以修身_设计文档.md) 与 [`ROUND2/查询分析报告.md`](ROUND2/查询分析报告.md)。

---

## AI 辅助开发工作流

本项目深度使用 **Claude Code CLI** 辅助完成「理论分析 → 热点定位 → 方案落地 → 量化验证」的完整闭环：

- **perf 根因分析**：用 `perf record/report` + 火焰图定位到「Buffer 管理」是查询主瓶颈（占比 ~15-20%），而非向量计算本身——这直接决定了 SQ8 量化（压缩访存）优于继续堆算力的优化方向；
- **代码生成与迭代**：SQ8 量化流水线、AVX-512 距离核、乐观锁三阶段协议等均借助 Claude Code 快速原型化并持续调优；
- **量化验证**：每项优化都用自定义脚本验证 Recall 与性能，确保不破坏正确性。

> 全程深度使用 **Claude Code CLI** 驱动 perf / objdump / addr2line 等工具完成根因定位与代码迭代。

---

## 目录结构

```
.
├── polardb/                    # PolarDB 完整源码（已排除编译产物与数据集）
│   └── external/
│       ├── pgvector/src/       # ★ 距离计算 SIMD 重写、SQ8 量化集成
│       └── pase/hnsw/          # ★ HNSW 核心：构建/扫描/乐观锁/Early Termination
├── ROUND1/                     # 初赛配置（HNSW 参数调优）
├── ROUND2/                     # ★ 决赛：技术报告、perf 分析、设计文档
│   ├── 1321481_静以修身_设计文档.md
│   ├── 查询分析报告.md
│   ├── 索引构建分析报告.md
│   └── *.pdf                   # 参考文献（HNSW 调参 / SQ8 量化）
├── img/                        # 赛题官方图示
└── README.md
```

---

## 如何复现

> 本仓库不含向量数据集（`.hdf5`）与 perf 采样（`.data`），需另行下载。

1. **数据集**：从 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 获取（sift / glove 等），按赛题要求导入。
2. **构建 PolarDB + pgvector**：参考 [PolarDB 比赛 baseline](https://gitee.com/polardb-tianchi/polardb_competition_2025) 与本仓库 `ROUND2/config/config.conf`。
3. **启用优化**：编译时确保 Makefile 中的 `-march=native -flto -O3 -mavx512*` 生效（**注意：需 AVX-512 CPU，否则非法指令异常**，详见设计文档「问题 4」）。
4. **建索引 + 跑测**：
   ```sql
   CREATE INDEX ON vector_table USING hnsw (embedding vector_l2_ops)
     WITH (m = 20, ef_construction = 200);
   SET hnsw.ef_search = 100;
   ```

---

## 局限与改进方向

- **SQ8 侵入 halfvec 路径**：破坏类型解耦 → 改为独立 `sq8vec` 类型或索引参数控制；
- **激进编译破坏可移植性**：`-march=native` 仅限评测机 → Makefile 加 AVX-512 运行时检测与标量 fallback；
- **乐观锁低并发负优化**：版本检查有 overhead → 条件编译/运行期自适应；
- **Buffer 管理是查询真正瓶颈**（perf 实测）→ 未来借鉴 VectorChord 做独立 buffer 管理。

详见设计文档「存在的问题与改进方向」。

---

## 致谢

- 上游：[PolarDB](https://github.com/ApsaraDB/PolarDB) / [pgvector](https://github.com/pgvector/pgvector) / [PASE](https://github.com/alipay/PASE)
- 参考文献：Malkov《HNSW》、FAISS ScalarQuantizer、Johnson《SQ 量化小世界图算法》
- 感谢赛事方与指导教师。

## License

继承上游 PostgreSQL License。
