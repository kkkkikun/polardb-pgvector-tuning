#ifndef HNSW_H
#define HNSW_H

#include "postgres.h"

#include "access/genam.h"
#include "access/parallel.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "port.h"				/* for random() */
#include "utils/relptr.h"
#include "utils/sampling.h"
#include "vector.h"
#include "halfvec.h"

#define HNSW_MAX_DIM 2000
#define HNSW_MAX_NNZ 1000

/* Support functions */
#define HNSW_DISTANCE_PROC 1
#define HNSW_NORM_PROC 2
#define HNSW_TYPE_INFO_PROC 3

#define HNSW_VERSION	1
#define HNSW_MAGIC_NUMBER 0xA953A953
#define HNSW_PAGE_ID	0xFF90

/* Preserved page numbers */
#define HNSW_METAPAGE_BLKNO	0
#define HNSW_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define HNSW_UPDATE_LOCK 	0
#define HNSW_SCAN_LOCK		1

/* HNSW parameters */
#define HNSW_DEFAULT_M	16
#define HNSW_MIN_M	2
#define HNSW_MAX_M		100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION		1000
#define HNSW_DEFAULT_EF_SEARCH	40
#define HNSW_MIN_EF_SEARCH		1
#define HNSW_MAX_EF_SEARCH		1000

/* Tuple types */
#define HNSW_ELEMENT_TUPLE_TYPE  1
#define HNSW_NEIGHBOR_TUPLE_TYPE 2

/* Make graph robust against non-HOT updates */
#define HNSW_HEAPTIDS 1

#define HNSW_UPDATE_ENTRY_GREATER 1
#define HNSW_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PROGRESS_HNSW_PHASE_LOAD		2

#define HNSW_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))
#define HNSW_TUPLE_ALLOC_SIZE BLCKSZ

#define HNSW_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(offsetof(HnswElementTupleData, data) + (size))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(offsetof(HnswNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

#define HNSW_NEIGHBOR_ARRAY_SIZE(lm)	(offsetof(HnswNeighborArray, items) + sizeof(HnswCandidate) * (lm))

#define HnswPageGetOpaque(page)	((HnswPageOpaque) PageGetSpecialPointer(page))
#define HnswPageGetMeta(page)	((HnswMetaPageData *) PageGetContents(page))

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#define HnswIsElementTuple(tup) ((tup)->type == HNSW_ELEMENT_TUPLE_TYPE)
#define HnswIsNeighborTuple(tup) ((tup)->type == HNSW_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define HnswGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define HnswGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define HnswGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - offsetof(HnswNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 255)

#define HnswGetSearchCandidate(membername, ptr) pairingheap_container(HnswSearchCandidate, membername, ptr)
#define HnswGetSearchCandidateConst(membername, ptr) pairingheap_const_container(HnswSearchCandidate, membername, ptr)

#define HnswGetValue(base, element) PointerGetDatum(HnswPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define HnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define HnswPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, value))
#define HnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define HnswPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define HnswPtrPointer(hp) (hp).ptr
#define HnswPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	hnsw_ef_search;
extern int	hnsw_iterative_scan;
extern int	hnsw_max_scan_tuples;
extern double hnsw_scan_mem_multiplier;
extern int	hnsw_lock_tranche_id;

typedef enum HnswIterativeScanMode
{
	HNSW_ITERATIVE_SCAN_OFF,
	HNSW_ITERATIVE_SCAN_RELAXED,
	HNSW_ITERATIVE_SCAN_STRICT
}			HnswIterativeScanMode;

typedef struct HnswElementData HnswElementData;
typedef struct HnswNeighborArray HnswNeighborArray;

#define HnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
HnswPtrDeclare(HnswElementData, HnswElementRelptr, HnswElementPtr);
HnswPtrDeclare(HnswNeighborArray, HnswNeighborArrayRelptr, HnswNeighborArrayPtr);
HnswPtrDeclare(HnswNeighborArrayPtr, HnswNeighborsRelptr, HnswNeighborsPtr);
HnswPtrDeclare(char, DatumRelptr, DatumPtr);

struct HnswElementData
{
	HnswElementPtr next;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	HnswNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;
} __attribute__((aligned(64))); //强制每个节点在内存中按照 64 字节（CPU Cache Line 大小）对齐。这样当 CPU 读取一个节点时，能一次性把该节点的关键信息（neighbors 指针、value 指针等）全部读入一级缓存，减少读取次数。

typedef HnswElementData * HnswElement;

typedef struct HnswCandidate
{
	HnswElementPtr element;
	float		distance;
	bool		closer;
}			HnswCandidate;

struct HnswNeighborArray
{
	int			length;
	bool		closerSet;
	HnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct HnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	HnswElementPtr element;
	double		distance;
}			HnswSearchCandidate;

/* HNSW index options */
typedef struct HnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			HnswOptions;

typedef struct HnswGraph
{
	/* Graph state */
	slock_t		lock;
	HnswElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	HnswElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			HnswGraph;

typedef struct HnswShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	HnswGraph	graphData;
}			HnswShared;

#define ParallelTableScanFromHnswShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(HnswShared)))

typedef struct HnswLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	HnswShared *hnswshared;
	Snapshot	snapshot;
	char	   *hnswarea;
}			HnswLeader;

typedef struct HnswAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			HnswAllocator;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			HnswTypeInfo;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
}			HnswSupport;

typedef struct HnswQuery
{
	Datum		value;
}			HnswQuery;

typedef struct HnswBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		HnswTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	HnswGraph	graphData;
	HnswGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	HnswAllocator allocator;

	/* Parallel builds */
	HnswLeader *hnswleader;
	HnswShared *hnswshared;
	char	   *hnswarea;
}			HnswBuildState;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* for identification of HNSW indexes */
}			HnswPageOpaqueData;

typedef HnswPageOpaqueData * HnswPageOpaque;

typedef struct HnswElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			HnswElementTupleData;

typedef HnswElementTupleData * HnswElementTuple;

typedef struct HnswNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborTupleData;

typedef HnswNeighborTupleData * HnswNeighborTuple;

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	struct tidhash_hash *tids;
}			visited_hash;

typedef union
{
	HnswElement element;
	ItemPointerData indextid;
}			HnswUnvisited;

typedef struct HnswScanOpaqueData
{
	const		HnswTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	pairingheap *discarded;
	HnswQuery	q;
	int			m;
	int64		tuples;
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;

	/* Support functions */
	HnswSupport support;
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

typedef struct HnswVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	struct tidhash_hash *deleted;
	BufferAccessStrategy bas;
	HnswNeighborTuple ntup;
	HnswElementData highestPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			HnswVacuumState;

/* Methods */
int			HnswGetM(Relation index);
int			HnswGetEfConstruction(Relation index);
FmgrInfo   *HnswOptionalProcInfo(Relation index, uint16 procnum);
void		HnswInitSupport(HnswSupport * support, Relation index);
Datum		HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value);
bool		HnswCheckNorm(HnswSupport * support, Datum value);
Buffer		HnswNewBuffer(Relation index, ForkNumber forkNum);
void		HnswInitPage(Buffer buf, Page page);
void		HnswInit(void);
List	   *HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples);
HnswElement HnswGetEntryPoint(Relation index);
void		HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint);
void	   *HnswAlloc(HnswAllocator * allocator, Size size);
HnswElement HnswInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, HnswAllocator * alloc);
HnswElement HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing);
HnswSearchCandidate *HnswEntryCandidate(char *base, HnswElement em, HnswQuery * q, Relation rel, HnswSupport * support, bool loadVec);
void		HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building);
void		HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m);
void		HnswAddHeapTid(HnswElement element, ItemPointer heaptid);
HnswNeighborArray *HnswInitNeighborArray(int lm, HnswAllocator * allocator);
void		HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * alloc);
bool		HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building);
void		HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building);
void		HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec);
void		HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance);
bool		HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support);
void		HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element);
void		HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support);
bool		HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc);
void		HnswInitLockTranche(void);
const		HnswTypeInfo *HnswGetTypeInfo(Relation index);
PGDLLEXPORT void HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		hnswbuildempty(Relation index);
bool		hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
void		hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		hnswgettuple(IndexScanDesc scan, ScanDirection dir);
void		hnswendscan(IndexScanDesc scan);

static inline HnswNeighborArray *
HnswGetNeighbors(char *base, HnswElement element, int lc)
{
	HnswNeighborArrayPtr *neighborList = HnswPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return HnswPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

/* * SQ8 数据包结构
 * 我们将其存储在标准 Vector 的 x[] 数组位置
 */
typedef struct HnswSQ8Payload {
    float   scale;
    float   bias;
    uint8_t data[FLEXIBLE_ARRAY_MEMBER];
} HnswSQ8Payload;

#include "halfutils.h"
#include <immintrin.h>
#include <string.h>

/* SQ8 量化数据的魔数标记，用于可靠识别 */
#define SQ8_TAG 0x5158  /* 'QX' - 量化标记 */

/* HNSW值格式枚举 */
typedef enum
{
	HNSW_FMT_RAW_FLOAT_VEC,    /* 原始vector：4字节float/元素 */
	HNSW_FMT_RAW_HALFVEC,      /* 原始halfvec：2字节half/元素 */
	HNSW_FMT_SQ8               /* SQ8量化：scale(4) + bias(4) + codes(dim) */
} HnswValueFormat;

/*
 * 统一的格式检测函数
 * 在解包之前判断数据类型，避免把halfvec误认为vector
 * 关键：必须在调用DatumGetVector/DatumGetHalfVector之前使用
 */
static inline HnswValueFormat
HnswDetectFormat(Datum d)
{
	Pointer p = (Pointer) PG_DETOAST_DATUM_PACKED(d);

	/* 头部布局：vl_len + int16 dim + uint16 unused */
	int16 dim = ((Vector *)p)->dim;
	int sz = VARSIZE_ANY(p);

	int hdr = VARHDRSZ + sizeof(int16) + sizeof(uint16);
	int payload = sz - hdr;

	if (dim <= 0 || payload < 0)
		elog(ERROR, "bad vector header: dim=%d size=%d payload=%d", dim, sz, payload);

	/* 原始float vector：payload == 4 * dim */
	if (payload == (int)(sizeof(float) * dim)) {
		/* 进一步检查：确保不是误判的SQ8（SQ8也有payload可能接近4*dim） */
		Vector *v = (Vector *)p;
		if (v->unused == SQ8_TAG) {
			/* 这是SQ8数据，但payload意外匹配 */
			elog(WARNING, "SQ8 data detected with vector-like payload, treating as SQ8");
			return HNSW_FMT_SQ8;
		}
		return HNSW_FMT_RAW_FLOAT_VEC;
	}

	/* 原始halfvec：payload == 2 * dim */
	if (payload == (int)(sizeof(uint16_t) * dim)) {
		return HNSW_FMT_RAW_HALFVEC;
	}

	/* SQ8量化：payload == 8 + dim (scale+bias+codes) */
	if (payload == (int)(sizeof(float) * 2 + dim)) {
		/* 进一步检查SQ8魔数标记 */
		Vector *v = (Vector *)p;
		if (v->unused == SQ8_TAG) {
			return HNSW_FMT_SQ8;
		} else {
			elog(WARNING, "SQ8-like payload without proper tag, treating as SQ8 anyway");
			return HNSW_FMT_SQ8;
		}
	}

	/* 无法识别的格式 */
	elog(ERROR, "unknown vector format: dim=%d size=%d payload=%d", dim, sz, payload);
	return HNSW_FMT_RAW_FLOAT_VEC; /* unreachable */
}

/*
 * 基于VARSIZE安全检测datum类型是vector还是halfvec
 * 不依赖维度阈值，而是通过实际的字节长度判断元素宽度
 */
static inline bool
IsHalfvecDatum(Datum d)
{
    void *p = PG_DETOAST_DATUM_PACKED(d);

    /* Vector和HalfVector的dim偏移相同（int16 dim），但x[]元素宽度不同 */
    int16 dim = ((Vector *)p)->dim;  /* 只用来取dim，不去读x[] */
    int sz = VARSIZE_ANY(p);

    /* 计算期望的字节大小 */
    int sz_vec  = VARHDRSZ + sizeof(int16) + sizeof(int16) + sizeof(float) * dim;
    int sz_half = VARHDRSZ + sizeof(int16) + sizeof(int16) + sizeof(uint16) * dim;

    if (sz == sz_half)
        return true;   /* halfvec: 2字节/元素 */
    if (sz == sz_vec)
        return false;  /* vector: 4字节/元素 */

    /* 如果都不匹配，说明数据格式有问题 */
    elog(ERROR, "unexpected datum size: dim=%d size=%d (expected vec=%d half=%d)",
         dim, sz, sz_vec, sz_half);
    return false;  /* 不会到达这里 */
}

/*
 * 针对 HalfVector 的专用熔断量化函数
 * 优势：
 * 1. 零内存分配 (Zero Allocation)
 * 2. 只需要遍历 2 次
 * 3. 避免了 FP16 -> FP32 的数组展开开销
 */
static inline void
HnswQuantizeHalfVector(HalfVector *src, Vector *dest)
{
	int dim = src->dim;
	half *hdata = src->x;

	/* dest->x 指向 Vector 数据区的起始位置 */
	char *buffer = (char *)dest->x;

	float min_v = 1e30f;
	float max_v = -1e30f;
	int i;

	/* 第一遍循环：直接读取 half 计算统计量 */
	for(i = 0; i < dim; i++) {
		float val = HalfToFloat4(hdata[i]); /* 寄存器内转换，不占内存 */
		if(val < min_v) min_v = val;
		if(val > max_v) max_v = val;
	}

	float scale, bias;
	if (max_v == min_v) {
		scale = 0.0f;
		bias = min_v;
	} else {
		scale = (max_v - min_v) / 255.0f;
		bias = min_v;
	}

	/* 序列化元数据 */
	memcpy(buffer, &scale, sizeof(float));
	memcpy(buffer + sizeof(float), &bias, sizeof(float));

	/* 定位量化数据写入区 */
	uint8_t *code_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

	/* 第二遍循环：直接读取 half -> 量化 -> 写入 */
	if (scale == 0.0f) {
		memset(code_ptr, 0, dim);
	} else {
		float inv_scale = 1.0f / scale;
		for (i = 0; i < dim; i++) {
			float val = HalfToFloat4(hdata[i]);
			int32_t q = (int32_t)((val - bias) * inv_scale + 0.5f);
			if (q < 0) q = 0; else if (q > 255) q = 255;
			code_ptr[i] = (uint8_t)q;
		}
	}

	/* 设置 Vector 头部信息，包含SQ8魔数标记 */
	dest->dim = dim;
	dest->unused = SQ8_TAG;  /* 添加SQ8识别标记 */
	SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) + sizeof(float)*2 + dim);
}

static inline void
QuantizeAndSerialize(Vector *src, Vector *dest)
{
    int dim = src->dim;
    float *fdata = src->x;  /* 修正：这里应该是float*，不是half* */

    /* dest->x 指向 Vector 数据区的起始位置 (Offset 8) */
    char *buffer = (char *)dest->x;

    float min_v = 1e30f;
    float max_v = -1e30f;
    for(int i=0; i<dim; i++) {
        float val = fdata[i];  /* 修正：直接使用float值 */
        if(val < min_v) min_v = val;
        if(val > max_v) max_v = val;
    }

    float scale = (max_v - min_v) / 255.0f;
    float bias = min_v;

    /* 序列化元数据 (Scale + Bias) */
    memcpy(buffer, &scale, sizeof(float));
    memcpy(buffer + sizeof(float), &bias, sizeof(float));

    /* 量化数据区从 buffer + 8 开始 */
    uint8_t *code_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        memset(code_ptr, 0, dim);
    } else {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < dim; i++) {
            float val = fdata[i];  /* 修正：直接使用float值 */
            int32_t q = (int32_t)((val - bias) * inv_scale + 0.5f);
            if (q < 0) q = 0; else if (q > 255) q = 255;
            code_ptr[i] = (uint8_t)q;
        }
    }

    dest->dim = dim;
    dest->unused = SQ8_TAG;  /* 添加SQ8识别标记 */
    /* 设置总长度：Scale(4) + Bias(4) + Data(dim) */
    SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) + sizeof(float)*2 + dim);
}

/*
 * 安全检测是否为SQ8量化数据
 * 使用多重验证避免误判，提高识别可靠性
 */
static inline bool
HnswIsSQ8(Datum d)
{
    Vector *v = (Vector *) PG_DETOAST_DATUM_PACKED(d);

    int dim = v->dim;
    if (dim <= 0 || dim > 2048)
        return false;

    /* 检查SQ8魔数标记 */
    if (v->unused != SQ8_TAG)
        return false;

    /* 检查数据大小是否符合SQ8格式 */
    int expected = VARHDRSZ + sizeof(int16) + sizeof(uint16) + sizeof(float)*2 + dim;
    if (VARSIZE_ANY(v) != expected)
        return false;

    /* 校验scale/bias是否为有限值 */
    char *buffer = (char *) v->x;
    float scale, bias;
    memcpy(&scale, buffer, sizeof(float));
    memcpy(&bias,  buffer + sizeof(float), sizeof(float));
    if (!isfinite(scale) || !isfinite(bias))
        return false;
    if (scale < 0.0f)  /* L2映射scale理应>=0 */
        return false;

    return true;
}

/*
 * 从 SQ8 量化的 Vector 中解码为 float 数组 (用于距离计算)
 */
static inline void
HnswDequantizeSQ8(Vector *quantized_vec, float *dequantized_data)
{
    int dim = quantized_vec->dim;
    char *buffer = (char *)quantized_vec->x;

    /* 提取 scale 和 bias */
    float scale, bias;
    memcpy(&scale, buffer, sizeof(float));
    memcpy(&bias, buffer + sizeof(float), sizeof(float));

    /* 解码量化数据 */
    uint8_t *code_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        /* 所有值都是 bias */
        for (int i = 0; i < dim; i++) {
            dequantized_data[i] = bias;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            dequantized_data[i] = (float)code_ptr[i] * scale + bias;
        }
    }
}

/*
 * 计算 SQ8 与 halfvec 之间的混合距离
 * a_sq8: SQ8量化的向量
 * b_half: 原始的halfvec向量
 * 返回: L2距离（不需要开方，用于排序）
 */
static inline double
HnswSQ8HalfvecDistance(Datum a_sq8, Datum b_half)
{
    Vector *sq8_vec = (Vector *) PG_DETOAST_DATUM_PACKED(a_sq8);
    HalfVector *half_vec = (HalfVector *) PG_DETOAST_DATUM_PACKED(b_half);

    int dim = sq8_vec->dim;
    if (half_vec->dim != dim)
        elog(ERROR, "dimension mismatch: sq8=%d halfvec=%d", dim, half_vec->dim);

    /* 解 SQ8 的 scale/bias + codes */
    char *buffer = (char *) sq8_vec->x;
    float scale, bias;
    memcpy(&scale, buffer, sizeof(float));
    memcpy(&bias,  buffer + sizeof(float), sizeof(float));
    uint8_t *code_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    double sum = 0.0;

    if (scale == 0.0f) {
        /* SQ8 全部是 bias */
        for (int i = 0; i < dim; i++) {
            float bval = HalfToFloat4(half_vec->x[i]);
            float diff = bias - bval;
            sum += (double)diff * (double)diff;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            float aval = (float)code_ptr[i] * scale + bias;
            float bval = HalfToFloat4(half_vec->x[i]);
            float diff = aval - bval;
            sum += (double)diff * (double)diff;
        }
    }

    return sum;  /* L2排序不需要sqrt */
}

/*
 * 计算两个 SQ8 量化向量之间的 L2 距离（优化版本，无内存分配）
 * 这个函数专门用于 HNSW 索引中的距离计算
 */
static inline double
HnswSQ8Distance(Datum a, Datum b)
{
    Vector *vec_a = (Vector *) PG_DETOAST_DATUM_PACKED(a);
    Vector *vec_b = (Vector *) PG_DETOAST_DATUM_PACKED(b);

    /* 检查维度一致性 */
    Assert(vec_a->dim == vec_b->dim);
    int dim = vec_a->dim;

    /* 解码第一个向量 */
    char *buffer_a = (char *) vec_a->x;
    float scale_a, bias_a;
    memcpy(&scale_a, buffer_a, sizeof(float));
    memcpy(&bias_a,  buffer_a + sizeof(float), sizeof(float));
    uint8_t *code_a = (uint8_t *)(buffer_a + sizeof(float) * 2);

    /* 解码第二个向量 */
    char *buffer_b = (char *) vec_b->x;
    float scale_b, bias_b;
    memcpy(&scale_b, buffer_b, sizeof(float));
    memcpy(&bias_b,  buffer_b + sizeof(float), sizeof(float));
    uint8_t *code_b = (uint8_t *)(buffer_b + sizeof(float) * 2);

    double sum = 0.0;

    /* 直接计算距离，避免分配临时数组 */
    if (scale_a == 0.0f && scale_b == 0.0f) {
        /* 两个向量都是常数 */
        float diff = bias_a - bias_b;
        sum = (double)dim * (double)diff * (double)diff;
    } else if (scale_a == 0.0f) {
        /* 向量A是常数 */
        for (int i = 0; i < dim; i++) {
            float bval = (float)code_b[i] * scale_b + bias_b;
            float diff = bias_a - bval;
            sum += (double)diff * (double)diff;
        }
    } else if (scale_b == 0.0f) {
        /* 向量B是常数 */
        for (int i = 0; i < dim; i++) {
            float aval = (float)code_a[i] * scale_a + bias_a;
            float diff = aval - bias_b;
            sum += (double)diff * (double)diff;
        }
    } else {
        /* 两个向量都是变量 */
        for (int i = 0; i < dim; i++) {
            float aval = (float)code_a[i] * scale_a + bias_a;
            float bval = (float)code_b[i] * scale_b + bias_b;
            float diff = aval - bval;
            sum += (double)diff * (double)diff;
        }
    }

    return sum;  /* L2排序不需要sqrt */
}

#endif
