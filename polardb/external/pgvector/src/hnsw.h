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
	pg_atomic_uint32 neighborVersion;  /* 邻居版本号，用于无锁读取 */
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

/* 量化类型枚举 - 通过操作符类选择 */
typedef enum HnswQuantType
{
	HNSW_QUANT_NONE = 0,   /* 不量化，保留原始格式 */
	HNSW_QUANT_HYBRID,     /* BQ+SQ8 混合量化 (当前默认) */
	HNSW_QUANT_SQ4,        /* SQ4 4-bit 量化 */
	HNSW_QUANT_BQ          /* 纯 Binary Quantization */
} HnswQuantType;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
	HnswQuantType quantType;  /* 量化类型，由操作符类决定 */
}			HnswTypeInfo;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	bool		allowQuantization;
	HnswQuantType quantType;  /* 量化类型，从 HnswTypeInfo 复制 */
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

/* 混合量化标记：BQ + SQ8 */
#define HYBRID_TAG 0x4859  /* 'HY' - Hybrid BQ+SQ8 */

/* SQ4 量化标记：4-bit 量化 */
#define SQ4_TAG 0x5134  /* 'Q4' - SQ4 量化标记 */

/* HNSW值格式枚举 */
typedef enum
{
	HNSW_FMT_RAW_FLOAT_VEC,    /* 原始vector：4字节float/元素 */
	HNSW_FMT_RAW_HALFVEC,      /* 原始halfvec：2字节half/元素 */
	HNSW_FMT_SQ8,              /* SQ8量化：scale(4) + bias(4) + codes(dim) */
	HNSW_FMT_HYBRID,           /* 混合量化：bq_bytes(2) + bq_data + scale(4) + bias(4) + sq8_codes */
	HNSW_FMT_SQ4,              /* SQ4量化：scale(4) + bias(4) + packed_codes(dim/2) */
	HNSW_FMT_UNKNOWN           /* 未知格式：bit, sparsevec等其他类型 */
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

	/* 混合量化：payload == 2 + bq_bytes + 8 + dim */
	/* bq_bytes = (dim + 7) / 8 */
	{
		int bq_bytes = (dim + 7) / 8;
		int hybrid_payload = sizeof(uint16_t) + bq_bytes + sizeof(float) * 2 + dim;
		if (payload == hybrid_payload) {
			Vector *v = (Vector *)p;
			if (v->unused == HYBRID_TAG) {
				return HNSW_FMT_HYBRID;
			}
		}
	}

	/* SQ4量化：payload == 8 + (dim+1)/2 (scale+bias+packed_codes) */
	/* 每2个维度打包成1个字节 */
	{
		int packed_bytes = (dim + 1) / 2;
		int sq4_payload = sizeof(float) * 2 + packed_bytes;
		if (payload == sq4_payload) {
			Vector *v = (Vector *)p;
			if (v->unused == SQ4_TAG) {
				return HNSW_FMT_SQ4;
			}
		}
	}

	/* 无法识别的格式 (bit, sparsevec等) - 返回UNKNOWN */
	return HNSW_FMT_UNKNOWN;
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
 * 混合量化函数：同时生成 BQ 和 SQ8
 * 内存布局: [bq_bytes:2][bq_data:bq_bytes][scale:4][bias:4][sq8_codes:dim]
 * 用于两阶段距离计算：BQ粗筛 + SQ8精排
 */
static inline void
HnswQuantizeHybrid(HalfVector *src, Vector *dest)
{
    int dim = src->dim;
    int bq_bytes = (dim + 7) / 8;  /* BQ 字节数，向上取整 */
    half *hdata = src->x;

    char *buffer = (char *)dest->x;

    /* 写入 BQ 字节数 */
    uint16_t bq_len = (uint16_t)bq_bytes;
    memcpy(buffer, &bq_len, sizeof(uint16_t));
    buffer += sizeof(uint16_t);

    /* BQ 数据区 */
    uint8_t *bq_ptr = (uint8_t *)buffer;
    memset(bq_ptr, 0, bq_bytes);

    /* 第一遍：计算 BQ（符号位量化）+ 统计 min/max */
    float min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < dim; i++) {
        float val = HalfToFloat4(hdata[i]);
        /* BQ: val > 0 -> 1, val <= 0 -> 0 */
        if (val > 0) {
            bq_ptr[i / 8] |= (1 << (7 - (i % 8)));
        }
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }
    buffer += bq_bytes;

    /* SQ8 量化参数 */
    float scale = (max_v == min_v) ? 0.0f : (max_v - min_v) / 255.0f;
    float bias = min_v;

    memcpy(buffer, &scale, sizeof(float));
    memcpy(buffer + sizeof(float), &bias, sizeof(float));

    /* SQ8 量化数据 */
    uint8_t *sq8_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        memset(sq8_ptr, 0, dim);
    } else {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < dim; i++) {
            float val = HalfToFloat4(hdata[i]);
            int32_t q = (int32_t)((val - bias) * inv_scale + 0.5f);
            if (q < 0) q = 0; else if (q > 255) q = 255;
            sq8_ptr[i] = (uint8_t)q;
        }
    }

    dest->dim = dim;
    dest->unused = HYBRID_TAG;
    /* 总长度: bq_bytes(2) + bq_data + scale(4) + bias(4) + sq8_codes(dim) */
    SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) +
                sizeof(uint16_t) + bq_bytes + sizeof(float)*2 + dim);
}

/* 从 Float Vector 生成混合量化 */
static inline void
HnswQuantizeHybridFromFloat(Vector *src, Vector *dest)
{
    int dim = src->dim;
    int bq_bytes = (dim + 7) / 8;
    float *fdata = src->x;

    char *buffer = (char *)dest->x;

    /* 写入 BQ 字节数 */
    uint16_t bq_len = (uint16_t)bq_bytes;
    memcpy(buffer, &bq_len, sizeof(uint16_t));
    buffer += sizeof(uint16_t);

    /* BQ 数据区 */
    uint8_t *bq_ptr = (uint8_t *)buffer;
    memset(bq_ptr, 0, bq_bytes);

    /* 第一遍：计算 BQ + 统计 min/max */
    float min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < dim; i++) {
        float val = fdata[i];
        if (val > 0) {
            bq_ptr[i / 8] |= (1 << (7 - (i % 8)));
        }
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }
    buffer += bq_bytes;

    /* SQ8 量化 */
    float scale = (max_v == min_v) ? 0.0f : (max_v - min_v) / 255.0f;
    float bias = min_v;

    memcpy(buffer, &scale, sizeof(float));
    memcpy(buffer + sizeof(float), &bias, sizeof(float));

    uint8_t *sq8_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        memset(sq8_ptr, 0, dim);
    } else {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < dim; i++) {
            float val = fdata[i];
            int32_t q = (int32_t)((val - bias) * inv_scale + 0.5f);
            if (q < 0) q = 0; else if (q > 255) q = 255;
            sq8_ptr[i] = (uint8_t)q;
        }
    }

    dest->dim = dim;
    dest->unused = HYBRID_TAG;
    SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) +
                sizeof(uint16_t) + bq_bytes + sizeof(float)*2 + dim);
}

/*
 * SQ4 量化函数：4-bit 量化
 * 内存布局: [scale:4][bias:4][packed_codes:(dim+1)/2]
 * 每2个维度打包成1个字节：高4位 = 第一个值，低4位 = 第二个值
 * 768维 -> 8 + 384 = 392 bytes (比 SQ8 压缩 50%)
 */
static inline void
HnswQuantizeToSQ4(HalfVector *src, Vector *dest)
{
    int dim = src->dim;
    int packed_bytes = (dim + 1) / 2;
    half *hdata = src->x;

    char *buffer = (char *)dest->x;

    /* 第一遍：统计 min/max */
    float min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < dim; i++) {
        float val = HalfToFloat4(hdata[i]);
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }

    /* SQ4 量化参数：映射到 [0, 15] */
    float scale = (max_v == min_v) ? 0.0f : (max_v - min_v) / 15.0f;
    float bias = min_v;

    memcpy(buffer, &scale, sizeof(float));
    memcpy(buffer + sizeof(float), &bias, sizeof(float));

    /* 打包的 4-bit 数据 */
    uint8_t *packed_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        memset(packed_ptr, 0, packed_bytes);
    } else {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < dim; i += 2) {
            /* 量化第一个值 */
            float val0 = HalfToFloat4(hdata[i]);
            int32_t q0 = (int32_t)((val0 - bias) * inv_scale + 0.5f);
            if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;

            /* 量化第二个值（如果存在） */
            int32_t q1 = 0;
            if (i + 1 < dim) {
                float val1 = HalfToFloat4(hdata[i + 1]);
                q1 = (int32_t)((val1 - bias) * inv_scale + 0.5f);
                if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
            }

            /* 打包：高4位 = q0, 低4位 = q1 */
            packed_ptr[i / 2] = (uint8_t)((q0 << 4) | q1);
        }
    }

    dest->dim = dim;
    dest->unused = SQ4_TAG;
    SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) + sizeof(float)*2 + packed_bytes);
}

/* 从 Float Vector 生成 SQ4 量化 */
static inline void
HnswQuantizeToSQ4FromFloat(Vector *src, Vector *dest)
{
    int dim = src->dim;
    int packed_bytes = (dim + 1) / 2;
    float *fdata = src->x;

    char *buffer = (char *)dest->x;

    /* 第一遍：统计 min/max */
    float min_v = 1e30f, max_v = -1e30f;
    for (int i = 0; i < dim; i++) {
        float val = fdata[i];
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }

    /* SQ4 量化参数：映射到 [0, 15] */
    float scale = (max_v == min_v) ? 0.0f : (max_v - min_v) / 15.0f;
    float bias = min_v;

    memcpy(buffer, &scale, sizeof(float));
    memcpy(buffer + sizeof(float), &bias, sizeof(float));

    /* 打包的 4-bit 数据 */
    uint8_t *packed_ptr = (uint8_t *)(buffer + sizeof(float) * 2);

    if (scale == 0.0f) {
        memset(packed_ptr, 0, packed_bytes);
    } else {
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < dim; i += 2) {
            /* 量化第一个值 */
            float val0 = fdata[i];
            int32_t q0 = (int32_t)((val0 - bias) * inv_scale + 0.5f);
            if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;

            /* 量化第二个值（如果存在） */
            int32_t q1 = 0;
            if (i + 1 < dim) {
                float val1 = fdata[i + 1];
                q1 = (int32_t)((val1 - bias) * inv_scale + 0.5f);
                if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;
            }

            /* 打包：高4位 = q0, 低4位 = q1 */
            packed_ptr[i / 2] = (uint8_t)((q0 << 4) | q1);
        }
    }

    dest->dim = dim;
    dest->unused = SQ4_TAG;
    SET_VARSIZE(dest, VARHDRSZ + sizeof(int16) + sizeof(uint16) + sizeof(float)*2 + packed_bytes);
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
 * 【AVX-512优化】SQ8 vs SQ8距离计算配置
 *
 * 强制启用AVX-512优化（根据评测环境CPU支持情况）
 * 如果需要兼容性，可以设为0启用运行期检测
 */
#ifndef HNSW_SQ8_FORCE_AVX512
#define HNSW_SQ8_FORCE_AVX512 1
#endif

/* ---- 标量版本：保留原有逻辑，作为 fallback ---- */
static inline float
HnswSQ8Distance2_Vector_scalar(Vector *a, Vector *b)
{
    int dim = a->dim;

    char *bufa = (char *) a->x;
    char *bufb = (char *) b->x;

    float sa, ba, sb, bb;
    memcpy(&sa, bufa, sizeof(float));
    memcpy(&ba, bufa + sizeof(float), sizeof(float));
    memcpy(&sb, bufb, sizeof(float));
    memcpy(&bb, bufb + sizeof(float), sizeof(float));

    uint8_t *qa = (uint8_t *)(bufa + sizeof(float) * 2);
    uint8_t *qb = (uint8_t *)(bufb + sizeof(float) * 2);

    double sum = 0.0;

    if (sa == 0.0f && sb == 0.0f) {
        float diff = ba - bb;
        sum = (double)dim * (double)diff * (double)diff;
    } else if (sa == 0.0f) {
        for (int i = 0; i < dim; i++) {
            float vb = (float)qb[i] * sb + bb;
            float d = ba - vb;
            sum += (double)d * (double)d;
        }
    } else if (sb == 0.0f) {
        for (int i = 0; i < dim; i++) {
            float va = (float)qa[i] * sa + ba;
            float d = va - bb;
            sum += (double)d * (double)d;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            float va = (float)qa[i] * sa + ba;
            float vb = (float)qb[i] * sb + bb;
            float d = va - vb;
            sum += (double)d * (double)d;
        }
    }

    return (float)sum;
}

/* ---- AVX-512 reduce add：避免编译器版本依赖 ---- */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static inline float
hnsw_reduce_add_ps(__m512 v)
{
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s  = _mm256_add_ps(lo, hi);
    __m128 lo2 = _mm256_castps256_ps128(s);
    __m128 hi2 = _mm256_extractf128_ps(s, 1);
    __m128 s2 = _mm_add_ps(lo2, hi2);
    s2 = _mm_hadd_ps(s2, s2);
    s2 = _mm_hadd_ps(s2, s2);
    return _mm_cvtss_f32(s2);
}

/* ---- AVX-512版本：SQ8 vs SQ8的平方L2距离 ---- */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
static inline float
HnswSQ8Distance2_Vector_avx512(Vector *a, Vector *b)
{
    int dim = a->dim;

    char *bufa = (char *) a->x;
    char *bufb = (char *) b->x;

    float sa, ba, sb, bb;
    memcpy(&sa, bufa, sizeof(float));
    memcpy(&ba, bufa + sizeof(float), sizeof(float));
    memcpy(&sb, bufb, sizeof(float));
    memcpy(&bb, bufb + sizeof(float), sizeof(float));

    uint8_t *qa = (uint8_t *)(bufa + sizeof(float) * 2);
    uint8_t *qb = (uint8_t *)(bufb + sizeof(float) * 2);

    /* 常数向量快速路径 */
    if (sa == 0.0f && sb == 0.0f) {
        float diff = ba - bb;
        return (float)dim * diff * diff;
    }

    __m512 vsum = _mm512_setzero_ps();
    int i = 0;

    if (sa == 0.0f) {
        /* A常数，B变量：d = ba - (qb*sb + bb) */
        __m512 v_ba = _mm512_set1_ps(ba);
        __m512 v_sb = _mm512_set1_ps(sb);
        __m512 v_bb = _mm512_set1_ps(bb);

        /* 32路展开（2x16） */
        for (; i <= dim - 32; i += 32) {
            __m128i ub1 = _mm_loadu_si128((const __m128i *)(qb + i));
            __m128i ub2 = _mm_loadu_si128((const __m128i *)(qb + i + 16));

            __m512 fb1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub1));
            __m512 fb2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub2));

            __m512 vb1 = _mm512_fmadd_ps(fb1, v_sb, v_bb);
            __m512 vb2 = _mm512_fmadd_ps(fb2, v_sb, v_bb);

            __m512 d1 = _mm512_sub_ps(v_ba, vb1);
            __m512 d2 = _mm512_sub_ps(v_ba, vb2);

            vsum = _mm512_fmadd_ps(d1, d1, vsum);
            vsum = _mm512_fmadd_ps(d2, d2, vsum);
        }

        /* 16路处理剩余 */
        for (; i <= dim - 16; i += 16) {
            __m128i ub = _mm_loadu_si128((const __m128i *)(qb + i));
            __m512 fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub));
            __m512 vb = _mm512_fmadd_ps(fb, v_sb, v_bb);
            __m512 d  = _mm512_sub_ps(v_ba, vb);
            vsum = _mm512_fmadd_ps(d, d, vsum);
        }

        /* 处理尾部 */
        int rem = dim - i;
        if (rem > 0) {
            __mmask16 m = (1u << rem) - 1u;
            __m128i ub = _mm_maskz_loadu_epi8(m, (const void *)(qb + i));
            __m512 fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub));
            __m512 vb = _mm512_fmadd_ps(fb, v_sb, v_bb);
            __m512 d  = _mm512_sub_ps(v_ba, vb);
            __m512 d2 = _mm512_mul_ps(d, d);
            vsum = _mm512_mask_add_ps(vsum, m, vsum, d2);
        }

        return hnsw_reduce_add_ps(vsum);
    }

    if (sb == 0.0f) {
        /* B常数，A变量：d = (qa*sa + ba) - bb */
        __m512 v_sa = _mm512_set1_ps(sa);
        __m512 v_ba = _mm512_set1_ps(ba);
        __m512 v_bb = _mm512_set1_ps(bb);

        for (; i <= dim - 32; i += 32) {
            __m128i ua1 = _mm_loadu_si128((const __m128i *)(qa + i));
            __m128i ua2 = _mm_loadu_si128((const __m128i *)(qa + i + 16));

            __m512 fa1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua1));
            __m512 fa2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua2));

            __m512 va1 = _mm512_fmadd_ps(fa1, v_sa, v_ba);
            __m512 va2 = _mm512_fmadd_ps(fa2, v_sa, v_ba);

            __m512 d1 = _mm512_sub_ps(va1, v_bb);
            __m512 d2 = _mm512_sub_ps(va2, v_bb);

            vsum = _mm512_fmadd_ps(d1, d1, vsum);
            vsum = _mm512_fmadd_ps(d2, d2, vsum);
        }

        for (; i <= dim - 16; i += 16) {
            __m128i ua = _mm_loadu_si128((const __m128i *)(qa + i));
            __m512 fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua));
            __m512 va = _mm512_fmadd_ps(fa, v_sa, v_ba);
            __m512 d  = _mm512_sub_ps(va, v_bb);
            vsum = _mm512_fmadd_ps(d, d, vsum);
        }

        int rem = dim - i;
        if (rem > 0) {
            __mmask16 m = (1u << rem) - 1u;
            __m128i ua = _mm_maskz_loadu_epi8(m, (const void *)(qa + i));
            __m512 fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua));
            __m512 va = _mm512_fmadd_ps(fa, v_sa, v_ba);
            __m512 d  = _mm512_sub_ps(va, v_bb);
            __m512 d2 = _mm512_mul_ps(d, d);
            vsum = _mm512_mask_add_ps(vsum, m, vsum, d2);
        }

        return hnsw_reduce_add_ps(vsum);
    }

    /* 最常见情况：sa!=0 && sb!=0 */
    __m512 v_sa = _mm512_set1_ps(sa);
    __m512 v_ba = _mm512_set1_ps(ba);
    __m512 v_sb = _mm512_set1_ps(sb);
    __m512 v_bb = _mm512_set1_ps(bb);

    for (; i <= dim - 32; i += 32) {
        __m128i ua1 = _mm_loadu_si128((const __m128i *)(qa + i));
        __m128i ub1 = _mm_loadu_si128((const __m128i *)(qb + i));
        __m128i ua2 = _mm_loadu_si128((const __m128i *)(qa + i + 16));
        __m128i ub2 = _mm_loadu_si128((const __m128i *)(qb + i + 16));

        __m512 fa1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua1));
        __m512 fb1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub1));
        __m512 fa2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua2));
        __m512 fb2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub2));

        __m512 va1 = _mm512_fmadd_ps(fa1, v_sa, v_ba);
        __m512 vb1 = _mm512_fmadd_ps(fb1, v_sb, v_bb);
        __m512 va2 = _mm512_fmadd_ps(fa2, v_sa, v_ba);
        __m512 vb2 = _mm512_fmadd_ps(fb2, v_sb, v_bb);

        __m512 d1 = _mm512_sub_ps(va1, vb1);
        __m512 d2 = _mm512_sub_ps(va2, vb2);

        vsum = _mm512_fmadd_ps(d1, d1, vsum);
        vsum = _mm512_fmadd_ps(d2, d2, vsum);
    }

    for (; i <= dim - 16; i += 16) {
        __m128i ua = _mm_loadu_si128((const __m128i *)(qa + i));
        __m128i ub = _mm_loadu_si128((const __m128i *)(qb + i));

        __m512 fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua));
        __m512 fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub));

        __m512 va = _mm512_fmadd_ps(fa, v_sa, v_ba);
        __m512 vb = _mm512_fmadd_ps(fb, v_sb, v_bb);

        __m512 d = _mm512_sub_ps(va, vb);
        vsum = _mm512_fmadd_ps(d, d, vsum);
    }

    int rem = dim - i;
    if (rem > 0) {
        __mmask16 m = (1u << rem) - 1u;

        __m128i ua = _mm_maskz_loadu_epi8(m, (const void *)(qa + i));
        __m128i ub = _mm_maskz_loadu_epi8(m, (const void *)(qb + i));

        __m512 fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ua));
        __m512 fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(ub));

        __m512 va = _mm512_fmadd_ps(fa, v_sa, v_ba);
        __m512 vb = _mm512_fmadd_ps(fb, v_sb, v_bb);
        __m512 d  = _mm512_sub_ps(va, vb);

        __m512 d2 = _mm512_mul_ps(d, d);
        vsum = _mm512_mask_add_ps(vsum, m, vsum, d2);
    }

    return hnsw_reduce_add_ps(vsum);
}

/* ---- 运行期CPU特性检测：缓存检测结果 ---- */
static inline int
HnswAvx512EnabledCached(void)
{
#if defined(__x86_64__) && defined(__GNUC__)
    static int cached = -1;
    if (__builtin_expect(cached < 0, 0)) {
        cached = (__builtin_cpu_supports("avx512f") &&
                  __builtin_cpu_supports("avx512bw") &&
                  __builtin_cpu_supports("avx512vl") &&
                  __builtin_cpu_supports("avx512dq") &&
                  __builtin_cpu_supports("fma")) ? 1 : 0;
    }
    return cached;
#else
    return 0;
#endif
}

/*
 * 【主入口】AVX-512优化的SQ8距离计算
 * 保持原有签名和返回语义，内部自动选择最优实现
 */
static inline float
HnswSQ8Distance2_Vector(Vector *a, Vector *b)
{
#if HNSW_SQ8_FORCE_AVX512
    return HnswSQ8Distance2_Vector_avx512(a, b);
#else
    if (__builtin_expect(HnswAvx512EnabledCached(), 1))
        return HnswSQ8Distance2_Vector_avx512(a, b);
    else
        return HnswSQ8Distance2_Vector_scalar(a, b);
#endif
}

/* ---- 256-bit 水平求和：用于 _mm512 的 reduce ---- */
__attribute__((target("avx,fma,avx2")))
static inline float
hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);

    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0xAA));
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static inline float
hsum512_ps(__m512 v)
{
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum = _mm256_add_ps(lo, hi);
    return hsum256_ps(sum);
}

/*
 * 【AVX-512优化】SQ8 vs halfvec混合距离计算（精确匹配HalfVector定义）
 *
 * 说明：
 * - sq8: Vector*，其中 sq8->x 指向 payload，布局为：
 *   [float scale][float bias][uint8 q[dim]]
 * - hv : HalfVector*，hv->x 是 half(=uint16) 数组
 *
 * 返回：L2 平方距离（不做 sqrt）
 */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static inline float
HnswSQ8HalfvecDistance2_AVX512(const Vector *sq8, const HalfVector *hv)
{
    const int dim = (int)sq8->dim;

    /* payload 起点：sq8->x 被当作字节数组使用 */
    const char *buf = (const char *)sq8->x;

    float s, b;
    memcpy(&s, buf, sizeof(float));
    memcpy(&b, buf + sizeof(float), sizeof(float));

    const uint8_t *q = (const uint8_t *)(buf + sizeof(float) * 2);
    const half *hx = (const half *)hv->x; /* half == uint16 */

    __m512 v_sum = _mm512_setzero_ps();

    /* s==0：常数向量（全是 bias），不需要读取 q */
    if (s == 0.0f)
    {
        const __m512 v_bias = _mm512_set1_ps(b);

        int i = 0;
        for (; i + 16 <= dim; i += 16)
        {
            /* load 16 half(=uint16) -> convert to 16 float */
            __m256i h16 = _mm256_loadu_si256((const __m256i *)(hx + i));
            __m512  v_h = _mm512_cvtph_ps(h16);

            __m512 v_diff = _mm512_sub_ps(v_h, v_bias);
            v_sum = _mm512_fmadd_ps(v_diff, v_diff, v_sum);
        }

        /* tail (<16) */
        if (i < dim)
        {
            const int remain = dim - i;
            const __mmask16 mask = (__mmask16)((1u << remain) - 1u);

            __m256i h16 = _mm256_maskz_loadu_epi16(mask, (const void *)(hx + i));
            __m512  v_h = _mm512_cvtph_ps(h16);

            __m512 v_diff = _mm512_sub_ps(v_h, v_bias);

            /* 只对 mask 内 lanes 累加 */
            __m512 v_sq = _mm512_mul_ps(v_diff, v_diff);
            v_sum = _mm512_add_ps(v_sum, _mm512_maskz_mov_ps(mask, v_sq));
        }

        return hsum512_ps(v_sum);
    }

    /* 一般情况：SQ8 解码 + half 转 float */
    const __m512 v_scale = _mm512_set1_ps(s);
    const __m512 v_bias  = _mm512_set1_ps(b);

    int i = 0;
    for (; i + 16 <= dim; i += 16)
    {
        /* ---- SQ8: load 16 bytes -> u8 to i32 -> float ---- */
        __m128i u8_16 = _mm_loadu_si128((const __m128i *)(q + i));
        __m512i i32   = _mm512_cvtepu8_epi32(u8_16);
        __m512  v_q8  = _mm512_cvtepi32_ps(i32);

        /* dequant: v_rec = v_q8 * scale + bias */
        __m512 v_rec = _mm512_fmadd_ps(v_q8, v_scale, v_bias);

        /* ---- halfvec: load 16 half -> float ---- */
        __m256i h16 = _mm256_loadu_si256((const __m256i *)(hx + i));
        __m512  v_h = _mm512_cvtph_ps(h16);

        /* diff + accumulate */
        __m512 v_diff = _mm512_sub_ps(v_h, v_rec);
        v_sum = _mm512_fmadd_ps(v_diff, v_diff, v_sum);
    }

    /* tail (<16) */
    if (i < dim)
    {
        const int remain = dim - i;
        const __mmask16 mask = (__mmask16)((1u << remain) - 1u);

        __m128i u8_16 = _mm_maskz_loadu_epi8(mask, (const void *)(q + i));
        __m512i i32   = _mm512_cvtepu8_epi32(u8_16);
        __m512  v_q8  = _mm512_cvtepi32_ps(i32);
        __m512  v_rec = _mm512_fmadd_ps(v_q8, v_scale, v_bias);

        __m256i h16 = _mm256_maskz_loadu_epi16(mask, (const void *)(hx + i));
        __m512  v_h = _mm512_cvtph_ps(h16);

        __m512 v_diff = _mm512_sub_ps(v_h, v_rec);

        __m512 v_sq = _mm512_mul_ps(v_diff, v_diff);
        v_sum = _mm512_add_ps(v_sum, _mm512_maskz_mov_ps(mask, v_sq));
    }

    return hsum512_ps(v_sum);
}

/*
 * 【主入口】AVX-512优化的SQ8 vs halfvec混合距离计算
 * 保持原有签名不变，内部直接调用AVX-512版本
 */
static inline float
HnswSQ8HalfvecDistance2(Vector *sq8, HalfVector *hv)
{
    /* 维度检查：保持原语义 */
    if ((int)hv->dim != (int)sq8->dim)
        elog(ERROR, "dimension mismatch: sq8=%d halfvec=%d", sq8->dim, hv->dim);

    return HnswSQ8HalfvecDistance2_AVX512(sq8, hv);
}

/*
 * 计算 SQ8 与 halfvec 之间的混合距离（保留版本，用于兼容）
 */
static inline double
HnswSQ8HalfvecDistance(Datum a_sq8, Datum b_half)
{
    Vector *sq8_vec = (Vector *) PG_DETOAST_DATUM_PACKED(a_sq8);
    HalfVector *half_vec = (HalfVector *) PG_DETOAST_DATUM_PACKED(b_half);

    int dim = sq8_vec->dim;
    if (half_vec->dim != dim)
        elog(ERROR, "dimension mismatch: sq8=%d halfvec=%d", dim, half_vec->dim);

    /* 调用优化版本 */
    return (double)HnswSQ8HalfvecDistance2(sq8_vec, half_vec);
}

/*
 * 计算两个 SQ8 量化向量之间的距离（零分配版本）
 * 直接调用优化版本，返回平方距离
 */
static inline double
HnswSQ8Distance(Datum a, Datum b)
{
    Vector *vec_a = (Vector *) PG_DETOAST_DATUM_PACKED(a);
    Vector *vec_b = (Vector *) PG_DETOAST_DATUM_PACKED(b);

    /* 检查维度一致性 */
    Assert(vec_a->dim == vec_b->dim);

    /* 直接调用零分配优化版本 */
    return (double)HnswSQ8Distance2_Vector(vec_a, vec_b);
}

/* ========== 混合量化 (BQ+SQ8) 距离计算函数 ========== */

/*
 * 【AVX-512优化】BQ Hamming 距离计算
 * 使用 AVX-512 VPOPCNT 指令实现快速 popcount
 */
__attribute__((target("avx512f,avx512vpopcntdq")))
static inline uint64_t
HnswBQHammingDistance_AVX512(const uint8_t *a, const uint8_t *b, int bq_bytes)
{
    __m512i dist = _mm512_setzero_si512();
    int i = 0;

    /* 64 字节（512 位）并行处理 */
    for (; i + 64 <= bq_bytes; i += 64) {
        __m512i va = _mm512_loadu_si512((const __m512i *)(a + i));
        __m512i vb = _mm512_loadu_si512((const __m512i *)(b + i));
        __m512i vxor = _mm512_xor_si512(va, vb);
        dist = _mm512_add_epi64(dist, _mm512_popcnt_epi64(vxor));
    }

    uint64_t distance = _mm512_reduce_add_epi64(dist);

    /* 处理尾部（标量） */
    for (; i < bq_bytes; i++) {
        distance += __builtin_popcount(a[i] ^ b[i]);
    }

    return distance;
}

/* 标量版本 BQ Hamming 距离（fallback） */
static inline uint64_t
HnswBQHammingDistance_Scalar(const uint8_t *a, const uint8_t *b, int bq_bytes)
{
    uint64_t distance = 0;
    int i = 0;

    /* 8 字节并行处理 */
    for (; i + 8 <= bq_bytes; i += 8) {
        uint64_t va, vb;
        memcpy(&va, a + i, 8);
        memcpy(&vb, b + i, 8);
        distance += __builtin_popcountll(va ^ vb);
    }

    /* 处理尾部 */
    for (; i < bq_bytes; i++) {
        distance += __builtin_popcount(a[i] ^ b[i]);
    }

    return distance;
}

/* BQ Hamming 距离主入口 */
static inline uint64_t
HnswBQHammingDistance(const uint8_t *a, const uint8_t *b, int bq_bytes)
{
#if HNSW_SQ8_FORCE_AVX512
    return HnswBQHammingDistance_AVX512(a, b, bq_bytes);
#else
    if (__builtin_expect(HnswAvx512EnabledCached(), 1))
        return HnswBQHammingDistance_AVX512(a, b, bq_bytes);
    else
        return HnswBQHammingDistance_Scalar(a, b, bq_bytes);
#endif
}

/*
 * 从混合量化数据提取 BQ 部分并计算 Hamming 距离
 * 用于粗筛阶段
 */
static inline double
HnswHybridBQDistance(Datum a, Datum b)
{
    Vector *va = (Vector *)PG_DETOAST_DATUM_PACKED(a);
    Vector *vb = (Vector *)PG_DETOAST_DATUM_PACKED(b);

    Assert(va->unused == HYBRID_TAG && vb->unused == HYBRID_TAG);
    Assert(va->dim == vb->dim);

    char *bufa = (char *)va->x;
    char *bufb = (char *)vb->x;

    uint16_t bq_bytes_a, bq_bytes_b;
    memcpy(&bq_bytes_a, bufa, sizeof(uint16_t));
    memcpy(&bq_bytes_b, bufb, sizeof(uint16_t));

    Assert(bq_bytes_a == bq_bytes_b);

    const uint8_t *bq_a = (const uint8_t *)(bufa + sizeof(uint16_t));
    const uint8_t *bq_b = (const uint8_t *)(bufb + sizeof(uint16_t));

    return (double)HnswBQHammingDistance(bq_a, bq_b, bq_bytes_a);
}

/*
 * 从混合量化数据提取 SQ8 部分并计算 L2 距离
 * 用于精排阶段
 */
static inline double
HnswHybridSQ8Distance(Datum a, Datum b)
{
    Vector *va = (Vector *)PG_DETOAST_DATUM_PACKED(a);
    Vector *vb = (Vector *)PG_DETOAST_DATUM_PACKED(b);

    Assert(va->unused == HYBRID_TAG && vb->unused == HYBRID_TAG);
    Assert(va->dim == vb->dim);

    int dim = va->dim;
    int bq_bytes = (dim + 7) / 8;

    char *bufa = (char *)va->x;
    char *bufb = (char *)vb->x;

    /* 跳过 bq_bytes(2) 和 bq_data，定位到 SQ8 数据 */
    char *sq8_a = bufa + sizeof(uint16_t) + bq_bytes;
    char *sq8_b = bufb + sizeof(uint16_t) + bq_bytes;

    /* 提取 SQ8 参数 */
    float sa, ba, sb, bb;
    memcpy(&sa, sq8_a, sizeof(float));
    memcpy(&ba, sq8_a + sizeof(float), sizeof(float));
    memcpy(&sb, sq8_b, sizeof(float));
    memcpy(&bb, sq8_b + sizeof(float), sizeof(float));

    const uint8_t *qa = (const uint8_t *)(sq8_a + sizeof(float) * 2);
    const uint8_t *qb = (const uint8_t *)(sq8_b + sizeof(float) * 2);

    /* 计算 L2 距离 */
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        float val_a = (float)qa[i] * sa + ba;
        float val_b = (float)qb[i] * sb + bb;
        float diff = val_a - val_b;
        sum += (double)diff * (double)diff;
    }

    return sum;
}

/*
 * 混合距离计算主入口
 * coarse=true: 返回 BQ Hamming 距离（用于粗筛）
 * coarse=false: 返回 SQ8 L2 距离（用于精排）
 */
static inline double
HnswHybridDistance(Datum a, Datum b, bool coarse)
{
    if (coarse) {
        return HnswHybridBQDistance(a, b);
    } else {
        return HnswHybridSQ8Distance(a, b);
    }
}

/*
 * HYBRID 格式 vs halfvec 距离计算
 * HYBRID 内存布局: [bq_bytes:2][bq_data:bq_bytes][scale:4][bias:4][sq8_codes:dim]
 * 需要跳过 BQ 部分，定位到 SQ8 部分计算距离
 */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static inline float
HnswHybridHalfvecDistance_AVX512(const Vector *hybrid, const HalfVector *hv)
{
    const int dim = (int)hybrid->dim;
    const char *buf = (const char *)hybrid->x;

    /* 读取 bq_bytes 并跳过 BQ 数据 */
    uint16_t bq_bytes;
    memcpy(&bq_bytes, buf, sizeof(uint16_t));

    /* 定位到 SQ8 部分：跳过 bq_bytes(2) + bq_data(bq_bytes) */
    const char *sq8_buf = buf + sizeof(uint16_t) + bq_bytes;

    /* 读取 SQ8 参数 */
    float s, b;
    memcpy(&s, sq8_buf, sizeof(float));
    memcpy(&b, sq8_buf + sizeof(float), sizeof(float));

    const uint8_t *q = (const uint8_t *)(sq8_buf + sizeof(float) * 2);
    const half *hx = (const half *)hv->x;

    __m512 v_sum = _mm512_setzero_ps();

    if (s == 0.0f)
    {
        const __m512 v_bias = _mm512_set1_ps(b);

        int i = 0;
        for (; i + 16 <= dim; i += 16)
        {
            __m256i h16 = _mm256_loadu_si256((const __m256i *)(hx + i));
            __m512  v_h = _mm512_cvtph_ps(h16);

            __m512 v_diff = _mm512_sub_ps(v_h, v_bias);
            v_sum = _mm512_fmadd_ps(v_diff, v_diff, v_sum);
        }

        if (i < dim)
        {
            const int remain = dim - i;
            const __mmask16 mask = (__mmask16)((1u << remain) - 1u);

            __m256i h16 = _mm256_maskz_loadu_epi16(mask, (const void *)(hx + i));
            __m512  v_h = _mm512_cvtph_ps(h16);

            __m512 v_diff = _mm512_sub_ps(v_h, v_bias);
            __m512 v_sq = _mm512_mul_ps(v_diff, v_diff);
            v_sum = _mm512_add_ps(v_sum, _mm512_maskz_mov_ps(mask, v_sq));
        }

        return hsum512_ps(v_sum);
    }

    /* 正常情况：scale != 0 */
    const __m512 v_scale = _mm512_set1_ps(s);
    const __m512 v_bias  = _mm512_set1_ps(b);

    int i = 0;
    for (; i + 16 <= dim; i += 16)
    {
        /* 加载 16 个 SQ8 codes -> 转换为 float */
        __m128i q16 = _mm_loadu_si128((const __m128i *)(q + i));
        __m512i q32 = _mm512_cvtepu8_epi32(q16);
        __m512  v_q = _mm512_cvtepi32_ps(q32);
        __m512  v_dequant = _mm512_fmadd_ps(v_q, v_scale, v_bias);

        /* 加载 16 个 half -> 转换为 float */
        __m256i h16 = _mm256_loadu_si256((const __m256i *)(hx + i));
        __m512  v_h = _mm512_cvtph_ps(h16);

        __m512 v_diff = _mm512_sub_ps(v_dequant, v_h);
        v_sum = _mm512_fmadd_ps(v_diff, v_diff, v_sum);
    }

    /* 处理尾部 */
    if (i < dim)
    {
        const int remain = dim - i;
        const __mmask16 mask = (__mmask16)((1u << remain) - 1u);

        __m128i q16 = _mm_maskz_loadu_epi8(mask, (const void *)(q + i));
        __m512i q32 = _mm512_cvtepu8_epi32(q16);
        __m512  v_q = _mm512_cvtepi32_ps(q32);
        __m512  v_dequant = _mm512_fmadd_ps(v_q, v_scale, v_bias);

        __m256i h16 = _mm256_maskz_loadu_epi16(mask, (const void *)(hx + i));
        __m512  v_h = _mm512_cvtph_ps(h16);

        __m512 v_diff = _mm512_sub_ps(v_dequant, v_h);
        __m512 v_sq = _mm512_mul_ps(v_diff, v_diff);
        v_sum = _mm512_add_ps(v_sum, _mm512_maskz_mov_ps(mask, v_sq));
    }

    return hsum512_ps(v_sum);
}

/*
 * HYBRID vs halfvec 距离计算主入口
 */
static inline float
HnswHybridHalfvecDistance(Vector *hybrid, HalfVector *hv)
{
    if ((int)hv->dim != (int)hybrid->dim)
        elog(ERROR, "dimension mismatch: hybrid=%d halfvec=%d", hybrid->dim, hv->dim);

    return HnswHybridHalfvecDistance_AVX512(hybrid, hv);
}

/*
 * SQ4 距离计算 - 标量版本
 * 内存布局: [scale:4][bias:4][packed_codes:(dim+1)/2]
 * 每个字节：高4位 = 第一个值，低4位 = 第二个值
 */
static inline double
HnswSQ4Distance_Scalar(Datum a, Datum b)
{
    Vector *va = (Vector *)PG_DETOAST_DATUM_PACKED(a);
    Vector *vb = (Vector *)PG_DETOAST_DATUM_PACKED(b);

    Assert(va->unused == SQ4_TAG && vb->unused == SQ4_TAG);
    Assert(va->dim == vb->dim);

    int dim = va->dim;
    int packed_bytes = (dim + 1) / 2;

    char *bufa = (char *)va->x;
    char *bufb = (char *)vb->x;

    /* 提取 SQ4 参数 */
    float sa, ba, sb, bb;
    memcpy(&sa, bufa, sizeof(float));
    memcpy(&ba, bufa + sizeof(float), sizeof(float));
    memcpy(&sb, bufb, sizeof(float));
    memcpy(&bb, bufb + sizeof(float), sizeof(float));

    const uint8_t *pa = (const uint8_t *)(bufa + sizeof(float) * 2);
    const uint8_t *pb = (const uint8_t *)(bufb + sizeof(float) * 2);

    /* 计算 L2 距离 */
    double sum = 0.0;
    for (int i = 0; i < packed_bytes; i++) {
        uint8_t byte_a = pa[i];
        uint8_t byte_b = pb[i];

        /* 解包高4位 */
        int q0a = (byte_a >> 4) & 0x0F;
        int q0b = (byte_b >> 4) & 0x0F;
        float val_a0 = (float)q0a * sa + ba;
        float val_b0 = (float)q0b * sb + bb;
        float diff0 = val_a0 - val_b0;
        sum += (double)diff0 * (double)diff0;

        /* 解包低4位（检查边界） */
        int idx1 = i * 2 + 1;
        if (idx1 < dim) {
            int q1a = byte_a & 0x0F;
            int q1b = byte_b & 0x0F;
            float val_a1 = (float)q1a * sa + ba;
            float val_b1 = (float)q1b * sb + bb;
            float diff1 = val_a1 - val_b1;
            sum += (double)diff1 * (double)diff1;
        }
    }

    return sum;
}

/*
 * SQ4 距离计算 - AVX-512 优化版本
 * 每次处理 64 个打包字节 = 128 个维度
 */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
static inline double
HnswSQ4Distance_AVX512(Datum a, Datum b)
{
    Vector *va = (Vector *)PG_DETOAST_DATUM_PACKED(a);
    Vector *vb = (Vector *)PG_DETOAST_DATUM_PACKED(b);

    Assert(va->unused == SQ4_TAG && vb->unused == SQ4_TAG);
    Assert(va->dim == vb->dim);

    int dim = va->dim;
    int packed_bytes = (dim + 1) / 2;

    char *bufa = (char *)va->x;
    char *bufb = (char *)vb->x;

    /* 提取 SQ4 参数 */
    float sa, ba, sb, bb;
    memcpy(&sa, bufa, sizeof(float));
    memcpy(&ba, bufa + sizeof(float), sizeof(float));
    memcpy(&sb, bufb, sizeof(float));
    memcpy(&bb, bufb + sizeof(float), sizeof(float));

    const uint8_t *pa = (const uint8_t *)(bufa + sizeof(float) * 2);
    const uint8_t *pb = (const uint8_t *)(bufb + sizeof(float) * 2);

    __m512 sum_vec = _mm512_setzero_ps();
    __m512 vsa = _mm512_set1_ps(sa);
    __m512 vba = _mm512_set1_ps(ba);
    __m512 vsb = _mm512_set1_ps(sb);
    __m512 vbb = _mm512_set1_ps(bb);

    int i = 0;
    /* 每次处理 8 个打包字节 = 16 个维度 */
    for (; i + 8 <= packed_bytes; i += 8) {
        /* 加载 8 字节打包数据 */
        uint64_t bytes_a, bytes_b;
        memcpy(&bytes_a, pa + i, 8);
        memcpy(&bytes_b, pb + i, 8);

        /* 手动解包16个4-bit值到16个int32 */
        int32_t qa[16], qb[16];
        for (int j = 0; j < 8; j++) {
            uint8_t ba_byte = (bytes_a >> (j * 8)) & 0xFF;
            uint8_t bb_byte = (bytes_b >> (j * 8)) & 0xFF;
            qa[j * 2]     = (ba_byte >> 4) & 0x0F;
            qa[j * 2 + 1] = ba_byte & 0x0F;
            qb[j * 2]     = (bb_byte >> 4) & 0x0F;
            qb[j * 2 + 1] = bb_byte & 0x0F;
        }

        /* 转换为 float 并计算距离 */
        __m512i vqa = _mm512_loadu_si512((__m512i *)qa);
        __m512i vqb = _mm512_loadu_si512((__m512i *)qb);
        __m512 vfa = _mm512_cvtepi32_ps(vqa);
        __m512 vfb = _mm512_cvtepi32_ps(vqb);

        /* dequantize: val = q * scale + bias */
        __m512 val_a = _mm512_fmadd_ps(vfa, vsa, vba);
        __m512 val_b = _mm512_fmadd_ps(vfb, vsb, vbb);

        /* diff = val_a - val_b; sum += diff * diff */
        __m512 diff = _mm512_sub_ps(val_a, val_b);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    /* 水平求和 */
    double sum = _mm512_reduce_add_ps(sum_vec);

    /* 标量处理剩余部分 */
    for (; i < packed_bytes; i++) {
        uint8_t byte_a = pa[i];
        uint8_t byte_b = pb[i];

        int q0a = (byte_a >> 4) & 0x0F;
        int q0b = (byte_b >> 4) & 0x0F;
        float val_a0 = (float)q0a * sa + ba;
        float val_b0 = (float)q0b * sb + bb;
        float diff0 = val_a0 - val_b0;
        sum += (double)diff0 * (double)diff0;

        int idx1 = i * 2 + 1;
        if (idx1 < dim) {
            int q1a = byte_a & 0x0F;
            int q1b = byte_b & 0x0F;
            float val_a1 = (float)q1a * sa + ba;
            float val_b1 = (float)q1b * sb + bb;
            float diff1 = val_a1 - val_b1;
            sum += (double)diff1 * (double)diff1;
        }
    }

    return sum;
}

/*
 * SQ4 距离计算 - 主入口
 */
static inline double
HnswSQ4Distance(Datum a, Datum b)
{
#ifdef __AVX512F__
    return HnswSQ4Distance_AVX512(a, b);
#else
    return HnswSQ4Distance_Scalar(a, b);
#endif
}

/*
 * SQ4 vs HalfVector 距离计算
 * sq4: SQ4量化向量
 * hv: 原始halfvec
 */
static inline double
HnswSQ4HalfvecDistance(Vector *sq4, HalfVector *hv)
{
    Assert(sq4->unused == SQ4_TAG);
    Assert(sq4->dim == hv->dim);

    int dim = sq4->dim;
    int packed_bytes = (dim + 1) / 2;

    char *buf = (char *)sq4->x;

    /* 提取 SQ4 参数 */
    float scale, bias;
    memcpy(&scale, buf, sizeof(float));
    memcpy(&bias, buf + sizeof(float), sizeof(float));

    const uint8_t *packed = (const uint8_t *)(buf + sizeof(float) * 2);
    half *hdata = hv->x;

    /* 计算 L2 距离 */
    double sum = 0.0;
    for (int i = 0; i < packed_bytes; i++) {
        uint8_t byte = packed[i];

        /* 解包高4位 */
        int q0 = (byte >> 4) & 0x0F;
        float val_sq4_0 = (float)q0 * scale + bias;
        float val_hv_0 = HalfToFloat4(hdata[i * 2]);
        float diff0 = val_sq4_0 - val_hv_0;
        sum += (double)diff0 * (double)diff0;

        /* 解包低4位（检查边界） */
        int idx1 = i * 2 + 1;
        if (idx1 < dim) {
            int q1 = byte & 0x0F;
            float val_sq4_1 = (float)q1 * scale + bias;
            float val_hv_1 = HalfToFloat4(hdata[idx1]);
            float diff1 = val_sq4_1 - val_hv_1;
            sum += (double)diff1 * (double)diff1;
        }
    }

    return sum;
}

/*
 * SQ4 vs SQ8/HYBRID 距离计算
 * sq4: SQ4量化向量（索引数据）
 * sq8_or_hybrid: SQ8或HYBRID量化向量（查询数据，可能预量化）
 *
 * 对于HYBRID格式，提取其中的SQ8部分进行计算
 */
static inline double
HnswSQ4vsSQ8Distance(Vector *sq4, Vector *sq8_or_hybrid)
{
    Assert(sq4->unused == SQ4_TAG);
    Assert(sq8_or_hybrid->unused == SQ8_TAG || sq8_or_hybrid->unused == HYBRID_TAG);
    Assert(sq4->dim == sq8_or_hybrid->dim);

    int dim = sq4->dim;
    int packed_bytes = (dim + 1) / 2;

    /* 提取 SQ4 参数 */
    char *buf_sq4 = (char *)sq4->x;
    float scale4, bias4;
    memcpy(&scale4, buf_sq4, sizeof(float));
    memcpy(&bias4, buf_sq4 + sizeof(float), sizeof(float));
    const uint8_t *packed4 = (const uint8_t *)(buf_sq4 + sizeof(float) * 2);

    /* 提取 SQ8 参数 */
    char *buf_sq8;
    float scale8, bias8;
    const uint8_t *codes8;

    if (sq8_or_hybrid->unused == HYBRID_TAG) {
        /* HYBRID 格式：跳过 bq_bytes(2) + bq_data，定位到 SQ8 数据 */
        int bq_bytes = (dim + 7) / 8;
        buf_sq8 = (char *)sq8_or_hybrid->x + sizeof(uint16_t) + bq_bytes;
    } else {
        /* SQ8 格式 */
        buf_sq8 = (char *)sq8_or_hybrid->x;
    }
    memcpy(&scale8, buf_sq8, sizeof(float));
    memcpy(&bias8, buf_sq8 + sizeof(float), sizeof(float));
    codes8 = (const uint8_t *)(buf_sq8 + sizeof(float) * 2);

    /* 计算 L2 距离 */
    double sum = 0.0;
    for (int i = 0; i < packed_bytes; i++) {
        uint8_t byte4 = packed4[i];

        /* 解包 SQ4 高4位 */
        int q4_0 = (byte4 >> 4) & 0x0F;
        float val4_0 = (float)q4_0 * scale4 + bias4;
        /* SQ8 对应值 */
        float val8_0 = (float)codes8[i * 2] * scale8 + bias8;
        float diff0 = val4_0 - val8_0;
        sum += (double)diff0 * (double)diff0;

        /* 解包 SQ4 低4位（检查边界） */
        int idx1 = i * 2 + 1;
        if (idx1 < dim) {
            int q4_1 = byte4 & 0x0F;
            float val4_1 = (float)q4_1 * scale4 + bias4;
            float val8_1 = (float)codes8[idx1] * scale8 + bias8;
            float diff1 = val4_1 - val8_1;
            sum += (double)diff1 * (double)diff1;
        }
    }

    return sum;
}

#endif
