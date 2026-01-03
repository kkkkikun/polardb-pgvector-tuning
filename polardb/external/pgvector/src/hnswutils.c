#include "postgres.h"

#include <math.h>
#include <immintrin.h>		/* AVX-512 支持 */

#include "access/generic_xlog.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "hnsw.h"
#include "halfutils.h"
#include "lib/pairingheap.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/memdebug.h"
#include "utils/rel.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

extern Datum vector_l2_squared_distance(PG_FUNCTION_ARGS);
extern Datum halfvec_l2_squared_distance(PG_FUNCTION_ARGS);

/* ================== 比赛专用 Hack 结构 ================== */

/* 计算存储所需大小的宏 - 用于检测量化格式 */
#define HNSW_QV_SIZE(dim) (sizeof(float)*2 + (dim) * sizeof(uint8_t))

/* ======================================================= */

/* 函数声明 */
static inline double HnswGetDistance(Datum a, Datum b, HnswSupport * support);

/* ======================================================= */

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

static bool
HnswSupportsQuantization(FmgrInfo *procinfo)
{
	PGFunction	fn;

	if (procinfo == NULL)
		return false;

	fn = procinfo->fn_addr;

	return fn == (PGFunction) vector_l2_squared_distance ||
		fn == (PGFunction) halfvec_l2_squared_distance;
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	const HnswTypeInfo *typeInfo;

	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
	support->allowQuantization = HnswSupportsQuantization(support->procinfo);

	/* 从 TypeInfo 获取量化类型 */
	typeInfo = HnswGetTypeInfo(index);
	support->quantType = typeInfo->quantType;
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;
	/* 初始化邻居版本号用于无锁读取 */
	pg_atomic_init_u32(&element->neighborVersion, 0);

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (Pointer) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (Pointer) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	/* 如果不支持量化（非L2距离）或量化类型为NONE，直接使用原值 */
	if (!support->allowQuantization || support->quantType == HNSW_QUANT_NONE)
	{
		*out = value;
		return true;
	}

	/* === 根据操作符类选择量化方法 === */
	/* 【关键修复】先检测格式，再解包，避免halfvec误入vector分支 */
	HnswValueFormat fmt = HnswDetectFormat(value);

	/* 已经是量化格式，直接返回 */
	if (fmt == HNSW_FMT_SQ8 || fmt == HNSW_FMT_HYBRID || fmt == HNSW_FMT_SQ4) {
		elog(DEBUG1, "Heap value is already quantized, using directly");
		*out = value;
		return true;
	}

	/* 根据 quantType 选择量化方法 */
	if (support->quantType == HNSW_QUANT_HYBRID) {
		/* 混合量化 (BQ + SQ8) */
		if (fmt == HNSW_FMT_RAW_HALFVEC) {
			HalfVector *halfvec = DatumGetHalfVector(value);
			int			dim = halfvec->dim;
			int			bq_bytes = (dim + 7) / 8;

			Size payload_size = sizeof(uint16_t) + bq_bytes + sizeof(float)*2 + dim;
			Vector	   *quantized_vec = (Vector *) palloc0(offsetof(Vector, x) + payload_size);

			HnswQuantizeHybrid(halfvec, quantized_vec);
			*out = PointerGetDatum(quantized_vec);

		} else if (fmt == HNSW_FMT_RAW_FLOAT_VEC) {
			Vector	   *vec = DatumGetVector(value);
			int			dim = vec->dim;
			int			bq_bytes = (dim + 7) / 8;

			Size		payload_size = sizeof(uint16_t) + bq_bytes + sizeof(float)*2 + dim;
			Vector	   *quantized_vec = (Vector *) palloc0(offsetof(Vector, x) + payload_size);

			HnswQuantizeHybridFromFloat(vec, quantized_vec);
			*out = PointerGetDatum(quantized_vec);

		} else {
			/* 未知格式：不量化 */
			*out = value;
		}

	} else if (support->quantType == HNSW_QUANT_SQ4) {
		/* SQ4 量化 (4-bit) */
		if (fmt == HNSW_FMT_RAW_HALFVEC) {
			HalfVector *halfvec = DatumGetHalfVector(value);
			int			dim = halfvec->dim;
			int			packed_bytes = (dim + 1) / 2;

			Size payload_size = sizeof(float)*2 + packed_bytes;
			Vector	   *quantized_vec = (Vector *) palloc0(offsetof(Vector, x) + payload_size);

			HnswQuantizeToSQ4(halfvec, quantized_vec);
			*out = PointerGetDatum(quantized_vec);

		} else if (fmt == HNSW_FMT_RAW_FLOAT_VEC) {
			Vector	   *vec = DatumGetVector(value);
			int			dim = vec->dim;
			int			packed_bytes = (dim + 1) / 2;

			Size		payload_size = sizeof(float)*2 + packed_bytes;
			Vector	   *quantized_vec = (Vector *) palloc0(offsetof(Vector, x) + payload_size);

			HnswQuantizeToSQ4FromFloat(vec, quantized_vec);
			*out = PointerGetDatum(quantized_vec);

		} else {
			/* 未知格式：不量化 */
			*out = value;
		}

	} else {
		/* HNSW_QUANT_BQ 或其他：暂不实现，不量化 */
		*out = value;
	}

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);
	Vector	   *vec = (Vector *) valuePtr;

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}

	/* === 比赛专用 Hack：量化存储 === */
	/* 数据已经在 HnswFormIndexValue 中量化过了，直接复制 */
	memcpy(&etup->data, vec, VARSIZE_ANY(vec));
}

/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 *
 * 【性能优化】优化分支预测，使用likely()减少分支预测失败
 * 零分配的距离计算，显著提升性能
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	/* 【优化】使用更快速的格式检测 - 直接检查unused字段 */
	Vector *va = (Vector *) PG_DETOAST_DATUM_PACKED(a);
	Vector *vb = (Vector *) PG_DETOAST_DATUM_PACKED(b);

	uint16 tag_a = va->unused;
	uint16 tag_b = vb->unused;

	/* 【最常见】混合格式 vs 混合格式（索引内部比较） */
	if (likely(tag_a == HYBRID_TAG && tag_b == HYBRID_TAG)) {
		/* 使用 SQ8 精确距离（用于候选集排序） */
		return HnswHybridSQ8Distance(a, b);
	}

	/* 【SQ4】SQ4 vs SQ4（4-bit量化） */
	if (tag_a == SQ4_TAG && tag_b == SQ4_TAG) {
		return HnswSQ4Distance(a, b);
	}

	/* 【次常见】SQ8 vs SQ8（兼容旧索引） */
	if (tag_a == SQ8_TAG && tag_b == SQ8_TAG) {
		return HnswSQ8Distance(a, b);
	}

	/* 【跨格式】SQ4 vs SQ8/HYBRID（索引是SQ4，查询可能预量化为SQ8） */
	if (tag_a == SQ4_TAG && (tag_b == SQ8_TAG || tag_b == HYBRID_TAG)) {
		return HnswSQ4vsSQ8Distance(va, vb);
	}
	if (tag_b == SQ4_TAG && (tag_a == SQ8_TAG || tag_a == HYBRID_TAG)) {
		return HnswSQ4vsSQ8Distance(vb, va);
	}

	/* 【较少】原始 vs 原始（纯原始数据比较） */
	if (tag_a != SQ8_TAG && tag_a != HYBRID_TAG && tag_a != SQ4_TAG &&
	    tag_b != SQ8_TAG && tag_b != HYBRID_TAG && tag_b != SQ4_TAG) {
		return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
	}

	/* 【较少情况】量化 vs 原始 */
	if (tag_a == SQ8_TAG || tag_a == HYBRID_TAG || tag_a == SQ4_TAG) {
		/* a=量化, b=halfvec */
		HalfVector *half_vec = (HalfVector *) PG_DETOAST_DATUM_PACKED(b);
		if (tag_a == SQ8_TAG) {
			return (double)HnswSQ8HalfvecDistance2(va, half_vec);
		} else if (tag_a == HYBRID_TAG) {
			/* HYBRID: 使用专用函数，正确跳过 BQ 部分 */
			return (double)HnswHybridHalfvecDistance(va, half_vec);
		} else {
			/* SQ4 vs halfvec */
			return (double)HnswSQ4HalfvecDistance(va, half_vec);
		}
	} else {
		/* b=量化, a=halfvec */
		HalfVector *half_vec = (HalfVector *) PG_DETOAST_DATUM_PACKED(a);
		if (tag_b == SQ8_TAG) {
			return (double)HnswSQ8HalfvecDistance2(vb, half_vec);
		} else if (tag_b == HYBRID_TAG) {
			/* HYBRID: 使用专用函数，正确跳过 BQ 部分 */
			return (double)HnswHybridHalfvecDistance(vb, half_vec);
		} else {
			/* SQ4 vs halfvec */
			return (double)HnswSQ4HalfvecDistance(vb, half_vec);
		}
	}
}

/*
 * 获取距离（支持粗筛模式）
 * coarse=true: 使用 BQ Hamming 距离（快速粗筛）
 * coarse=false: 使用 SQ8 L2 距离（精确排序）
 */
static inline double
HnswGetDistanceCoarse(Datum a, Datum b, HnswSupport * support, bool coarse)
{
	if (!coarse) {
		return HnswGetDistance(a, b, support);
	}

	/* 粗筛模式：检查是否为混合格式 */
	Vector *va = (Vector *) PG_DETOAST_DATUM_PACKED(a);
	Vector *vb = (Vector *) PG_DETOAST_DATUM_PACKED(b);

	if (va->unused == HYBRID_TAG && vb->unused == HYBRID_TAG) {
		/* 使用 BQ Hamming 距离快速粗筛 */
		return HnswHybridBQDistance(a, b);
	}

	/* 非混合格式：回退到精确计算 */
	return HnswGetDistance(a, b, support);
}

/*
 * Load an element and optionally get its distance from q
 */
static void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element)
{
    Buffer      buf;
    Page        page;
    HnswElementTuple etup;

    /* Read vector */
    buf = ReadBuffer(index, blkno);
    LockBuffer(buf, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buf);

    etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

    Assert(HnswIsElementTuple(etup));

    /* Calculate distance */
    if (distance != NULL)
    {
        if (DatumGetPointer(q->value) == NULL)
            *distance = 0;
        else
        {
            /* 从 HnswElementTuple 的 data 字段构造正确的 Datum */
            /* 注意：etup->data 是一个 Vector 结构体，不是指针 */
            Datum       index_datum = PointerGetDatum(&etup->data);

            /* 使用统一的距离计算函数，它会自动判断 SQ8 vs 原始数据 */
            double      result = HnswGetDistance(q->value, index_datum, support);

            *distance = result;
        }
    }

    /* Load element */
    if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
    {
        if (*element == NULL)
            *element = HnswInitElementFromBlock(blkno, offno);

        HnswLoadElementFromTuple(*element, etup, true, loadVec);
    }

    UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance < HnswGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(c_node, a)->distance > HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 *
 * 【优化】使用无锁读取 + 版本号验证（乐观锁）
 * 参考 VectorChord 的减少锁争用策略
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);
	uint32		version_before;
	uint32		version_after;
	int			retry_count = 0;
	const int	max_retries = 3;

retry:
	/* 读取版本号（读前） */
	version_before = pg_atomic_read_u32(&element->neighborVersion);

	/* 内存屏障：确保版本号读取在数据读取之前完成 */
	pg_memory_barrier();

	/* 无锁复制邻居数据 */
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);

	/* 内存屏障：确保数据读取在版本号读取之前完成 */
	pg_memory_barrier();

	/* 读取版本号（读后） */
	version_after = pg_atomic_read_u32(&element->neighborVersion);

	/* 版本号不匹配说明有并发修改，需要重试或降级到加锁 */
	if (version_before != version_after)
	{
		retry_count++;
		if (retry_count < max_retries)
			goto retry;

		/* 超过重试次数，降级到传统加锁方式 */
		LWLockAcquire(&element->lock, LW_SHARED);
		memcpy(localNeighborhood, neighborhood, neighborhoodSize);
		LWLockRelease(&element->lock);
	}

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	int			start;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Load unvisited neighbors from disk
 */
static void
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];

	*unvisitedLength = 0;

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].indextid = *indextid;
	}
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	int			unvisitedLength;
	bool		inMemory = index == NULL;

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		if (c->distance > f->distance)
			break;

		cElement = HnswPtrAccess(base, c->element);

		if (inMemory)
			HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
			HnswLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, m, lm, lc);

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		/* 【优化】磁盘查询: 预取初始批次的元素页面 */
		if (!inMemory && unvisitedLength > 0)
		{
			int prefetch_count = Min(unvisitedLength, 4);  /* 预取前4个页面 */
			for (int p = 0; p < prefetch_count; p++)
			{
				ItemPointer tid = &unvisited[p].indextid;
				PrefetchBuffer(index, MAIN_FORKNUM, ItemPointerGetBlockNumber(tid));
			}
		}

		/* 【优化】缓存最远距离，避免每次迭代都访问堆 */
		double cached_f_distance = HnswGetSearchCandidate(w_node, pairingheap_first(W))->distance;

		for (int i = 0; i < unvisitedLength; i++)
		{
			HnswElement eElement;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;

			/* ==========================================================
             * 【针对 200维 & M=10 优化的深度预取】
             * ========================================================== */
            const int prefetch_step = 2; // M=10 时邻居少，步长不宜过大

            if (i + prefetch_step < unvisitedLength)
            {
                if (inMemory)
                {
                    HnswElement next_el = unvisited[i + prefetch_step].element;
                    if (next_el)
                    {
                        // 1. 预取节点元数据
                        __builtin_prefetch(next_el, 0, 3);

                        void *next_vec = HnswPtrAccess(base, next_el->value);
                        if (next_vec)
                        {
                            /* * 2. 深度预取向量数据：
                             * 200维占 800字节，即 12.5 个 Cache Line。
                             * 我们不需要全部预取（以免浪费带宽），预取前 3-4 个 Line 通常收益最高。
                             */
                            char *ptr = (char *)next_vec;
                            __builtin_prefetch(ptr, 0, 3);       // 第 1-16 维
                            __builtin_prefetch(ptr + 64, 0, 3);  // 第 17-32 维
                            __builtin_prefetch(ptr + 128, 0, 3); // 第 33-48 维
                            __builtin_prefetch(ptr + 192, 0, 3); // 第 49-64 维
                        }
                    }
                }
                else
                {
                    /* 【优化】磁盘查询时预取元素页面 */
                    ItemPointer next_tid = &unvisited[i + prefetch_step].indextid;
                    BlockNumber next_blkno = ItemPointerGetBlockNumber(next_tid);
                    PrefetchBuffer(index, MAIN_FORKNUM, next_blkno);
                }
            }
            /* ========================================================== */

			if (inMemory)
			{
				eElement = unvisited[i].element;

				/* 【BQ+SQ8 混合量化优化】两阶段距离计算 */
				Datum elemValue = HnswGetValue(base, eElement);
				Vector *elemVec = (Vector *) PG_DETOAST_DATUM_PACKED(elemValue);

				/*
				 * 【BQ 粗筛 - 固定阈值策略】
				 * 使用 Hamming 距离的绝对阈值，而不是与 L2 距离比较
				 * 阈值 = dim * BQ_THRESHOLD_RATIO（可调参数）
				 */
#define BQ_THRESHOLD_RATIO 0.42  /* 经验值，可根据测试结果调整 */

				if (!alwaysAdd && elemVec->unused == HYBRID_TAG) {
					Vector *queryVec = (Vector *) PG_DETOAST_DATUM_PACKED(q->value);
					if (queryVec->unused == HYBRID_TAG) {
						int dim = elemVec->dim;
						double bq_threshold = dim * BQ_THRESHOLD_RATIO;

						/* BQ 粗筛 */
						double bq_dist = HnswHybridBQDistance(q->value, elemValue);

						if (bq_dist > bq_threshold) {
							/* 快速拒绝：Hamming 距离过大 */
							if (discarded != NULL) {
								/* 需要精确距离用于 discarded 堆 */
								eDistance = HnswHybridSQ8Distance(q->value, elemValue);
								e = HnswInitSearchCandidate(base, eElement, eDistance);
								pairingheap_add(*discarded, &e->w_node);
							}
							continue;
						}
					}
				}

				/* SQ8 精确距离 */
				eDistance = HnswGetDistance(q->value, elemValue, support);
			}
			else
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

				/* Avoid any allocations if not adding */
				eElement = NULL;
				/* 【优化】使用缓存的距离阈值 */
				HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, alwaysAdd || discarded != NULL ? NULL : &cached_f_distance, &eElement);

				if (eElement == NULL)
					continue;
			}

			if (eElement == NULL || !(eDistance < cached_f_distance || alwaysAdd))
			{
				if (discarded != NULL)
				{
					/* Create a new candidate */
					e = HnswInitSearchCandidate(base, eElement, eDistance);
					pairingheap_add(*discarded, &e->w_node);
				}

				continue;
			}

			/* Make robust to issues */
			if (eElement->level < lc)
				continue;

			/* Create a new candidate */
			e = HnswInitSearchCandidate(base, eElement, eDistance);
			pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);

			/*
			 * Do not count elements being deleted towards ef when vacuuming.
			 * It would be ideal to do this for inserts as well, but this
			 * could affect insert performance.
			 */
			if (CountElement(skipElement, eElement))
			{
				wlen++;

				/* No need to decrement wlen */
				if (wlen > ef)
				{
					HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

					if (discarded != NULL)
						pairingheap_add(*discarded, &d->w_node);

					/* 【优化】堆变化后更新缓存 */
					cached_f_distance = HnswGetSearchCandidate(w_node, pairingheap_first(W))->distance;
				}
			}
			else
			{
				/* 【优化】添加新候选后，如果比当前最远更远则更新缓存 */
				if (eDistance > cached_f_distance)
					cached_f_distance = eDistance;
			}
		}
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/*
 * 【优化】数组版本的CheckElementCloser - 避免List遍历开销
 */
static bool
CheckElementCloserArray(char *base, HnswCandidate *e, HnswCandidate **r, int rlen, HnswSupport *support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);

	for (int i = 0; i < rlen; i++)
	{
		HnswCandidate *ri = r[i];
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = (float)HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * 【优化】qsort比较函数 - 降序排列（距离大的在前）
 */
static int
CompareCandidateDistancesQsort(const void *a, const void *b)
{
	HnswCandidate *hca = *(HnswCandidate **)a;
	HnswCandidate *hcb = *(HnswCandidate **)b;

	if (hca->distance < hcb->distance)
		return 1;
	if (hca->distance > hcb->distance)
		return -1;

	/* Pointer tie-breaker for deterministic order */
	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;
	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

static int
CompareCandidateDistancesOffsetQsort(const void *a, const void *b)
{
	HnswCandidate *hca = *(HnswCandidate **)a;
	HnswCandidate *hcb = *(HnswCandidate **)b;

	if (hca->distance < hcb->distance)
		return 1;
	if (hca->distance > hcb->distance)
		return -1;

	/* Offset tie-breaker for deterministic order */
	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;
	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Algorithm 4 from paper
 * 【优化】使用数组代替List，避免list_copy/list_sort/list_delete_last/lappend开销
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	int			clen = list_length(c);
	ListCell   *lc;
	int			i;

	/* 快速路径：候选数不超过lm，直接复制返回 */
	if (clen <= lm)
		return list_copy(c);

	/* 【优化】使用栈分配小数组，避免palloc开销 */
	/* 最大支持256个候选，超出则使用堆分配 */
	HnswCandidate *w_stack[256];
	HnswCandidate *r_stack[256];
	HnswCandidate *wd_stack[256];
	HnswCandidate *added_stack[256];

	HnswCandidate **w = (clen <= 256) ? w_stack : palloc(sizeof(HnswCandidate *) * clen);
	HnswCandidate **r_arr = (lm <= 256) ? r_stack : palloc(sizeof(HnswCandidate *) * lm);
	HnswCandidate **wd = (clen <= 256) ? wd_stack : palloc(sizeof(HnswCandidate *) * clen);
	HnswCandidate **added_arr = (lm <= 256) ? added_stack : palloc(sizeof(HnswCandidate *) * lm);

	int			wlen = 0;
	int			rlen = 0;
	int			wdlen = 0;
	int			wdoff = 0;
	int			addedlen = 0;
	bool		mustCalculate = !(*closerSet);
	bool		removedAny = false;

	/* 复制候选到数组 */
	foreach(lc, c)
	{
		w[wlen++] = (HnswCandidate *) lfirst(lc);
	}

	/* 【优化】使用qsort代替list_sort - 降序排列（距离大的在前） */
	if (sortCandidates)
	{
		if (base == NULL)
			qsort(w, wlen, sizeof(HnswCandidate *), CompareCandidateDistancesQsort);
		else
			qsort(w, wlen, sizeof(HnswCandidate *), CompareCandidateDistancesOffsetQsort);
	}

	/* 从后往前处理（距离最小的在后面） */
	while (wlen > 0 && rlen < lm)
	{
		/* 取最后一个元素（距离最小） */
		HnswCandidate *e = w[--wlen];

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloserArray(base, e, r_arr, rlen, support);
		else if (addedlen > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloserArray(base, e, added_arr, addedlen, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloserArray(base, e, r_arr, rlen, support);
					if (e->closer)
						added_arr[addedlen++] = e;
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloserArray(base, e, r_arr, rlen, support);
			if (e->closer)
				added_arr[addedlen++] = e;
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r_arr[rlen++] = e;
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && rlen < lm)
		r_arr[rlen++] = wd[wdoff++];

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else if (wlen > 0)
			*pruned = w[0];  /* 相当于原来的 linitial(w) */
		else
			*pruned = NULL;
	}

	/* 【最后】将结果数组转换回List返回 */
	List *result = NIL;
	for (i = 0; i < rlen; i++)
		result = lappend(result, r_arr[i]);

	/* 释放堆分配的内存（栈分配的不需要释放） */
	if (clen > 256)
	{
		pfree(w);
		pfree(wd);
	}
	if (lm > 256)
	{
		pfree(r_arr);
		pfree(added_arr);
	}

	return result;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = HnswGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	/*
	 * 【优化】跨层共享 visited set
	 * 同一个元素在不同层被访问时，距离是相同的，无需重复计算
	 * 首层初始化 visited，后续层复用
	 */
	{
		visited_hash sharedVisited;
		bool		firstLayer = true;

		for (int lc = level; lc >= 0; lc--)
		{
			int			lm = HnswGetLayerM(m, lc);
			List	   *neighbors;
			List	   *lw = NIL;
			ListCell   *lc2;

			/*
			 * 首层: 初始化 visited set (initVisited=true)
			 * 后续层: 复用 visited set (initVisited=false)
			 */
			w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement,
								inMemory ? &sharedVisited : NULL, NULL, firstLayer, NULL);
			firstLayer = false;

			/* Convert search candidates to candidates */
			foreach(lc2, w)
			{
				HnswSearchCandidate *sc = lfirst(lc2);
				HnswCandidate *hc = palloc(sizeof(HnswCandidate));

				hc->element = sc->element;
				hc->distance = sc->distance;

				lw = lappend(lw, hc);
			}

			/* Elements being deleted or skipped can help with search */
			/* but should be removed before selecting neighbors */
			if (!inMemory)
				lw = RemoveElements(base, lw, skipElement);

			/*
			 * Candidates are sorted, but not deterministically. Could set
			 * sortCandidates to true for in-memory builds to enable closer
			 * caching, but there does not seem to be a difference in performance.
			 */
			neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

			AddConnections(base, element, neighbors, lc);

			ep = w;
		}
	}
}

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		/* 默认 vector 类型：使用混合量化 */
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL,
			.quantType = HNSW_QUANT_HYBRID
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	/* halfvec_l2_ops 默认使用混合量化 (BQ+SQ8) */
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL,
		.quantType = HNSW_QUANT_HYBRID
	};

	PG_RETURN_POINTER(&typeInfo);
}

/* SQ4 量化支持函数 - 用于 halfvec_l2_sq4_ops */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_sq4_support);
Datum
hnsw_halfvec_sq4_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL,
		.quantType = HNSW_QUANT_SQ4
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	/* bit 类型不支持量化 */
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL,
		.quantType = HNSW_QUANT_NONE
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	/* sparsevec 类型不支持量化 */
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue,
		.quantType = HNSW_QUANT_NONE
	};

	PG_RETURN_POINTER(&typeInfo);
}
