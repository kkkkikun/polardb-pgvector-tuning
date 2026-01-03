#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "halfvec.h"
#include "vector.h"

/* SIMD headers */
#ifdef __x86_64__
#include <immintrin.h>
#endif

/* Simple half to float conversion for query conversion */
static inline float
HalfToFloatSimple(half h)
{
#ifdef F16C_SUPPORT
	return _cvtsh_ss(h);
#else
	uint32		sign = (h >> 15) & 0x1;
	uint32		exp = (h >> 10) & 0x1F;
	uint32		mant = h & 0x3FF;
	uint32		f;

	if (exp == 0)
	{
		if (mant == 0)
			f = sign << 31;
		else
		{
			exp = 1;
			while ((mant & 0x400) == 0)
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x3FF;
			f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
		}
	}
	else if (exp == 31)
		f = (sign << 31) | 0x7F800000 | (mant << 13);
	else
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);

	return *((float *) &f);
#endif
}

/*
 * Convert query datum to float array for RQ distance computation.
 * Uses VARSIZE to determine if it's halfvec or vector.
 */
static float *
QueryDatumToFloat(Datum value, int dimensions)
{
	Pointer		ptr = DatumGetPointer(value);
	float	   *result;
	int			i;

	if (ptr == NULL)
		return NULL;

	result = palloc(dimensions * sizeof(float));

	/* Check if it's a halfvec or vector based on size */
	if (VARSIZE_ANY(ptr) <= HALFVEC_SIZE(dimensions) + 8)
	{
		/* Likely halfvec */
		HalfVector *hv = (HalfVector *) ptr;

		for (i = 0; i < hv->dim && i < dimensions; i++)
			result[i] = HalfToFloatSimple(hv->x[i]);
	}
	else
	{
		/* Likely full vector */
		Vector	   *v = (Vector *) ptr;

		for (i = 0; i < v->dim && i < dimensions; i++)
			result[i] = v->x[i];
	}

	return result;
}

/*
 * Initialize RQ support for query
 *
 * Note: RQ distance table is created but NOT yet used in search path.
 * This is infrastructure for future optimization.
 */

/*
 * Initialize RaBitQ query state for early filtering
 */
static void
InitRaBitQForQuery(HnswScanOpaque so, Relation index, Datum value)
{
	Vector	   *vec;
	MemoryContext oldCtx;

	/* Already initialized */
	if (so->rabitqQueryState != NULL)
		return;

	/* Load encoder if not loaded */
	if (so->rabitqEncoder == NULL)
	{
		so->rabitqEncoder = HnswLoadRaBitQEncoder(index);
		if (so->rabitqEncoder == NULL)
		{
			so->useRaBitQ = false;
			return;
		}
		so->useRaBitQ = true;
	}

	/* Prepare SQ4 query state */
	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	vec = DatumGetVector(value);
	so->rabitqQueryState = palloc(sizeof(RaBitQQueryStateSQ4));
	RaBitQPrepareSQ4Query(so->rabitqEncoder, vec->x, so->rabitqQueryState);

	MemoryContextSwitchTo(oldCtx);

	elog(DEBUG1, "RaBitQ: prepared SQ4 query state");
}

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	/* Initialize RaBitQ query state for this query */
	InitRaBitQForQuery(so, index, value);

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		/* Upper layers: no RQ filtering, just find entry to ground layer */
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL, NULL, NULL);
		ep = w;
	}

	/* Ground layer: use RaBitQ for early filtering if available */
	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples, NULL, so->rabitqQueryState);
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	/* Use RaBitQ for early filtering if available */
	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples, NULL, so->rabitqQueryState);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/* Initialize RaBitQ support (will be loaded on first query) */
	so->rabitqEncoder = NULL;
	so->rabitqQueryState = NULL;
	so->useRaBitQ = false;

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();

	/*
	 * RaBitQ structures are allocated in tmpCtx, so they will be freed
	 * by MemoryContextReset. Just set pointers to NULL.
	 * The encoder will be reloaded from index on next query.
	 */
	so->rabitqQueryState = NULL;
	/* rabitqEncoder is allocated in persistent context, not reset here */

	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	/*
	 * RQ/RaBitQ structures are allocated in tmpCtx, so they will be freed
	 * by MemoryContextDelete. Just delete the context.
	 * RaBitQ encoder is allocated in query context, free it explicitly.
	 */
	if (so->rabitqEncoder != NULL)
	{
		RaBitQEncoderFree(so->rabitqEncoder);
		pfree(so->rabitqEncoder);
	}

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
