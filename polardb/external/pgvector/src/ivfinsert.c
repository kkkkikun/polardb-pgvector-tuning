#include "postgres.h"

#include <float.h>

#include "access/generic_xlog.h"
#include "ivfflat.h"
#include "halfvec.h"
#include "halfutils.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/*
 * Fast inline L2 squared distance for halfvec (same as ivfscan.c)
 */
static inline float
FastHalfvecL2SquaredDistance(const HalfVector *a, const HalfVector *b)
{
	int			dim = a->dim;
	const half *ax = a->x;
	const half *bx = b->x;
	float		distance = 0.0f;

#ifdef __AVX512F__
	__m512		vsum = _mm512_setzero_ps();
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m256i		ha = _mm256_loadu_si256((const __m256i *) &ax[i]);
		__m256i		hb = _mm256_loadu_si256((const __m256i *) &bx[i]);
		__m512		fa = _mm512_cvtph_ps(ha);
		__m512		fb = _mm512_cvtph_ps(hb);

		__m512		diff = _mm512_sub_ps(fa, fb);

		vsum = _mm512_fmadd_ps(diff, diff, vsum);
	}

	distance = _mm512_reduce_add_ps(vsum);

	for (; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);

		distance += diff * diff;
	}
#elif defined(__AVX2__)
	__m256		vsum = _mm256_setzero_ps();
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m128i		ha = _mm_loadu_si128((const __m128i *) &ax[i]);
		__m128i		hb = _mm_loadu_si128((const __m128i *) &bx[i]);
		__m256		fa = _mm256_cvtph_ps(ha);
		__m256		fb = _mm256_cvtph_ps(hb);

		__m256		diff = _mm256_sub_ps(fa, fb);

		vsum = _mm256_fmadd_ps(diff, diff, vsum);
	}

	__m128		vlow = _mm256_castps256_ps128(vsum);
	__m128		vhigh = _mm256_extractf128_ps(vsum, 1);

	vlow = _mm_add_ps(vlow, vhigh);
	__m128		shuf = _mm_movehdup_ps(vlow);

	vlow = _mm_add_ps(vlow, shuf);
	shuf = _mm_movehl_ps(shuf, vlow);
	vlow = _mm_add_ss(vlow, shuf);
	distance = _mm_cvtss_f32(vlow);

	for (; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);

		distance += diff * diff;
	}
#else
	for (int i = 0; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);

		distance += diff * diff;
	}
#endif

	return distance;
}

/*
 * Find the list that minimizes the distance function
 * Optimized with fast path for halfvec
 */
static void
FindInsertPage(Relation index, Datum *values, BlockNumber *insertPage, ListInfo * listInfo)
{
	double		minDistance = DBL_MAX;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	FmgrInfo   *procinfo;
	Oid			collation;
	HalfVector *queryHv = NULL;

	/* Try to use fast path for halfvec (safe to cast, will check on use) */
	queryHv = (HalfVector *) DatumGetPointer(values[0]);

	/* Avoid compiler warning */
	listInfo->blkno = nextblkno;
	listInfo->offno = FirstOffsetNumber;

	procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	collation = index->rd_indcollation[0];

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(index, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);
		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list;
			double		distance;

			list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));

			/* Use fast path for halfvec, bypass function calls */
			if (queryHv != NULL)
			{
				HalfVector *centerHv = (HalfVector *) &list->center;

				/* Fast inline distance for both halfvec and vector types */
				distance = (double) FastHalfvecL2SquaredDistance(queryHv, centerHv);
			}
			else
			{
				distance = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, values[0], PointerGetDatum(&list->center)));
			}

			if (distance < minDistance || !BlockNumberIsValid(*insertPage))
			{
				*insertPage = list->insertPage;
				listInfo->blkno = nextblkno;
				listInfo->offno = offno;
				minDistance = distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}
}

/*
 * Insert a tuple into the index
 */
static void
InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid)
{
	const		IvfflatTypeInfo *typeInfo = IvfflatGetTypeInfo(index);
	IndexTuple	itup;
	Datum		value;
	FmgrInfo   *normprocinfo;
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	Size		itemsz;
	BlockNumber insertPage = InvalidBlockNumber;
	ListInfo	listInfo;
	BlockNumber originalInsertPage;

	/* Detoast once for all calls */
	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Normalize if needed */
	normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	if (normprocinfo != NULL)
	{
		Oid			collation = index->rd_indcollation[0];

		if (!IvfflatCheckNorm(normprocinfo, collation, value))
			return;

		value = IvfflatNormValue(typeInfo, collation, value);
	}

	/* Ensure index is valid */
	IvfflatGetMetaPageInfo(index, NULL, NULL);

	/* Find the insert page - sets the page and list info */
	FindInsertPage(index, &value, &insertPage, &listInfo);
	Assert(BlockNumberIsValid(insertPage));
	originalInsertPage = insertPage;

	/* Form tuple */
	itup = index_form_tuple(RelationGetDescr(index), &value, isnull);
	itup->t_tid = *heap_tid;

	/* Get tuple size */
	itemsz = MAXALIGN(IndexTupleSize(itup));
	Assert(itemsz <= BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(IvfflatPageOpaqueData)) - sizeof(ItemIdData));

	/* Find a page to insert the item */
	for (;;)
	{
		buf = ReadBuffer(index, insertPage);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);

		if (PageGetFreeSpace(page) >= itemsz)
			break;

		insertPage = IvfflatPageGetOpaque(page)->nextblkno;

		if (BlockNumberIsValid(insertPage))
		{
			/* Move to next page */
			GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
		}
		else
		{
			Buffer		newbuf;
			Page		newpage;

			/* Add a new page */
			LockRelationForExtension(index, ExclusiveLock);
			newbuf = IvfflatNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);

			/* Init new page */
			newpage = GenericXLogRegisterBuffer(state, newbuf, GENERIC_XLOG_FULL_IMAGE);
			IvfflatInitPage(newbuf, newpage);

			/* Update insert page */
			insertPage = BufferGetBlockNumber(newbuf);

			/* Update previous buffer */
			IvfflatPageGetOpaque(page)->nextblkno = insertPage;

			/* Commit */
			GenericXLogFinish(state);

			/* Unlock previous buffer */
			UnlockReleaseBuffer(buf);

			/* Prepare new buffer */
			state = GenericXLogStart(index);
			buf = newbuf;
			page = GenericXLogRegisterBuffer(state, buf, 0);
			break;
		}
	}

	/* Add to next offset */
	if (PageAddItem(page, (Item) itup, itemsz, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

	IvfflatCommitBuffer(buf, state);

	/* Update the insert page */
	if (insertPage != originalInsertPage)
		IvfflatUpdateList(index, listInfo, insertPage, originalInsertPage, InvalidBlockNumber, MAIN_FORKNUM);
}

/*
 * Insert a tuple into the index
 */
bool
ivfflatinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			  Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			  ,bool indexUnchanged
#endif
			  ,IndexInfo *indexInfo
)
{
	MemoryContext oldCtx;
	MemoryContext insertCtx;

	/* Skip nulls */
	if (isnull[0])
		return false;

	/*
	 * Use memory context since detoast, IvfflatNormValue, and
	 * index_form_tuple can allocate
	 */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "Ivfflat insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/* Insert tuple */
	InsertTuple(index, values, isnull, heap_tid);

	/* Delete memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
