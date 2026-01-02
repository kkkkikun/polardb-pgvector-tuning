/*
 * Residual Quantization (RQ) Implementation
 *
 * This implements a multi-stage residual quantization for vector compression.
 * Each stage quantizes the residual from the previous stage.
 */

#include "postgres.h"
#include "rq.h"
#include "utils/memutils.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

/* Half-precision conversion utilities */
#ifdef F16C_SUPPORT
#include <immintrin.h>

static inline float
HalfToFloat(half h)
{
	return _cvtsh_ss(h);
}

static inline half
FloatToHalf(float f)
{
	return _cvtss_sh(f, 0);
}
#else
/* Software half-to-float conversion */
static inline float
HalfToFloat(half h)
{
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
			/* Denormalized */
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
}

static inline half
FloatToHalf(float f)
{
	uint32		x = *((uint32 *) &f);
	uint32		sign = (x >> 16) & 0x8000;
	int32		exp = ((x >> 23) & 0xFF) - 127 + 15;
	uint32		mant = (x >> 13) & 0x3FF;

	if (exp <= 0)
		return sign;
	if (exp >= 31)
		return sign | 0x7C00;

	return sign | (exp << 10) | mant;
}
#endif

/*
 * Convert HalfVector to float array
 */
void
HalfvecToFloat(HalfVector *hv, float *output)
{
	int			dim = hv->dim;

#ifdef __AVX512F__
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m256i		h = _mm256_loadu_si256((__m256i *) & hv->x[i]);
		__m512		f = _mm512_cvtph_ps(h);

		_mm512_storeu_ps(&output[i], f);
	}
	for (; i < dim; i++)
		output[i] = HalfToFloat(hv->x[i]);
#elif defined(__AVX2__) && defined(F16C_SUPPORT)
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m128i		h = _mm_loadu_si128((__m128i *) & hv->x[i]);
		__m256		f = _mm256_cvtph_ps(h);

		_mm256_storeu_ps(&output[i], f);
	}
	for (; i < dim; i++)
		output[i] = HalfToFloat(hv->x[i]);
#else
	for (int i = 0; i < dim; i++)
		output[i] = HalfToFloat(hv->x[i]);
#endif
}

/*
 * Convert float array to HalfVector
 */
void
FloatToHalfvec(float *input, int dim, HalfVector *output)
{
	SET_VARSIZE(output, HALFVEC_SIZE(dim));
	output->dim = dim;
	output->unused = 0;

#ifdef __AVX512F__
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		f = _mm512_loadu_ps(&input[i]);
		__m256i		h = _mm512_cvtps_ph(f, 0);

		_mm256_storeu_si256((__m256i *) & output->x[i], h);
	}
	for (; i < dim; i++)
		output->x[i] = FloatToHalf(input[i]);
#elif defined(__AVX2__) && defined(F16C_SUPPORT)
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		f = _mm256_loadu_ps(&input[i]);
		__m128i		h = _mm256_cvtps_ph(f, 0);

		_mm_storeu_si128((__m128i *) & output->x[i], h);
	}
	for (; i < dim; i++)
		output->x[i] = FloatToHalf(input[i]);
#else
	for (int i = 0; i < dim; i++)
		output->x[i] = FloatToHalf(input[i]);
#endif
}

/*
 * Create a new RQ codebook
 */
RQCodebook *
RQCreateCodebook(int dim, int numStages, int numCentroids)
{
	RQCodebook *cb;
	Size		centroidSize;

	cb = palloc0(sizeof(RQCodebook));
	cb->dim = dim;
	cb->numStages = numStages;
	cb->numCentroids = numCentroids;

	centroidSize = (Size) numStages * numCentroids * dim * sizeof(float);
	cb->centroids = palloc0(centroidSize);

	return cb;
}

/*
 * Free an RQ codebook
 */
void
RQFreeCodebook(RQCodebook *cb)
{
	if (cb)
	{
		if (cb->centroids)
			pfree(cb->centroids);
		pfree(cb);
	}
}

/*
 * Get pointer to a specific centroid
 */
float *
RQGetCentroid(RQCodebook *cb, int stage, int centroidIdx)
{
	Size		offset = ((Size) stage * cb->numCentroids + centroidIdx) * cb->dim;

	return &cb->centroids[offset];
}

/*
 * Compute L2 squared distance between two vectors
 */
static float
ComputeL2Squared(float *a, float *b, int dim)
{
	float		sum = 0.0f;

#ifdef __AVX512F__
	__m512		vsum = _mm512_setzero_ps();
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a[i]);
		__m512		vb = _mm512_loadu_ps(&b[i]);
		__m512		diff = _mm512_sub_ps(va, vb);

		vsum = _mm512_fmadd_ps(diff, diff, vsum);
	}
	sum = _mm512_reduce_add_ps(vsum);
	for (; i < dim; i++)
	{
		float		diff = a[i] - b[i];

		sum += diff * diff;
	}
#elif defined(__AVX2__)
	__m256		vsum = _mm256_setzero_ps();
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a[i]);
		__m256		vb = _mm256_loadu_ps(&b[i]);
		__m256		diff = _mm256_sub_ps(va, vb);

		vsum = _mm256_fmadd_ps(diff, diff, vsum);
	}
	__m128		sum128 = _mm_add_ps(_mm256_castps256_ps128(vsum),
									_mm256_extractf128_ps(vsum, 1));

	sum128 = _mm_hadd_ps(sum128, sum128);
	sum128 = _mm_hadd_ps(sum128, sum128);
	sum = _mm_cvtss_f32(sum128);
	for (; i < dim; i++)
	{
		float		diff = a[i] - b[i];

		sum += diff * diff;
	}
#else
	for (int i = 0; i < dim; i++)
	{
		float		diff = a[i] - b[i];

		sum += diff * diff;
	}
#endif

	return sum;
}

/*
 * Find nearest centroid in a codebook stage
 */
static int
FindNearestCentroid(float *vector, float *centroids, int numCentroids, int dim)
{
	int			nearest = 0;
	float		minDist = FLT_MAX;

	for (int i = 0; i < numCentroids; i++)
	{
		float	   *centroid = &centroids[i * dim];
		float		dist = ComputeL2Squared(vector, centroid, dim);

		if (dist < minDist)
		{
			minDist = dist;
			nearest = i;
		}
	}

	return nearest;
}

/*
 * Subtract vector b from vector a (a = a - b)
 */
static void
SubtractVector(float *a, float *b, int dim)
{
#ifdef __AVX512F__
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a[i]);
		__m512		vb = _mm512_loadu_ps(&b[i]);

		_mm512_storeu_ps(&a[i], _mm512_sub_ps(va, vb));
	}
	for (; i < dim; i++)
		a[i] -= b[i];
#elif defined(__AVX2__)
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a[i]);
		__m256		vb = _mm256_loadu_ps(&b[i]);

		_mm256_storeu_ps(&a[i], _mm256_sub_ps(va, vb));
	}
	for (; i < dim; i++)
		a[i] -= b[i];
#else
	for (int i = 0; i < dim; i++)
		a[i] -= b[i];
#endif
}

/*
 * Add vector b to vector a (a = a + b)
 */
static void
AddVector(float *a, float *b, int dim)
{
#ifdef __AVX512F__
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		va = _mm512_loadu_ps(&a[i]);
		__m512		vb = _mm512_loadu_ps(&b[i]);

		_mm512_storeu_ps(&a[i], _mm512_add_ps(va, vb));
	}
	for (; i < dim; i++)
		a[i] += b[i];
#elif defined(__AVX2__)
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		va = _mm256_loadu_ps(&a[i]);
		__m256		vb = _mm256_loadu_ps(&b[i]);

		_mm256_storeu_ps(&a[i], _mm256_add_ps(va, vb));
	}
	for (; i < dim; i++)
		a[i] += b[i];
#else
	for (int i = 0; i < dim; i++)
		a[i] += b[i];
#endif
}

/*
 * K-means clustering for one stage
 */
static void
KMeansCluster(float *data, int n, int dim, float *centroids, int k, int maxIters)
{
	int		   *assignments = palloc(n * sizeof(int));
	int		   *counts = palloc(k * sizeof(int));
	float	   *newCentroids = palloc((Size) k * dim * sizeof(float));
	float	   *temp = palloc(dim * sizeof(float));

	/* Initialize centroids using random sampling */
	for (int i = 0; i < k; i++)
	{
		int			idx = (i * n) / k;	/* Spread samples evenly */

		memcpy(&centroids[i * dim], &data[idx * dim], dim * sizeof(float));
	}

	/* K-means iterations */
	for (int iter = 0; iter < maxIters; iter++)
	{
		/* Assign points to nearest centroids */
		for (int i = 0; i < n; i++)
		{
			float	   *point = &data[i * dim];

			assignments[i] = FindNearestCentroid(point, centroids, k, dim);
		}

		/* Compute new centroids */
		memset(newCentroids, 0, (Size) k * dim * sizeof(float));
		memset(counts, 0, k * sizeof(int));

		for (int i = 0; i < n; i++)
		{
			int			c = assignments[i];
			float	   *point = &data[i * dim];
			float	   *centroid = &newCentroids[c * dim];

			for (int d = 0; d < dim; d++)
				centroid[d] += point[d];
			counts[c]++;
		}

		/* Normalize centroids */
		for (int i = 0; i < k; i++)
		{
			if (counts[i] > 0)
			{
				float	   *centroid = &newCentroids[i * dim];
				float		scale = 1.0f / counts[i];

				for (int d = 0; d < dim; d++)
					centroid[d] *= scale;
			}
			else
			{
				/* Reinitialize empty cluster with random point */
				int			idx = rand() % n;

				memcpy(&newCentroids[i * dim], &data[idx * dim], dim * sizeof(float));
			}
		}

		/* Update centroids */
		memcpy(centroids, newCentroids, (Size) k * dim * sizeof(float));
	}

	pfree(assignments);
	pfree(counts);
	pfree(newCentroids);
	pfree(temp);
}

/*
 * Train RQ codebook on a set of vectors
 */
void
RQTrainCodebook(RQCodebook *cb, float *vectors, int numVectors)
{
	int			dim = cb->dim;
	int			numStages = cb->numStages;
	int			numCentroids = cb->numCentroids;
	float	   *residuals;

	/* Allocate residuals storage */
	residuals = palloc((Size) numVectors * dim * sizeof(float));

	/* Copy input vectors as initial residuals */
	memcpy(residuals, vectors, (Size) numVectors * dim * sizeof(float));

	/* Train each stage */
	for (int stage = 0; stage < numStages; stage++)
	{
		float	   *centroids = RQGetCentroid(cb, stage, 0);

		/* Train K-means on current residuals */
		KMeansCluster(residuals, numVectors, dim, centroids, numCentroids, RQ_KMEANS_ITERS);

		/* Compute new residuals for next stage */
		for (int i = 0; i < numVectors; i++)
		{
			float	   *residual = &residuals[i * dim];
			int			nearest = FindNearestCentroid(residual, centroids, numCentroids, dim);
			float	   *centroid = &centroids[nearest * dim];

			SubtractVector(residual, centroid, dim);
		}

		ereport(DEBUG1, (errmsg("RQ: trained stage %d/%d", stage + 1, numStages)));
	}

	pfree(residuals);
}

/*
 * Encode a single vector to RQ code
 */
void
RQEncode(RQCodebook *cb, float *vector, RQCode *code)
{
	int			dim = cb->dim;
	int			numStages = cb->numStages;
	int			numCentroids = cb->numCentroids;
	float	   *residual;
	Size		allocSize;

	/* Validate codebook parameters */
	if (dim <= 0 || dim > RQ_MAX_DIM)
	{
		ereport(WARNING, (errmsg("RQEncode: invalid dim %d", dim)));
		memset(code, 0, sizeof(RQCode));
		return;
	}
	if (numStages <= 0 || numStages > RQ_NUM_STAGES)
	{
		ereport(WARNING, (errmsg("RQEncode: invalid numStages %d", numStages)));
		memset(code, 0, sizeof(RQCode));
		return;
	}
	if (numCentroids <= 0 || numCentroids > RQ_NUM_CENTROIDS)
	{
		ereport(WARNING, (errmsg("RQEncode: invalid numCentroids %d", numCentroids)));
		memset(code, 0, sizeof(RQCode));
		return;
	}
	if (cb->centroids == NULL)
	{
		ereport(WARNING, (errmsg("RQEncode: centroids is NULL")));
		memset(code, 0, sizeof(RQCode));
		return;
	}

	allocSize = (Size)dim * sizeof(float);
	residual = palloc(allocSize);

	/* Start with original vector */
	memcpy(residual, vector, dim * sizeof(float));

	/* Quantize each stage */
	for (int stage = 0; stage < numStages; stage++)
	{
		float	   *centroids = RQGetCentroid(cb, stage, 0);
		int			nearest = FindNearestCentroid(residual, centroids, numCentroids, dim);
		float	   *centroid = &centroids[nearest * dim];

		code->codes[stage] = (uint8) nearest;
		SubtractVector(residual, centroid, dim);
	}

	pfree(residual);
}

/*
 * Decode RQ code to vector
 */
void
RQDecode(RQCodebook *cb, RQCode *code, float *output)
{
	int			dim = cb->dim;
	int			numStages = cb->numStages;

	/* Initialize output to zero */
	memset(output, 0, dim * sizeof(float));

	/* Sum all selected centroids */
	for (int stage = 0; stage < numStages; stage++)
	{
		float	   *centroid = RQGetCentroid(cb, stage, code->codes[stage]);

		AddVector(output, centroid, dim);
	}
}

/*
 * Encode halfvec to RQ code
 */
void
RQEncodeHalfvec(RQCodebook *cb, HalfVector *vec, RQCode *code)
{
	float	   *floatVec = palloc(vec->dim * sizeof(float));

	HalfvecToFloat(vec, floatVec);
	RQEncode(cb, floatVec, code);
	pfree(floatVec);
}

/*
 * Decode RQ code to halfvec
 */
void
RQDecodeToHalfvec(RQCodebook *cb, RQCode *code, HalfVector *output)
{
	float	   *floatVec = palloc(cb->dim * sizeof(float));

	RQDecode(cb, code, floatVec);
	FloatToHalfvec(floatVec, cb->dim, output);
	pfree(floatVec);
}

/*
 * Create distance lookup table for a query vector
 *
 * For asymmetric distance computation (ADC), we precompute the
 * squared distance from the query to each centroid at each stage.
 */
RQDistTable *
RQCreateDistTable(RQCodebook *cb, float *query)
{
	RQDistTable *dt;
	int			dim = cb->dim;
	int			numStages = cb->numStages;
	int			numCentroids = cb->numCentroids;

	dt = palloc(sizeof(RQDistTable));
	dt->dim = dim;
	dt->numStages = numStages;
	dt->numCentroids = numCentroids;
	dt->tables = palloc((Size) numStages * numCentroids * sizeof(float));

	/*
	 * For each stage and centroid, compute the contribution to the distance.
	 * For L2 squared distance: ||q - (c1 + c2 + ... + cM)||^2
	 * We use: sum_i ||q_partial - ci||^2 + cross terms
	 *
	 * Simplified approach: store ||q - c_i||^2 for each centroid
	 * Then reconstruct and compute exact distance (slower but more accurate)
	 *
	 * Or use an approximation based on cumulative residuals.
	 */

	/* For simplicity, precompute dot products: <query, centroid_i> */
	/* And ||centroid_i||^2 */
	for (int stage = 0; stage < numStages; stage++)
	{
		for (int c = 0; c < numCentroids; c++)
		{
			float	   *centroid = RQGetCentroid(cb, stage, c);
			float		dotProduct = 0.0f;
			float		normSq = 0.0f;

			for (int d = 0; d < dim; d++)
			{
				dotProduct += query[d] * centroid[d];
				normSq += centroid[d] * centroid[d];
			}

			/* Store: -2 * <q, c> + ||c||^2 for easy summation */
			dt->tables[stage * numCentroids + c] = -2.0f * dotProduct + normSq;
		}
	}

	return dt;
}

/*
 * Free distance lookup table
 */
void
RQFreeDistTable(RQDistTable *dt)
{
	if (dt)
	{
		if (dt->tables)
			pfree(dt->tables);
		pfree(dt);
	}
}

/*
 * Compute approximate L2 squared distance using lookup table
 *
 * This is an approximation. For exact distance, use RQComputeDistanceL2.
 * The approximation ignores cross terms between centroids from different stages.
 */
float
RQComputeDistance(RQDistTable *dt, RQCode *code)
{
	float		distance = 0.0f;
	int			numStages = dt->numStages;
	int			numCentroids = dt->numCentroids;

	for (int stage = 0; stage < numStages; stage++)
	{
		int			c = code->codes[stage];

		distance += dt->tables[stage * numCentroids + c];
	}

	return distance;
}

/*
 * Compute exact L2 squared distance (decode and compute)
 */
float
RQComputeDistanceL2(RQCodebook *cb, float *query, RQCode *code)
{
	float		distance = 0.0f;
	int			dim = cb->dim;
	float	   *decoded = palloc(dim * sizeof(float));

	RQDecode(cb, code, decoded);
	distance = ComputeL2Squared(query, decoded, dim);
	pfree(decoded);

	return distance;
}

/*
 * Batch encode vectors
 */
void
RQEncodeBatch(RQCodebook *cb, float *vectors, int numVectors, RQCode *codes)
{
	int			dim = cb->dim;

	for (int i = 0; i < numVectors; i++)
	{
		float	   *vector = &vectors[i * dim];

		RQEncode(cb, vector, &codes[i]);
	}
}

/*
 * Batch compute distances
 */
void
RQComputeDistanceBatch(RQDistTable *dt, RQCode *codes, int numCodes, float *distances)
{
	for (int i = 0; i < numCodes; i++)
		distances[i] = RQComputeDistance(dt, &codes[i]);
}

/*
 * Compute serialized size of codebook
 */
Size
RQCodebookSize(RQCodebook *cb)
{
	Size		size = 0;

	size += sizeof(int) * 3;	/* dim, numStages, numCentroids */
	size += (Size) cb->numStages * cb->numCentroids * cb->dim * sizeof(float);

	return size;
}

/*
 * Serialize codebook to buffer
 */
void
RQSerializeCodebook(RQCodebook *cb, char *buffer)
{
	int		   *intPtr = (int *) buffer;
	float	   *floatPtr;
	Size		centroidSize;

	intPtr[0] = cb->dim;
	intPtr[1] = cb->numStages;
	intPtr[2] = cb->numCentroids;

	floatPtr = (float *) (intPtr + 3);
	centroidSize = (Size) cb->numStages * cb->numCentroids * cb->dim * sizeof(float);
	memcpy(floatPtr, cb->centroids, centroidSize);
}

/*
 * Deserialize codebook from buffer
 */
RQCodebook *
RQDeserializeCodebook(char *buffer)
{
	int		   *intPtr = (int *) buffer;
	float	   *floatPtr;
	RQCodebook *cb;
	Size		centroidSize;

	cb = palloc(sizeof(RQCodebook));
	cb->dim = intPtr[0];
	cb->numStages = intPtr[1];
	cb->numCentroids = intPtr[2];

	floatPtr = (float *) (intPtr + 3);
	centroidSize = (Size) cb->numStages * cb->numCentroids * cb->dim * sizeof(float);
	cb->centroids = palloc(centroidSize);
	memcpy(cb->centroids, floatPtr, centroidSize);

	return cb;
}

#ifdef __AVX512F__
/*
 * AVX-512 optimized distance computation
 */
float
RQComputeDistanceAVX512(RQDistTable *dt, RQCode *code)
{
	float		distance = 0.0f;
	int			numStages = dt->numStages;
	int			numCentroids = dt->numCentroids;

	/* Simple loop - could be vectorized if processing multiple codes */
	for (int stage = 0; stage < numStages; stage++)
	{
		int			c = code->codes[stage];

		distance += dt->tables[stage * numCentroids + c];
	}

	return distance;
}

/*
 * AVX-512 batch distance computation
 */
void
RQComputeDistanceBatchAVX512(RQDistTable *dt, RQCode *codes, int numCodes, float *distances)
{
	int			numStages = dt->numStages;
	int			numCentroids = dt->numCentroids;
	int			i = 0;

	/* Process 16 codes at a time */
	for (; i + 16 <= numCodes; i += 16)
	{
		__m512		vdist = _mm512_setzero_ps();

		for (int stage = 0; stage < numStages; stage++)
		{
			/* Gather table values for 16 codes */
			__m512i		indices = _mm512_set_epi32(
											   codes[i + 15].codes[stage],
												   codes[i + 14].codes[stage],
												   codes[i + 13].codes[stage],
												   codes[i + 12].codes[stage],
												   codes[i + 11].codes[stage],
												   codes[i + 10].codes[stage],
												   codes[i + 9].codes[stage],
												   codes[i + 8].codes[stage],
												   codes[i + 7].codes[stage],
												   codes[i + 6].codes[stage],
												   codes[i + 5].codes[stage],
												   codes[i + 4].codes[stage],
												   codes[i + 3].codes[stage],
												   codes[i + 2].codes[stage],
												   codes[i + 1].codes[stage],
												   codes[i + 0].codes[stage]);
			float	   *tableBase = &dt->tables[stage * numCentroids];
			__m512		values = _mm512_i32gather_ps(indices, tableBase, 4);

			vdist = _mm512_add_ps(vdist, values);
		}

		_mm512_storeu_ps(&distances[i], vdist);
	}

	/* Handle remaining codes */
	for (; i < numCodes; i++)
		distances[i] = RQComputeDistance(dt, &codes[i]);
}
#endif

#ifdef __AVX2__
/*
 * AVX2 optimized distance computation
 */
float
RQComputeDistanceAVX2(RQDistTable *dt, RQCode *code)
{
	/* Same as scalar for single code */
	return RQComputeDistance(dt, code);
}
#endif
