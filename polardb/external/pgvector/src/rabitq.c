/*
 * RaBitQ (Randomized Bit Quantization) Implementation
 *
 * SIGMOD 2024 paper implementation for efficient binary vector quantization.
 *
 * Algorithm:
 * 1. Normalize input vector: x' = x / ||x||
 * 2. Random rotation: x'' = R * x' (R is random orthogonal matrix)
 * 3. Sign quantization: b[i] = (x''[i] > 0) ? 1 : 0
 *
 * Distance computation:
 * d(q, x) ≈ ||q|| * ||x|| * (1 - 2*popcount(q_bits XOR x_bits)/D)
 */

#include "postgres.h"
#include "rabitq.h"
#include "halfvec.h"
#include "halfutils.h"

#include <string.h>
#include <math.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ============== Random Orthogonal Matrix Generation ============== */

/*
 * Simple xorshift64 PRNG for reproducibility
 */
static uint64
xorshift64(uint64 *state)
{
	uint64		x = *state;

	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*state = x;
	return x;
}

/*
 * Generate random float in [0, 1)
 */
static float
random_float(uint64 *state)
{
	return (float) (xorshift64(state) >> 11) / (float) (1ULL << 53);
}

/*
 * Generate random Gaussian using Box-Muller transform
 */
static float
random_gaussian(uint64 *state)
{
	float		u1 = random_float(state);
	float		u2 = random_float(state);

	/* Avoid log(0) */
	if (u1 < 1e-10f)
		u1 = 1e-10f;

	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/*
 * Gram-Schmidt orthogonalization for a set of vectors
 * Modifies vectors in-place to make them orthonormal
 */
static void
gram_schmidt(float *vectors, int n, int dim)
{
	for (int i = 0; i < n; i++)
	{
		float	   *vi = vectors + i * dim;

		/* Subtract projections onto previous vectors */
		for (int j = 0; j < i; j++)
		{
			float	   *vj = vectors + j * dim;
			float		dot = 0.0f;

			for (int k = 0; k < dim; k++)
				dot += vi[k] * vj[k];

			for (int k = 0; k < dim; k++)
				vi[k] -= dot * vj[k];
		}

		/* Normalize */
		float		norm = 0.0f;

		for (int k = 0; k < dim; k++)
			norm += vi[k] * vi[k];
		norm = sqrtf(norm);

		if (norm > 1e-10f)
		{
			for (int k = 0; k < dim; k++)
				vi[k] /= norm;
		}
	}
}

/*
 * Generate random orthogonal matrix using Gram-Schmidt
 *
 * For better numerical stability with high dimensions, we generate
 * random Gaussian vectors and orthogonalize them.
 */
static void
generate_orthogonal_matrix(float *matrix, int dim, uint64 seed)
{
	uint64		state = seed;

	/* Fill with random Gaussian values */
	for (int i = 0; i < dim * dim; i++)
		matrix[i] = random_gaussian(&state);

	/* Orthogonalize using Gram-Schmidt */
	gram_schmidt(matrix, dim, dim);
}

/* ============== Encoder Functions ============== */

/*
 * Initialize RaBitQ encoder
 */
void
RaBitQEncoderInit(RaBitQEncoder *enc, int dim, uint64 seed)
{
	if (dim > RABITQ_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("RaBitQ dimension %d exceeds maximum %d", dim, RABITQ_MAX_DIM)));

	enc->dim = dim;
	enc->nwords = RABITQ_WORDS(dim);
	enc->seed = seed;
	enc->initialized = false;

	/* Allocate rotation matrix */
	enc->rotationMatrix = (float *) palloc(dim * dim * sizeof(float));

	/* Generate random orthogonal matrix */
	generate_orthogonal_matrix(enc->rotationMatrix, dim, seed);

	/* No mean vector by default */
	enc->meanVector = NULL;

	enc->initialized = true;
}

/*
 * Free encoder resources
 */
void
RaBitQEncoderFree(RaBitQEncoder *enc)
{
	if (enc->rotationMatrix)
		pfree(enc->rotationMatrix);
	if (enc->meanVector)
		pfree(enc->meanVector);

	enc->rotationMatrix = NULL;
	enc->meanVector = NULL;
	enc->initialized = false;
}

/* ============== Encoding Functions ============== */

/*
 * Encode a float vector to RaBitQ binary code
 *
 * Steps:
 * 1. Compute and store L2 norm
 * 2. Normalize to unit vector
 * 3. Apply random rotation
 * 4. Sign quantization to bits
 */
void
RaBitQEncode(RaBitQEncoder *enc, const float *vector, RaBitQCode *code)
{
	int			dim = enc->dim;
	int			nwords = enc->nwords;
	float	   *R = enc->rotationMatrix;
	float		norm = 0.0f;
	float	   *normalized;
	float	   *rotated;

	/* Compute L2 norm */
	for (int i = 0; i < dim; i++)
		norm += vector[i] * vector[i];
	norm = sqrtf(norm);
	code->norm = norm;

	/* Handle zero vector */
	if (norm < 1e-10f)
	{
		memset(code->bits, 0, nwords * sizeof(uint64));
		return;
	}

	/* Allocate temporary buffers */
	normalized = (float *) palloc(dim * sizeof(float));
	rotated = (float *) palloc(dim * sizeof(float));

	/* Normalize */
	float		inv_norm = 1.0f / norm;

	for (int i = 0; i < dim; i++)
		normalized[i] = vector[i] * inv_norm;

	/* Apply rotation: rotated = R * normalized */
#ifdef __AVX512F__
	/* AVX-512 optimized matrix-vector multiplication */
	for (int i = 0; i < dim; i++)
	{
		__m512		vsum = _mm512_setzero_ps();
		int			j = 0;

		for (; j + 16 <= dim; j += 16)
		{
			__m512		vr = _mm512_loadu_ps(&R[i * dim + j]);
			__m512		vn = _mm512_loadu_ps(&normalized[j]);

			vsum = _mm512_fmadd_ps(vr, vn, vsum);
		}

		float		sum = _mm512_reduce_add_ps(vsum);

		/* Handle remaining elements */
		for (; j < dim; j++)
			sum += R[i * dim + j] * normalized[j];

		rotated[i] = sum;
	}
#elif defined(__AVX2__)
	/* AVX2 optimized */
	for (int i = 0; i < dim; i++)
	{
		__m256		vsum = _mm256_setzero_ps();
		int			j = 0;

		for (; j + 8 <= dim; j += 8)
		{
			__m256		vr = _mm256_loadu_ps(&R[i * dim + j]);
			__m256		vn = _mm256_loadu_ps(&normalized[j]);

			vsum = _mm256_fmadd_ps(vr, vn, vsum);
		}

		/* Horizontal sum */
		__m128		vlow = _mm256_castps256_ps128(vsum);
		__m128		vhigh = _mm256_extractf128_ps(vsum, 1);

		vlow = _mm_add_ps(vlow, vhigh);
		__m128		shuf = _mm_movehdup_ps(vlow);

		vlow = _mm_add_ps(vlow, shuf);
		shuf = _mm_movehl_ps(shuf, vlow);
		vlow = _mm_add_ss(vlow, shuf);
		float		sum = _mm_cvtss_f32(vlow);

		/* Handle remaining elements */
		for (; j < dim; j++)
			sum += R[i * dim + j] * normalized[j];

		rotated[i] = sum;
	}
#else
	/* Scalar fallback */
	for (int i = 0; i < dim; i++)
	{
		float		sum = 0.0f;

		for (int j = 0; j < dim; j++)
			sum += R[i * dim + j] * normalized[j];
		rotated[i] = sum;
	}
#endif

	/* Sign quantization: bit[i] = (rotated[i] > 0) ? 1 : 0 */
	memset(code->bits, 0, nwords * sizeof(uint64));

	for (int i = 0; i < dim; i++)
	{
		if (rotated[i] > 0.0f)
		{
			int			word = i / 64;
			int			bit = i % 64;

			code->bits[word] |= (1ULL << bit);
		}
	}

	pfree(normalized);
	pfree(rotated);
}

/*
 * Encode halfvec (float16) to RaBitQ
 */
void
RaBitQEncodeHalfvec(RaBitQEncoder *enc, const void *halfvec_ptr, RaBitQCode *code)
{
	HalfVector *hv = (HalfVector *) halfvec_ptr;
	int			dim = hv->dim;
	float	   *floats;

	/* Convert halfvec to float */
	floats = (float *) palloc(dim * sizeof(float));
	for (int i = 0; i < dim; i++)
		floats[i] = HalfToFloat4(hv->x[i]);

	/* Encode as float */
	RaBitQEncode(enc, floats, code);

	pfree(floats);
}

/* ============== Distance Functions ============== */

/*
 * Compute approximate squared L2 distance between two RaBitQ codes
 *
 * Formula:
 * cosine ≈ 1 - 2*hamming/dim
 * dist² ≈ ||a||² + ||b||² - 2*||a||*||b||*cosine
 */
float
RaBitQDistance(const RaBitQCode *a, const RaBitQCode *b, int dim)
{
	int			nwords = RABITQ_WORDS(dim);
	int			hamming = 0;

	/* Compute Hamming distance using popcount */
	for (int i = 0; i < nwords; i++)
	{
		hamming += __builtin_popcountll(a->bits[i] ^ b->bits[i]);
	}

	/* Convert to approximate cosine */
	float		cosine = 1.0f - 2.0f * hamming / (float) dim;

	/* Convert to squared L2 distance */
	float		normA = a->norm;
	float		normB = b->norm;

	return normA * normA + normB * normB - 2.0f * normA * normB * cosine;
}

/*
 * Asymmetric distance computation (query float -> code)
 *
 * More accurate than symmetric because we use full precision for query.
 */
float
RaBitQAsymmetricDistance(const float *query, float queryNorm,
						 const RaBitQCode *code, RaBitQEncoder *enc)
{
	int			dim = enc->dim;
	float	   *R = enc->rotationMatrix;
	float	   *rotatedQuery;
	float		dotProduct = 0.0f;

	/* Normalize query */
	if (queryNorm < 1e-10f)
		return code->norm * code->norm;	/* Distance to zero vector */

	float		invNorm = 1.0f / queryNorm;

	/* Allocate and compute rotated query */
	rotatedQuery = (float *) palloc(dim * sizeof(float));

	for (int i = 0; i < dim; i++)
	{
		float		sum = 0.0f;

		for (int j = 0; j < dim; j++)
			sum += R[i * dim + j] * query[j] * invNorm;
		rotatedQuery[i] = sum;
	}

	/*
	 * Compute dot product with sign-quantized code For each bit: if bit[i] =
	 * 1, we add +rotatedQuery[i], else -rotatedQuery[i] This is equivalent
	 * to: sum of rotatedQuery[i] * (2*bit[i] - 1)
	 */
	for (int i = 0; i < dim; i++)
	{
		int			word = i / 64;
		int			bit = i % 64;
		int			sign = (code->bits[word] >> bit) & 1;

		/* sign = 0 -> -1, sign = 1 -> +1 */
		dotProduct += rotatedQuery[i] * (2 * sign - 1);
	}

	pfree(rotatedQuery);

	/* Scale by norms and convert to distance */
	float		cosine = dotProduct;	/* Already normalized */
	float		codeNorm = code->norm;

	return queryNorm * queryNorm + codeNorm * codeNorm - 2.0f * queryNorm * codeNorm * cosine;
}

/* ============== Query Optimization ============== */

/*
 * Prepare query state for fast repeated comparisons
 */
void
RaBitQPrepareQuery(RaBitQEncoder *enc, const float *query, RaBitQQueryState *state)
{
	int			dim = enc->dim;

	/* Compute query norm */
	state->queryNorm = 0.0f;
	for (int i = 0; i < dim; i++)
		state->queryNorm += query[i] * query[i];
	state->queryNorm = sqrtf(state->queryNorm);

	/* Encode query */
	RaBitQEncode(enc, query, &state->queryCode);

	/* Allocate and store rotated query for ADC */
	state->rotatedQuery = (float *) palloc(dim * sizeof(float));

	if (state->queryNorm > 1e-10f)
	{
		float	   *R = enc->rotationMatrix;
		float		invNorm = 1.0f / state->queryNorm;

		for (int i = 0; i < dim; i++)
		{
			float		sum = 0.0f;

			for (int j = 0; j < dim; j++)
				sum += R[i * dim + j] * query[j] * invNorm;
			state->rotatedQuery[i] = sum;
		}
	}
	else
	{
		memset(state->rotatedQuery, 0, dim * sizeof(float));
	}
}

/*
 * Fast distance using prepared query (symmetric popcount)
 */
float
RaBitQFastDistance(const RaBitQQueryState *query, const RaBitQCode *code, int dim)
{
	return RaBitQDistance(&query->queryCode, code, dim);
}

/* ============== Serialization ============== */

/*
 * Get size needed to serialize encoder
 */
Size
RaBitQEncoderSerializedSize(int dim)
{
	/* dim (4) + nwords (4) + seed (8) + rotation matrix (dim*dim*4) */
	return sizeof(int32) * 2 + sizeof(uint64) + dim * dim * sizeof(float);
}

/*
 * Serialize encoder to buffer
 */
void
RaBitQEncoderSerialize(RaBitQEncoder *enc, char *buffer)
{
	char	   *ptr = buffer;
	int32		dim32 = enc->dim;
	int32		nwords32 = enc->nwords;

	/* Write header */
	memcpy(ptr, &dim32, sizeof(int32));
	ptr += sizeof(int32);
	memcpy(ptr, &nwords32, sizeof(int32));
	ptr += sizeof(int32);
	memcpy(ptr, &enc->seed, sizeof(uint64));
	ptr += sizeof(uint64);

	/* Write rotation matrix */
	memcpy(ptr, enc->rotationMatrix, enc->dim * enc->dim * sizeof(float));
}

/*
 * Deserialize encoder from buffer
 */
void
RaBitQEncoderDeserialize(RaBitQEncoder *enc, const char *buffer, int dim)
{
	const char *ptr = buffer;
	int32		dim32,
				nwords32;

	/* Read header */
	memcpy(&dim32, ptr, sizeof(int32));
	ptr += sizeof(int32);
	memcpy(&nwords32, ptr, sizeof(int32));
	ptr += sizeof(int32);
	memcpy(&enc->seed, ptr, sizeof(uint64));
	ptr += sizeof(uint64);

	/* Validate */
	if (dim32 != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("RaBitQ encoder dimension mismatch: expected %d, got %d", dim, dim32)));

	enc->dim = dim32;
	enc->nwords = nwords32;

	/* Allocate and read rotation matrix */
	enc->rotationMatrix = (float *) palloc(dim * dim * sizeof(float));
	memcpy(enc->rotationMatrix, ptr, dim * dim * sizeof(float));

	enc->meanVector = NULL;
	enc->initialized = true;
}

/* ============== SQ4 Query Optimization ============== */

/*
 * Global function pointer for SQ4 distance computation
 * Set at module initialization based on CPU capabilities
 */
RaBitQSQ4DistanceFunc RaBitQSQ4Distance = NULL;

/*
 * Prepare SQ4 query state for efficient distance computations
 *
 * Quantizes the query vector to 4-bit and reorganizes by bit position
 * for efficient bitwise AND + popcount distance calculations.
 */
void
RaBitQPrepareSQ4Query(RaBitQEncoder *enc, const float *query,
					  RaBitQQueryStateSQ4 *state)
{
	int			dim = enc->dim;
	int			nwords = RABITQ_WORDS(dim);

	/* 1. Normalize query vector */
	float		norm = 0.0f;
	for (int i = 0; i < dim; i++)
		norm += query[i] * query[i];
	norm = sqrtf(norm);
	state->queryNorm = norm;

	if (norm < 1e-10f)
	{
		/* Zero vector - initialize zero state */
		for (int b = 0; b < 4; b++)
		{
			state->sq4Bits[b] = (uint64 *) palloc0(nwords * sizeof(uint64));
		}
		state->queryBits = (uint64 *) palloc0(nwords * sizeof(uint64));
		state->rotatedQuery = (float *) palloc0(dim * sizeof(float));
		state->sq4Total = 0;
		state->nwords = nwords;
		state->dim = dim;
		return;
	}

	/* 2. Normalize input vector */
	float	   *normalized = (float *) palloc(dim * sizeof(float));
	for (int i = 0; i < dim; i++)
		normalized[i] = query[i] / norm;

	/* 3. Rotate: rotated = R * normalized */
	float	   *rotated = (float *) palloc(dim * sizeof(float));
	for (int i = 0; i < dim; i++)
	{
		float		sum = 0.0f;
		const float *row = enc->rotationMatrix + i * dim;
		for (int j = 0; j < dim; j++)
			sum += row[j] * normalized[j];
		rotated[i] = sum;
	}

	state->rotatedQuery = rotated;

	/* 4. Find min/max for SQ4 quantization bounds */
	float		minVal = rotated[0],
				maxVal = rotated[0];
	for (int i = 1; i < dim; i++)
	{
		if (rotated[i] < minVal)
			minVal = rotated[i];
		if (rotated[i] > maxVal)
			maxVal = rotated[i];
	}

	float		delta = (maxVal - minVal) / 15.0f;	/* 4-bit range [0,15] */
	if (delta < 1e-10f)
		delta = 1e-10f;

	/* 5. Quantize to 4-bit and compute binary sign bits */
	uint8	   *quantized = (uint8 *) palloc(dim);
	int			sq4Total = 0;

	state->queryBits = (uint64 *) palloc0(nwords * sizeof(uint64));

	for (int i = 0; i < dim; i++)
	{
		int			val = (int) ((rotated[i] - minVal) / delta);
		if (val > 15)
			val = 15;
		if (val < 0)
			val = 0;
		quantized[i] = (uint8) val;
		sq4Total += val;

		/* Binary sign quantization: bit = 1 if rotated value is positive */
		if (rotated[i] > 0.0f)
		{
			state->queryBits[i / 64] |= (1ULL << (i % 64));
		}
	}

	state->sq4Total = sq4Total;

	/* 6. Reorganize by bit position (horizontal to vertical) */
	for (int b = 0; b < 4; b++)
	{
		state->sq4Bits[b] = (uint64 *) palloc0(nwords * sizeof(uint64));
		for (int d = 0; d < dim; d++)
		{
			if (quantized[d] & (1 << b))
			{
				state->sq4Bits[b][d / 64] |= (1ULL << (d % 64));
			}
		}
	}

	state->nwords = nwords;
	state->dim = dim;
	pfree(normalized);
	pfree(quantized);
}

/*
 * Free SQ4 query state resources
 */
void
RaBitQFreeSQ4Query(RaBitQQueryStateSQ4 *state)
{
	if (state == NULL)
		return;

	for (int b = 0; b < 4; b++)
	{
		if (state->sq4Bits[b] != NULL)
			pfree(state->sq4Bits[b]);
	}
	if (state->queryBits != NULL)
		pfree(state->queryBits);
	if (state->rotatedQuery != NULL)
		pfree(state->rotatedQuery);
}

/*
 * Generic SQ4 distance computation
 *
 * Uses binary Hamming distance between sign-quantized query and code.
 * Formula: dist² ≈ ||q||² + ||c||² - 2*||q||*||c||*cosine
 * where cosine ≈ 1 - 2*hamming/dim
 */
float
RaBitQSQ4Distance_Generic(const RaBitQQueryStateSQ4 *query,
						  const RaBitQCode256 *code)
{
	int			hamming = 0;
	static int	debug_count = 0;

	/* Compute Hamming distance using XOR + popcount */
	for (int w = 0; w < query->nwords; w++)
	{
		hamming += __builtin_popcountll(query->queryBits[w] ^ code->bits[w]);
	}

	/* Convert Hamming to approximate cosine similarity */
	float		cosine = 1.0f - 2.0f * hamming / (float) query->dim;

	/* Convert to squared L2 distance */
	float		dist = query->queryNorm * query->queryNorm +
		code->norm * code->norm -
		2.0f * query->queryNorm * code->norm * cosine;

	/* Debug output for first few calls */
	if (debug_count < 10)
	{
		elog(DEBUG1, "RaBitQ: hamming=%d dim=%d cosine=%.4f qNorm=%.4f cNorm=%.4f dist=%.4f",
			 hamming, query->dim, cosine, query->queryNorm, code->norm, dist);
		debug_count++;
	}

	return dist;
}

/* ============== AVX-512 Batch Processing ============== */

#ifdef __AVX512F__

/*
 * Process 8 codes at once using AVX-512 popcount
 */
void
RaBitQBatchDistance_AVX512(const RaBitQCode *query, const RaBitQCode *codes,
						   int nCodes, int dim, float *distances)
{
	int			nwords = RABITQ_WORDS(dim);
	float		queryNorm = query->norm;
	float		queryNormSq = queryNorm * queryNorm;
	float		dimInv = 2.0f / (float) dim;

	for (int i = 0; i < nCodes; i++)
	{
		int			hamming = 0;

		for (int w = 0; w < nwords; w++)
		{
			hamming += __builtin_popcountll(query->bits[w] ^ codes[i].bits[w]);
		}

		float		cosine = 1.0f - dimInv * hamming;
		float		codeNorm = codes[i].norm;

		distances[i] = queryNormSq + codeNorm * codeNorm - 2.0f * queryNorm * codeNorm * cosine;
	}
}

/*
 * AVX-512 optimized SQ4 distance computation using VPOPCNTDQ
 *
 * Uses binary Hamming distance with hardware popcount instruction.
 */
float
RaBitQSQ4Distance_AVX512(const RaBitQQueryStateSQ4 *query,
						 const RaBitQCode256 *code)
{
	/* Just call generic version for now to use shared debug code */
	return RaBitQSQ4Distance_Generic(query, code);
}

#endif							/* __AVX512F__ */

#ifdef __AVX2__

/*
 * AVX2 optimized SQ4 distance computation
 *
 * Since AVX2 doesn't have POPCNT instruction for 64-bit values,
 * we use the same scalar popcount but with vectorized data loading.
 * For full optimization, could use lookup tables for popcount.
 */
float
RaBitQSQ4Distance_AVX2(const RaBitQQueryStateSQ4 *query,
					   const RaBitQCode256 *code)
{
	/* For now, use same implementation as Generic */
	/* Full AVX2 optimization with LUT popcount can be added later */
	return RaBitQSQ4Distance_Generic(query, code);
}

#endif							/* __AVX2__ */

/* ============== Runtime Dispatch ============== */

/*
 * Initialize SIMD dispatch for RaBitQ SQ4 distance computation
 *
 * Detects CPU capabilities and sets the global function pointer
 * to the fastest available implementation.
 */
void
RaBitQInit(void)
{
	static bool initialized = false;

	if (initialized)
		return;

	/* Initialize CPUID detection */
	__builtin_cpu_init();

	/* Select best available implementation */
#ifdef __AVX512F__
	if (__builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512vpopcntdq"))
	{
		RaBitQSQ4Distance = RaBitQSQ4Distance_AVX512;
		elog(DEBUG1, "RaBitQ using AVX-512 with VPOPCNTDQ");
	}
	else
#endif
#ifdef __AVX2__
	if (__builtin_cpu_supports("avx2"))
	{
		RaBitQSQ4Distance = RaBitQSQ4Distance_AVX2;
		elog(DEBUG1, "RaBitQ using AVX2");
	}
	else
#endif
	{
		RaBitQSQ4Distance = RaBitQSQ4Distance_Generic;
		elog(DEBUG1, "RaBitQ using Generic (scalar) implementation");
	}

	initialized = true;
}
