/*
 * RaBitQ (Randomized Bit Quantization) - SIGMOD 2024
 *
 * Implements efficient binary quantization for vector similarity search.
 * D-dimensional vectors are encoded as D-bit strings using random rotation
 * and sign quantization, enabling O(1) distance computation via popcount.
 *
 * Reference: https://arxiv.org/abs/2405.12497
 */

#ifndef RABITQ_H
#define RABITQ_H

#include "postgres.h"
#include <stdint.h>
#include <math.h>

/* Maximum supported dimensions for RaBitQ */
#define RABITQ_MAX_DIM 2048

/* Number of uint64 words needed for given dimension */
#define RABITQ_WORDS(dim) (((dim) + 63) / 64)

/* Maximum words for RABITQ_MAX_DIM */
#define RABITQ_MAX_WORDS 32

/*
 * RaBitQ encoded vector
 *
 * For D dimensions, we use ceil(D/64) uint64 words to store D bits.
 * The norm is stored separately for distance estimation.
 */
typedef struct RaBitQCode
{
	uint64		bits[RABITQ_MAX_WORDS];	/* Bit-packed quantized values */
	float		norm;					/* L2 norm of original vector */
} RaBitQCode;

/*
 * Compact RaBitQ code for storage (128-dim vectors = 20 bytes)
 * This is what gets stored in index pages
 */
typedef struct RaBitQCode128
{
	uint64		bits[2];	/* 128 bits for 128 dimensions */
	float		norm;		/* 4 bytes */
} RaBitQCode128;			/* Total: 20 bytes */

/*
 * RaBitQ Encoder state
 *
 * Contains the random orthogonal rotation matrix used during encoding.
 * The same encoder must be used for both index building and querying.
 */
typedef struct RaBitQEncoder
{
	int			dim;			/* Vector dimension */
	int			nwords;			/* Number of uint64 words needed */
	uint64		seed;			/* Random seed for reproducibility */
	float	   *rotationMatrix;	/* D×D orthogonal matrix (row-major) */
	float	   *meanVector;		/* Optional: training set mean (D floats) */
	bool		initialized;	/* Whether encoder is ready */
} RaBitQEncoder;

/*
 * Precomputed query state for asymmetric distance computation
 *
 * When processing a query, we pre-encode it once and reuse for all comparisons.
 */
typedef struct RaBitQQueryState
{
	RaBitQCode	queryCode;		/* Encoded query */
	float		queryNorm;		/* Query L2 norm */
	float	   *rotatedQuery;	/* Query after rotation (for ADC) */
} RaBitQQueryState;

/* ============== Core Functions ============== */

/*
 * Initialize a RaBitQ encoder with given dimension and random seed.
 * This generates the random orthogonal rotation matrix.
 */
void RaBitQEncoderInit(RaBitQEncoder *enc, int dim, uint64 seed);

/*
 * Free resources allocated by RaBitQEncoderInit
 */
void RaBitQEncoderFree(RaBitQEncoder *enc);

/*
 * Encode a float vector into RaBitQ binary code
 */
void RaBitQEncode(RaBitQEncoder *enc, const float *vector, RaBitQCode *code);

/*
 * Encode a halfvec (float16) vector into RaBitQ binary code
 */
void RaBitQEncodeHalfvec(RaBitQEncoder *enc, const void *halfvec, RaBitQCode *code);

/*
 * Compute approximate squared L2 distance between two encoded vectors
 * Uses XOR + popcount for O(1) computation
 */
float RaBitQDistance(const RaBitQCode *a, const RaBitQCode *b, int dim);

/*
 * Compute approximate distance from a float query to an encoded vector
 * More accurate than code-to-code comparison (asymmetric)
 */
float RaBitQAsymmetricDistance(const float *query, float queryNorm,
							   const RaBitQCode *code, RaBitQEncoder *enc);

/* ============== Query Optimization ============== */

/*
 * Prepare query state for efficient repeated distance computations
 */
void RaBitQPrepareQuery(RaBitQEncoder *enc, const float *query,
						RaBitQQueryState *state);

/*
 * Fast distance using prepared query state
 */
float RaBitQFastDistance(const RaBitQQueryState *query, const RaBitQCode *code,
						 int dim);

/* ============== SIMD-Optimized Functions ============== */

#ifdef __AVX512F__
/*
 * AVX-512 optimized popcount for multiple codes
 */
void RaBitQBatchDistance_AVX512(const RaBitQCode *query, const RaBitQCode *codes,
								int nCodes, int dim, float *distances);
#endif

#ifdef __AVX2__
/*
 * AVX2 optimized rotation and encoding
 */
void RaBitQEncode_AVX2(RaBitQEncoder *enc, const float *vector, RaBitQCode *code);
#endif

/* ============== Serialization ============== */

/*
 * Get serialized size of encoder (for storage)
 */
Size RaBitQEncoderSerializedSize(int dim);

/*
 * Serialize encoder to buffer
 */
void RaBitQEncoderSerialize(RaBitQEncoder *enc, char *buffer);

/*
 * Deserialize encoder from buffer
 */
void RaBitQEncoderDeserialize(RaBitQEncoder *enc, const char *buffer, int dim);

/* ============== Utility Functions ============== */

/*
 * Compute Hamming distance between two bit vectors
 */
static inline int
RaBitQHamming(const uint64 *a, const uint64 *b, int nwords)
{
	int			hamming = 0;

	for (int i = 0; i < nwords; i++)
	{
		hamming += __builtin_popcountll(a[i] ^ b[i]);
	}
	return hamming;
}

/*
 * Compute Hamming distance for 128-bit codes (optimized for common case)
 */
static inline int
RaBitQHamming128(const RaBitQCode128 *a, const RaBitQCode128 *b)
{
	return __builtin_popcountll(a->bits[0] ^ b->bits[0]) +
		   __builtin_popcountll(a->bits[1] ^ b->bits[1]);
}

/*
 * Convert Hamming distance to approximate cosine similarity
 * cosine ≈ 1 - 2*hamming/dim
 */
static inline float
RaBitQHammingToCosine(int hamming, int dim)
{
	return 1.0f - 2.0f * hamming / (float) dim;
}

/*
 * Convert cosine similarity to squared L2 distance (for unit vectors)
 * ||a - b||² = 2(1 - cos(a,b))
 */
static inline float
RaBitQCosineToL2Squared(float cosine, float normA, float normB)
{
	/* For non-unit vectors: ||a - b||² = ||a||² + ||b||² - 2*||a||*||b||*cos */
	return normA * normA + normB * normB - 2.0f * normA * normB * cosine;
}

#endif							/* RABITQ_H */
