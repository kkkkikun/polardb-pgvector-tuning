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
 * Extended RaBitQ code for 256-dim vectors (36 bytes)
 * Supports up to 256 dimensions (e.g., 200-dim vectors)
 */
typedef struct RaBitQCode256
{
	uint64		bits[4];	/* 256 bits for up to 256 dimensions */
	float		norm;		/* 4 bytes */
} RaBitQCode256;			/* Total: 36 bytes */

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

/*
 * SQ4 (4-bit quantized query) state for fast distance computation
 *
 * Stores query vector as 4 bit-planes for efficient bitwise distance calculation.
 * Query vector is quantized to 4-bit and reorganized by bit position.
 * Distance computation uses bitwise AND + popcount with weighted bit accumulation.
 */
typedef struct RaBitQQueryStateSQ4
{
	uint64	   *sq4Bits[4];		/* 4 bit-planes (bit0, bit1, bit2, bit3) */
	uint64	   *queryBits;		/* Binary sign quantization (for Hamming fallback) */
	float		queryNorm;		/* Query L2 norm */
	float	   *rotatedQuery;	/* Query after rotation (for min/max bounds) */
	int			sq4Total;		/* Sum of all SQ4 values (for correct formula) */
	int			nwords;			/* Number of uint64 words for bit storage */
	int			dim;			/* Vector dimensionality */
} RaBitQQueryStateSQ4;

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

/* ============== SQ4 Query Optimization ============== */

/*
 * Prepare SQ4 query state for fast distance computations
 * Quantizes query to 4-bit and reorganizes by bit position
 */
void RaBitQPrepareSQ4Query(RaBitQEncoder *enc, const float *query,
						   RaBitQQueryStateSQ4 *state);

/*
 * Free SQ4 query state resources
 */
void RaBitQFreeSQ4Query(RaBitQQueryStateSQ4 *state);

/*
 * Compute distance using SQ4-optimized query state (Generic version)
 * Uses bitwise AND + popcount with weighted bit accumulation
 */
float RaBitQSQ4Distance_Generic(const RaBitQQueryStateSQ4 *query,
								const RaBitQCode256 *code);

/* ============== SIMD-Optimized Functions ============== */

#ifdef __AVX512F__
/*
 * AVX-512 optimized popcount for multiple codes
 */
void RaBitQBatchDistance_AVX512(const RaBitQCode *query, const RaBitQCode *codes,
								int nCodes, int dim, float *distances);

/*
 * AVX-512 optimized SQ4 distance computation using VPOPCNTDQ
 */
float RaBitQSQ4Distance_AVX512(const RaBitQQueryStateSQ4 *query,
							   const RaBitQCode256 *code);
#endif

#ifdef __AVX2__
/*
 * AVX2 optimized rotation and encoding
 */
void RaBitQEncode_AVX2(RaBitQEncoder *enc, const float *vector, RaBitQCode *code);

/*
 * AVX2 optimized SQ4 distance computation (using lookup tables for popcount)
 */
float RaBitQSQ4Distance_AVX2(const RaBitQQueryStateSQ4 *query,
							 const RaBitQCode256 *code);
#endif

/* ============== Runtime Dispatch ============== */

/*
 * Function pointer type for SQ4 distance computation
 */
typedef float (*RaBitQSQ4DistanceFunc)(const RaBitQQueryStateSQ4 *query,
										const RaBitQCode256 *code);

/*
 * Global function pointer for SQ4 distance (set at module init)
 */
extern RaBitQSQ4DistanceFunc RaBitQSQ4Distance;

/*
 * Initialize SIMD dispatch for SQ4 distance computation
 * Must be called at module initialization
 */
void RaBitQInit(void);

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
 * Compute Hamming distance for 256-bit codes (up to 256 dimensions)
 */
static inline int
RaBitQHamming256(const RaBitQCode256 *a, const RaBitQCode256 *b)
{
	return __builtin_popcountll(a->bits[0] ^ b->bits[0]) +
		   __builtin_popcountll(a->bits[1] ^ b->bits[1]) +
		   __builtin_popcountll(a->bits[2] ^ b->bits[2]) +
		   __builtin_popcountll(a->bits[3] ^ b->bits[3]);
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
