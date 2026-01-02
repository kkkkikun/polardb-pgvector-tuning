/*
 * Residual Quantization (RQ) for pgvector
 *
 * Residual Quantization compresses vectors by:
 * 1. Finding the nearest centroid in codebook 1
 * 2. Computing the residual (difference)
 * 3. Finding the nearest centroid for the residual in codebook 2
 * 4. Repeating for M stages
 *
 * Final representation: M uint8 codes (M bytes per vector)
 * Reconstruction: sum of M centroids
 */

#ifndef RQ_H
#define RQ_H

#include "postgres.h"
#include "halfvec.h"

/* RQ configuration */
#define RQ_NUM_STAGES 8          /* Number of quantization stages (M) */
#define RQ_NUM_CENTROIDS 256     /* Number of centroids per stage (K) */
#define RQ_MAX_DIM 2048          /* Maximum vector dimension */
#define RQ_KMEANS_ITERS 20       /* K-means iterations for training */
#define RQ_SAMPLE_SIZE 50000     /* Training sample size */

/* RQ code type - stores M centroid indices */
typedef struct RQCode
{
	uint8		codes[RQ_NUM_STAGES];
}			RQCode;

/* RQ Codebook - stores all centroids for all stages */
typedef struct RQCodebook
{
	int			dim;			/* Vector dimension */
	int			numStages;		/* Number of stages (M) */
	int			numCentroids;	/* Centroids per stage (K) */
	float	   *centroids;		/* [numStages][numCentroids][dim] */
}			RQCodebook;

/* Distance lookup table for fast asymmetric distance computation */
typedef struct RQDistTable
{
	int			dim;
	int			numStages;
	int			numCentroids;
	float	   *tables;			/* [numStages][numCentroids] - precomputed partial distances */
}			RQDistTable;

/* Training data structure */
typedef struct RQTrainer
{
	int			dim;
	int			numVectors;
	float	   *vectors;		/* Training vectors [numVectors][dim] */
	float	   *residuals;		/* Current residuals [numVectors][dim] */
}			RQTrainer;

/* Function declarations */

/* Codebook management */
RQCodebook *RQCreateCodebook(int dim, int numStages, int numCentroids);
void		RQFreeCodebook(RQCodebook *cb);
float	   *RQGetCentroid(RQCodebook *cb, int stage, int centroidIdx);

/* Training */
void		RQTrainCodebook(RQCodebook *cb, float *vectors, int numVectors);
void		RQKMeansIteration(float *data, int n, int dim, float *centroids, int k, int *assignments);

/* Encoding and decoding */
void		RQEncode(RQCodebook *cb, float *vector, RQCode *code);
void		RQDecode(RQCodebook *cb, RQCode *code, float *output);
void		RQEncodeHalfvec(RQCodebook *cb, HalfVector *vec, RQCode *code);
void		RQDecodeToHalfvec(RQCodebook *cb, RQCode *code, HalfVector *output);

/* Distance computation */
RQDistTable *RQCreateDistTable(RQCodebook *cb, float *query);
void		RQFreeDistTable(RQDistTable *dt);
float		RQComputeDistance(RQDistTable *dt, RQCode *code);
float		RQComputeDistanceL2(RQCodebook *cb, float *query, RQCode *code);

/* Batch operations for SIMD optimization */
void		RQEncodeBatch(RQCodebook *cb, float *vectors, int numVectors, RQCode *codes);
void		RQComputeDistanceBatch(RQDistTable *dt, RQCode *codes, int numCodes, float *distances);

/* Serialization for storage */
Size		RQCodebookSize(RQCodebook *cb);
void		RQSerializeCodebook(RQCodebook *cb, char *buffer);
RQCodebook *RQDeserializeCodebook(char *buffer);

/* Halfvec conversion utilities */
void		HalfvecToFloat(HalfVector *hv, float *output);
void		FloatToHalfvec(float *input, int dim, HalfVector *output);

/* SIMD-optimized distance functions */
#ifdef __AVX512F__
float		RQComputeDistanceAVX512(RQDistTable *dt, RQCode *code);
void		RQComputeDistanceBatchAVX512(RQDistTable *dt, RQCode *codes, int numCodes, float *distances);
#endif

#ifdef __AVX2__
float		RQComputeDistanceAVX2(RQDistTable *dt, RQCode *code);
#endif

#endif							/* RQ_H */
