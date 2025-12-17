/* ==========================================
 * 🚀 Mimalloc 强制覆盖 (PolarDB 比赛专用)
 * ========================================== */
#include "mimalloc.h"

// 强制将标准内存函数替换为 mimalloc 版本
#define malloc(n)       mi_malloc(n)
#define free(p)         mi_free(p)
#define realloc(p,n)    mi_realloc(p,n)
#define calloc(n,c)     mi_calloc(n,c)
#define strdup(s)       mi_strdup(s)
/* ========================================== */
#ifndef VECTOR_H
#define VECTOR_H

#define VECTOR_MAX_DIM 16000

#define VECTOR_SIZE(_dim)		(offsetof(Vector, x) + sizeof(float)*(_dim))
#define DatumGetVector(x)		((Vector *) PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x)	DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x)	PG_RETURN_POINTER(x)

typedef struct Vector
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for future use, always zero */
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;

Vector	   *InitVector(int dim);
void		PrintVector(char *msg, Vector * vector);
int			vector_cmp_internal(Vector * a, Vector * b);

/* TODO Move to better place */
#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif

#endif
