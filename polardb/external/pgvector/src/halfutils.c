#include "postgres.h"

#include "halfutils.h"
#include "halfvec.h"

#ifdef HALFVEC_DISPATCH
#include <immintrin.h>

#if defined(USE__GET_CPUID)
#include <cpuid.h>
#else
#include <intrin.h>
#endif

#ifdef _MSC_VER
#define TARGET_F16C
#define TARGET_AVX512
#else
#define TARGET_F16C __attribute__((target("avx,f16c,fma")))
#define TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,f16c,fma")))
#endif
#endif

float		(*HalfvecL2SquaredDistance) (int dim, half * ax, half * bx);
float		(*HalfvecInnerProduct) (int dim, half * ax, half * bx);
double		(*HalfvecCosineSimilarity) (int dim, half * ax, half * bx);
float		(*HalfvecL1Distance) (int dim, half * ax, half * bx);

static float
HalfvecL2SquaredDistanceDefault(int dim, half * ax, half * bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);

		distance += diff * diff;
	}

	return distance;
}

#ifdef HALFVEC_DISPATCH
TARGET_F16C static float
HalfvecL2SquaredDistanceF16c(int dim, half * ax, half * bx)
{
	float		distance;
	int			i;
	float		s[8];
	int			count = (dim / 8) * 8;
	__m256		dist = _mm256_setzero_ps();

	for (i = 0; i < count; i += 8)
	{
		__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
		__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
		__m256		axs = _mm256_cvtph_ps(axi);
		__m256		bxs = _mm256_cvtph_ps(bxi);
		__m256		diff = _mm256_sub_ps(axs, bxs);

		dist = _mm256_fmadd_ps(diff, diff, dist);
	}

	_mm256_storeu_ps(s, dist);

	distance = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	for (; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);

		distance += diff * diff;
	}

	return distance;
}

/*
 * AVX-512 optimized L2 squared distance for halfvec
 * Processes 16 half values per iteration using 512-bit registers
 */
TARGET_AVX512 static float
HalfvecL2SquaredDistanceAVX512(int dim, half * ax, half * bx)
{
	int			i = 0;
	int			count16 = (dim / 16) * 16;
	__m512		dist = _mm512_setzero_ps();
	float		distance;
	int			count8;
	__m256		dist256;
	float		s[8];

	/* Prefetch next cache lines */
	_mm_prefetch((const char *)(ax + 64), _MM_HINT_T0);
	_mm_prefetch((const char *)(bx + 64), _MM_HINT_T0);

	/* Process 16 half values at a time */
	for (; i < count16; i += 16)
	{
		/* Load 16 half values (256 bits) and convert to 16 floats (512 bits) */
		__m256i		axi = _mm256_loadu_si256((__m256i *) (ax + i));
		__m256i		bxi = _mm256_loadu_si256((__m256i *) (bx + i));
		__m512		axs = _mm512_cvtph_ps(axi);
		__m512		bxs = _mm512_cvtph_ps(bxi);
		__m512		diff = _mm512_sub_ps(axs, bxs);

		/* Prefetch ahead */
		if (i + 64 < dim)
		{
			_mm_prefetch((const char *)(ax + i + 64), _MM_HINT_T0);
			_mm_prefetch((const char *)(bx + i + 64), _MM_HINT_T0);
		}

		/* FMA: dist = dist + diff * diff */
		dist = _mm512_fmadd_ps(diff, diff, dist);
	}

	/* Reduce 512-bit register to scalar */
	distance = _mm512_reduce_add_ps(dist);

	/* Handle remaining elements with AVX path */
	if (i < dim)
	{
		count8 = ((dim - i) / 8) * 8 + i;
		dist256 = _mm256_setzero_ps();

		for (; i < count8; i += 8)
		{
			__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
			__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
			__m256		axs = _mm256_cvtph_ps(axi);
			__m256		bxs = _mm256_cvtph_ps(bxi);
			__m256		diff = _mm256_sub_ps(axs, bxs);

			dist256 = _mm256_fmadd_ps(diff, diff, dist256);
		}

		_mm256_storeu_ps(s, dist256);
		distance += s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

		/* Handle remaining elements */
		for (; i < dim; i++)
		{
			float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);
			distance += diff * diff;
		}
	}

	return distance;
}
#endif

static float
HalfvecInnerProductDefault(int dim, half * ax, half * bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += HalfToFloat4(ax[i]) * HalfToFloat4(bx[i]);

	return distance;
}

#ifdef HALFVEC_DISPATCH
TARGET_F16C static float
HalfvecInnerProductF16c(int dim, half * ax, half * bx)
{
	float		distance;
	int			i;
	float		s[8];
	int			count = (dim / 8) * 8;
	__m256		dist = _mm256_setzero_ps();

	for (i = 0; i < count; i += 8)
	{
		__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
		__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
		__m256		axs = _mm256_cvtph_ps(axi);
		__m256		bxs = _mm256_cvtph_ps(bxi);

		dist = _mm256_fmadd_ps(axs, bxs, dist);
	}

	_mm256_storeu_ps(s, dist);

	distance = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	for (; i < dim; i++)
		distance += HalfToFloat4(ax[i]) * HalfToFloat4(bx[i]);

	return distance;
}

/*
 * AVX-512 optimized inner product for halfvec
 */
TARGET_AVX512 static float
HalfvecInnerProductAVX512(int dim, half * ax, half * bx)
{
	int			i = 0;
	int			count16 = (dim / 16) * 16;
	__m512		sum = _mm512_setzero_ps();
	float		result;
	int			count8;
	__m256		sum256;
	float		s[8];

	/* Prefetch next cache lines */
	_mm_prefetch((const char *)(ax + 64), _MM_HINT_T0);
	_mm_prefetch((const char *)(bx + 64), _MM_HINT_T0);

	/* Process 16 half values at a time */
	for (; i < count16; i += 16)
	{
		__m256i		axi = _mm256_loadu_si256((__m256i *) (ax + i));
		__m256i		bxi = _mm256_loadu_si256((__m256i *) (bx + i));
		__m512		axs = _mm512_cvtph_ps(axi);
		__m512		bxs = _mm512_cvtph_ps(bxi);

		/* Prefetch ahead */
		if (i + 64 < dim)
		{
			_mm_prefetch((const char *)(ax + i + 64), _MM_HINT_T0);
			_mm_prefetch((const char *)(bx + i + 64), _MM_HINT_T0);
		}

		/* FMA: sum = sum + a * b */
		sum = _mm512_fmadd_ps(axs, bxs, sum);
	}

	/* Reduce 512-bit register to scalar */
	result = _mm512_reduce_add_ps(sum);

	/* Handle remaining elements */
	if (i < dim)
	{
		count8 = ((dim - i) / 8) * 8 + i;
		sum256 = _mm256_setzero_ps();

		for (; i < count8; i += 8)
		{
			__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
			__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
			__m256		axs = _mm256_cvtph_ps(axi);
			__m256		bxs = _mm256_cvtph_ps(bxi);

			sum256 = _mm256_fmadd_ps(axs, bxs, sum256);
		}

		_mm256_storeu_ps(s, sum256);
		result += s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

		for (; i < dim; i++)
			result += HalfToFloat4(ax[i]) * HalfToFloat4(bx[i]);
	}

	return result;
}
#endif

static double
HalfvecCosineSimilarityDefault(int dim, half * ax, half * bx)
{
	float		similarity = 0.0;
	float		norma = 0.0;
	float		normb = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		axi = HalfToFloat4(ax[i]);
		float		bxi = HalfToFloat4(bx[i]);

		similarity += axi * bxi;
		norma += axi * axi;
		normb += bxi * bxi;
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

#ifdef HALFVEC_DISPATCH
TARGET_F16C static double
HalfvecCosineSimilarityF16c(int dim, half * ax, half * bx)
{
	float		similarity;
	float		norma;
	float		normb;
	int			i;
	float		s[8];
	int			count = (dim / 8) * 8;
	__m256		sim = _mm256_setzero_ps();
	__m256		na = _mm256_setzero_ps();
	__m256		nb = _mm256_setzero_ps();

	for (i = 0; i < count; i += 8)
	{
		__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
		__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
		__m256		axs = _mm256_cvtph_ps(axi);
		__m256		bxs = _mm256_cvtph_ps(bxi);

		sim = _mm256_fmadd_ps(axs, bxs, sim);
		na = _mm256_fmadd_ps(axs, axs, na);
		nb = _mm256_fmadd_ps(bxs, bxs, nb);
	}

	_mm256_storeu_ps(s, sim);
	similarity = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	_mm256_storeu_ps(s, na);
	norma = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	_mm256_storeu_ps(s, nb);
	normb = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	/* Auto-vectorized */
	for (; i < dim; i++)
	{
		float		axi = HalfToFloat4(ax[i]);
		float		bxi = HalfToFloat4(bx[i]);

		similarity += axi * bxi;
		norma += axi * axi;
		normb += bxi * bxi;
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

/*
 * AVX-512 optimized cosine similarity for halfvec
 */
TARGET_AVX512 static double
HalfvecCosineSimilarityAVX512(int dim, half * ax, half * bx)
{
	int			i = 0;
	int			count16 = (dim / 16) * 16;
	__m512		sim = _mm512_setzero_ps();
	__m512		na = _mm512_setzero_ps();
	__m512		nb = _mm512_setzero_ps();
	float		similarity;
	float		norma;
	float		normb;

	/* Prefetch next cache lines */
	_mm_prefetch((const char *)(ax + 64), _MM_HINT_T0);
	_mm_prefetch((const char *)(bx + 64), _MM_HINT_T0);

	/* Process 16 half values at a time */
	for (; i < count16; i += 16)
	{
		__m256i		axi = _mm256_loadu_si256((__m256i *) (ax + i));
		__m256i		bxi = _mm256_loadu_si256((__m256i *) (bx + i));
		__m512		axs = _mm512_cvtph_ps(axi);
		__m512		bxs = _mm512_cvtph_ps(bxi);

		/* Prefetch ahead */
		if (i + 64 < dim)
		{
			_mm_prefetch((const char *)(ax + i + 64), _MM_HINT_T0);
			_mm_prefetch((const char *)(bx + i + 64), _MM_HINT_T0);
		}

		sim = _mm512_fmadd_ps(axs, bxs, sim);
		na = _mm512_fmadd_ps(axs, axs, na);
		nb = _mm512_fmadd_ps(bxs, bxs, nb);
	}

	/* Reduce 512-bit registers to scalars */
	similarity = _mm512_reduce_add_ps(sim);
	norma = _mm512_reduce_add_ps(na);
	normb = _mm512_reduce_add_ps(nb);

	/* Handle remaining elements */
	for (; i < dim; i++)
	{
		float		axi = HalfToFloat4(ax[i]);
		float		bxi = HalfToFloat4(bx[i]);

		similarity += axi * bxi;
		norma += axi * axi;
		normb += bxi * bxi;
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}
#endif

static float
HalfvecL1DistanceDefault(int dim, half * ax, half * bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += fabsf(HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]));

	return distance;
}

#ifdef HALFVEC_DISPATCH
/* Does not require FMA, but keep logic simple */
TARGET_F16C static float
HalfvecL1DistanceF16c(int dim, half * ax, half * bx)
{
	float		distance;
	int			i;
	float		s[8];
	int			count = (dim / 8) * 8;
	__m256		dist = _mm256_setzero_ps();
	__m256		sign = _mm256_set1_ps(-0.0);

	for (i = 0; i < count; i += 8)
	{
		__m128i		axi = _mm_loadu_si128((__m128i *) (ax + i));
		__m128i		bxi = _mm_loadu_si128((__m128i *) (bx + i));
		__m256		axs = _mm256_cvtph_ps(axi);
		__m256		bxs = _mm256_cvtph_ps(bxi);

		dist = _mm256_add_ps(dist, _mm256_andnot_ps(sign, _mm256_sub_ps(axs, bxs)));
	}

	_mm256_storeu_ps(s, dist);

	distance = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];

	for (; i < dim; i++)
		distance += fabsf(HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]));

	return distance;
}

/*
 * AVX-512 optimized L1 distance for halfvec
 */
TARGET_AVX512 static float
HalfvecL1DistanceAVX512(int dim, half * ax, half * bx)
{
	int			i = 0;
	int			count16 = (dim / 16) * 16;
	__m512		dist = _mm512_setzero_ps();
	__m512		sign = _mm512_set1_ps(-0.0f);
	float		distance;

	/* Prefetch next cache lines */
	_mm_prefetch((const char *)(ax + 64), _MM_HINT_T0);
	_mm_prefetch((const char *)(bx + 64), _MM_HINT_T0);

	/* Process 16 half values at a time */
	for (; i < count16; i += 16)
	{
		__m256i		axi = _mm256_loadu_si256((__m256i *) (ax + i));
		__m256i		bxi = _mm256_loadu_si256((__m256i *) (bx + i));
		__m512		axs = _mm512_cvtph_ps(axi);
		__m512		bxs = _mm512_cvtph_ps(bxi);
		__m512		diff = _mm512_sub_ps(axs, bxs);
		/* Absolute value using AND NOT with sign mask */
		__m512		abs_diff = _mm512_andnot_ps(sign, diff);

		/* Prefetch ahead */
		if (i + 64 < dim)
		{
			_mm_prefetch((const char *)(ax + i + 64), _MM_HINT_T0);
			_mm_prefetch((const char *)(bx + i + 64), _MM_HINT_T0);
		}

		dist = _mm512_add_ps(dist, abs_diff);
	}

	/* Reduce 512-bit register to scalar */
	distance = _mm512_reduce_add_ps(dist);

	/* Handle remaining elements */
	for (; i < dim; i++)
		distance += fabsf(HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]));

	return distance;
}
#endif

#ifdef HALFVEC_DISPATCH
#define CPU_FEATURE_FMA     (1 << 12)
#define CPU_FEATURE_OSXSAVE (1 << 27)
#define CPU_FEATURE_AVX     (1 << 28)
#define CPU_FEATURE_F16C    (1 << 29)

/* AVX-512 feature bits in EBX from CPUID leaf 7 */
#define CPU_FEATURE_AVX512F  (1 << 16)
#define CPU_FEATURE_AVX512DQ (1 << 17)
#define CPU_FEATURE_AVX512BW (1 << 30)
#define CPU_FEATURE_AVX512VL (1 << 31)

#ifdef _MSC_VER
#define TARGET_XSAVE
#else
#define TARGET_XSAVE __attribute__((target("xsave")))
#endif

TARGET_XSAVE static bool
SupportsCpuFeature(unsigned int feature)
{
	unsigned int exx[4] = {0, 0, 0, 0};

#if defined(USE__GET_CPUID)
	__get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuid(exx, 1);
#endif

	/* Check OS supports XSAVE */
	if ((exx[2] & CPU_FEATURE_OSXSAVE) != CPU_FEATURE_OSXSAVE)
		return false;

	/* Check XMM and YMM registers are enabled */
	if ((_xgetbv(0) & 6) != 6)
		return false;

	/* Now check features */
	return (exx[2] & feature) == feature;
}

TARGET_XSAVE static bool
SupportsAVX512(void)
{
	unsigned int exx[4] = {0, 0, 0, 0};
	unsigned int avx512_features;

#if defined(USE__GET_CPUID)
	__get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuid(exx, 1);
#endif

	/* Check OS supports XSAVE */
	if ((exx[2] & CPU_FEATURE_OSXSAVE) != CPU_FEATURE_OSXSAVE)
		return false;

	/* Check XMM, YMM, and ZMM registers are enabled (bits 1, 2, 5, 6, 7) */
	/* 0xE6 = 0b11100110 */
	if ((_xgetbv(0) & 0xE6) != 0xE6)
		return false;

	/* Check AVX-512 features from CPUID leaf 7 */
#if defined(USE__GET_CPUID)
	__get_cpuid_count(7, 0, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
	__cpuidex((int *)exx, 7, 0);
#endif

	/* Check for AVX512F, AVX512DQ, AVX512BW, AVX512VL */
	avx512_features = CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512DQ | 
	                  CPU_FEATURE_AVX512BW | CPU_FEATURE_AVX512VL;
	return (exx[1] & avx512_features) == avx512_features;
}
#endif

void
HalfvecInit(void)
{
	/*
	 * Could skip pointer when single function, but no difference in
	 * performance
	 */
	HalfvecL2SquaredDistance = HalfvecL2SquaredDistanceDefault;
	HalfvecInnerProduct = HalfvecInnerProductDefault;
	HalfvecCosineSimilarity = HalfvecCosineSimilarityDefault;
	HalfvecL1Distance = HalfvecL1DistanceDefault;

#ifdef HALFVEC_DISPATCH
	/* Check for AVX-512 first (best performance) */
	if (SupportsAVX512() && SupportsCpuFeature(CPU_FEATURE_F16C | CPU_FEATURE_FMA))
	{
		HalfvecL2SquaredDistance = HalfvecL2SquaredDistanceAVX512;
		HalfvecInnerProduct = HalfvecInnerProductAVX512;
		HalfvecCosineSimilarity = HalfvecCosineSimilarityAVX512;
		HalfvecL1Distance = HalfvecL1DistanceAVX512;
	}
	/* Fall back to AVX+F16C if AVX-512 not available */
	else if (SupportsCpuFeature(CPU_FEATURE_AVX | CPU_FEATURE_F16C | CPU_FEATURE_FMA))
	{
		HalfvecL2SquaredDistance = HalfvecL2SquaredDistanceF16c;
		HalfvecInnerProduct = HalfvecInnerProductF16c;
		HalfvecCosineSimilarity = HalfvecCosineSimilarityF16c;
		/* Does not require FMA, but keep logic simple */
		HalfvecL1Distance = HalfvecL1DistanceF16c;
	}
#endif
}
