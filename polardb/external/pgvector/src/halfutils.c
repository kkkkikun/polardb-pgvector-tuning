#ifdef _MSC_VER
#define TARGET_AVX512
#else
// 开启 AVX512 基础指令、位运算和字节/字扩展
#define TARGET_AVX512 __attribute__((target("avx512f,avx512vl,avx512bw,avx512dq,fma")))
#endif
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
#else
#define TARGET_F16C __attribute__((target("avx,f16c,fma")))
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

/* * 针对 200 维 halfvec 极致优化的 AVX-512 实现
 * 逻辑：200 维 = 32 * 6 + 8
 * 1000 万数据量下，手动展开能极大减少分支预测开销
 */
float
HalfvecL2SquaredDistance200_Avx512(int dim, half *ax, half *bx)
{
    const uint16_t *a = (const uint16_t *) ax;
    const uint16_t *b = (const uint16_t *) bx;
    
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 diff;

    #define PROCESS_32(offset, s_reg) { \
        __m512i ra = _mm512_loadu_si512((__m512i *)(a + (offset))); \
        __m512i rb = _mm512_loadu_si512((__m512i *)(b + (offset))); \
        __m512 fa1 = _mm512_cvtph_ps(_mm512_castsi512_si256(ra)); \
        __m512 fb1 = _mm512_cvtph_ps(_mm512_castsi512_si256(rb)); \
        diff = _mm512_sub_ps(fa1, fb1); \
        s_reg = _mm512_fmadd_ps(diff, diff, s_reg); \
        __m512 fa2 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra, 1)); \
        __m512 fb2 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb, 1)); \
        diff = _mm512_sub_ps(fa2, fb2); \
        s_reg = _mm512_fmadd_ps(diff, diff, s_reg); \
    }

    PROCESS_32(0,   sum0);
    PROCESS_32(32,  sum1);
    PROCESS_32(64,  sum0);
    PROCESS_32(96,  sum1);
    PROCESS_32(128, sum0);
    PROCESS_32(160, sum1);

    sum0 = _mm512_add_ps(sum0, sum1);

    // 【修复点】：使用 _mm256_castsi128_si256 解决类型不匹配
    __mmask16 mask = 0x00FF; 
    __m128i ra_tail = _mm_loadu_si128((__m128i *)(a + 192));
    __m128i rb_tail = _mm_loadu_si128((__m128i *)(b + 192));
    __m512 fa_t = _mm512_cvtph_ps(_mm256_castsi128_si256(ra_tail));
    __m512 fb_t = _mm512_cvtph_ps(_mm256_castsi128_si256(rb_tail));
    
    diff = _mm512_sub_ps(fa_t, fb_t);
    sum0 = _mm512_maskz_fmadd_ps(mask, diff, diff, sum0);

    return _mm512_reduce_add_ps(sum0);
}

// #ifdef HALFVEC_DISPATCH
// TARGET_AVX512 static float
// HalfvecL2SquaredDistanceAVX512(int dim, half * ax, half * bx)
// {
//     int         i = 0;
//     // 使用双累加器，压榨 FMA 流水线
//     __m512      sum1 = _mm512_setzero_ps();
//     __m512      sum2 = _mm512_setzero_ps();

//     // 每次处理 32 个元素 (16 * 2)
//     for (; i <= dim - 32; i += 32)
//     {
//         // 加载 FP16 数据并转换为 FP32
//         __m512  a1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(ax + i)));
//         __m512  b1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(bx + i)));
//         __m512  a2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(ax + i + 16)));
//         __m512  b2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(bx + i + 16)));

//         __m512  diff1 = _mm512_sub_ps(a1, b1);
//         __m512  diff2 = _mm512_sub_ps(a2, b2);

//         sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
//         sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
//     }

//     // 处理剩下的 16 个（如果 dim=200，192 之后剩 8 个，这里跳过）
//     for (; i <= dim - 16; i += 16)
//     {
//         __m512  a = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(ax + i)));
//         __m512  b = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i *)(bx + i)));
//         __m512  diff = _mm512_sub_ps(a, b);
//         sum1 = _mm512_fmadd_ps(diff, diff, sum1);
//     }

//     // 汇总并处理最后的尾部 (针对 dim=200，此处处理最后的 8 个)
//     float distance = _mm512_reduce_add_ps(_mm512_add_ps(sum1, sum2));

//     for (; i < dim; i++)
//     {
//         float diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);
//         distance += diff * diff;
//     }

//     return distance;
// }
// #endif

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
#endif

#ifdef HALFVEC_DISPATCH
#define CPU_FEATURE_FMA     (1 << 12)
#define CPU_FEATURE_OSXSAVE (1 << 27)
#define CPU_FEATURE_AVX     (1 << 28)
#define CPU_FEATURE_F16C    (1 << 29)

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
#endif


// 在文件头部添加 AVX512 特征位定义
#define CPU_FEATURE_AVX512F (1 << 16) // 指向 EBX 的特征位

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
	if (SupportsCpuFeature(CPU_FEATURE_AVX | CPU_FEATURE_F16C | CPU_FEATURE_FMA))
	{
		HalfvecL2SquaredDistance = HalfvecL2SquaredDistanceF16c;
		HalfvecInnerProduct = HalfvecInnerProductF16c;
		HalfvecCosineSimilarity = HalfvecCosineSimilarityF16c;
		/* Does not require FMA, but keep logic simple */
		HalfvecL1Distance = HalfvecL1DistanceF16c;
	}
#endif
}
