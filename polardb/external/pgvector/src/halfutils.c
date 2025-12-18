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
/* * 200 维 halfvec 极速版 (4路 ILP + 纯 AVX-512)
 * 核心优化：使用 4 个累加器打破 FMA 延迟链，榨干 CPU 流水线
 */
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma")))
float
HalfvecL2SquaredDistance200_Avx512(int dim, half *ax, half *bx)
{
    const uint16_t *a = (const uint16_t *)ax;
    const uint16_t *b = (const uint16_t *)bx;
    
    // 策略：使用4个独立的累加器，完全掩盖 FMA 指令延迟 (Latency Hiding)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    
    // ==================== 主循环：192维 (6个32元素块) ====================
    // 我们手动展开并交错执行，最大化流水线利用率

    // --- Block 0 & 1 (0-63) ---
    {
        // 加载数据
        __m512i ra0 = _mm512_loadu_si512((__m512i *)(a + 0));
        __m512i rb0 = _mm512_loadu_si512((__m512i *)(b + 0));
        __m512i ra1 = _mm512_loadu_si512((__m512i *)(a + 32));
        __m512i rb1 = _mm512_loadu_si512((__m512i *)(b + 32));

        // Block 0 计算 -> 累加到 sum0
        __m512 fa0_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra0));
        __m512 fb0_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb0));
        __m512 diff0_low = _mm512_sub_ps(fa0_low, fb0_low);
        sum0 = _mm512_fmadd_ps(diff0_low, diff0_low, sum0);

        __m512 fa0_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra0, 1));
        __m512 fb0_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb0, 1));
        __m512 diff0_high = _mm512_sub_ps(fa0_high, fb0_high);
        sum0 = _mm512_fmadd_ps(diff0_high, diff0_high, sum0);

        // Block 1 计算 -> 累加到 sum1
        __m512 fa1_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra1));
        __m512 fb1_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb1));
        __m512 diff1_low = _mm512_sub_ps(fa1_low, fb1_low);
        sum1 = _mm512_fmadd_ps(diff1_low, diff1_low, sum1);

        __m512 fa1_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra1, 1));
        __m512 fb1_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb1, 1));
        __m512 diff1_high = _mm512_sub_ps(fa1_high, fb1_high);
        sum1 = _mm512_fmadd_ps(diff1_high, diff1_high, sum1);
    }
    
    // --- Block 2 & 3 (64-127) ---
    {
        __m512i ra2 = _mm512_loadu_si512((__m512i *)(a + 64));
        __m512i rb2 = _mm512_loadu_si512((__m512i *)(b + 64));
        __m512i ra3 = _mm512_loadu_si512((__m512i *)(a + 96));
        __m512i rb3 = _mm512_loadu_si512((__m512i *)(b + 96));

        // Block 2 -> sum2
        __m512 fa2_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra2));
        __m512 fb2_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb2));
        __m512 diff2_low = _mm512_sub_ps(fa2_low, fb2_low);
        sum2 = _mm512_fmadd_ps(diff2_low, diff2_low, sum2);

        __m512 fa2_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra2, 1));
        __m512 fb2_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb2, 1));
        __m512 diff2_high = _mm512_sub_ps(fa2_high, fb2_high);
        sum2 = _mm512_fmadd_ps(diff2_high, diff2_high, sum2);

        // Block 3 -> sum3
        __m512 fa3_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra3));
        __m512 fb3_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb3));
        __m512 diff3_low = _mm512_sub_ps(fa3_low, fb3_low);
        sum3 = _mm512_fmadd_ps(diff3_low, diff3_low, sum3);

        __m512 fa3_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra3, 1));
        __m512 fb3_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb3, 1));
        __m512 diff3_high = _mm512_sub_ps(fa3_high, fb3_high);
        sum3 = _mm512_fmadd_ps(diff3_high, diff3_high, sum3);
    }
    
    // --- Block 4 & 5 (128-191) ---
    {
        __m512i ra4 = _mm512_loadu_si512((__m512i *)(a + 128));
        __m512i rb4 = _mm512_loadu_si512((__m512i *)(b + 128));
        __m512i ra5 = _mm512_loadu_si512((__m512i *)(a + 160));
        __m512i rb5 = _mm512_loadu_si512((__m512i *)(b + 160));

        // Block 4 -> sum0 (复用 sum0)
        __m512 fa4_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra4));
        __m512 fb4_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb4));
        __m512 diff4_low = _mm512_sub_ps(fa4_low, fb4_low);
        sum0 = _mm512_fmadd_ps(diff4_low, diff4_low, sum0);

        __m512 fa4_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra4, 1));
        __m512 fb4_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb4, 1));
        __m512 diff4_high = _mm512_sub_ps(fa4_high, fb4_high);
        sum0 = _mm512_fmadd_ps(diff4_high, diff4_high, sum0);

        // Block 5 -> sum1 (复用 sum1)
        __m512 fa5_low = _mm512_cvtph_ps(_mm512_castsi512_si256(ra5));
        __m512 fb5_low = _mm512_cvtph_ps(_mm512_castsi512_si256(rb5));
        __m512 diff5_low = _mm512_sub_ps(fa5_low, fb5_low);
        sum1 = _mm512_fmadd_ps(diff5_low, diff5_low, sum1);

        __m512 fa5_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(ra5, 1));
        __m512 fb5_high = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(rb5, 1));
        __m512 diff5_high = _mm512_sub_ps(fa5_high, fb5_high);
        sum1 = _mm512_fmadd_ps(diff5_high, diff5_high, sum1);
    }
    
    // ==================== 尾部处理 (192-199) ====================
    // 使用“零扩展”技术，避免 NaN，兼容纯 AVX512F
    __m128i ra_tail_128 = _mm_loadu_si128((__m128i *)(a + 192));
    __m128i rb_tail_128 = _mm_loadu_si128((__m128i *)(b + 192));
    
    // 扩展到 256 位，高位强制为 0
    __m256i ra_tail_256 = _mm256_set_m128i(_mm_setzero_si128(), ra_tail_128);
    __m256i rb_tail_256 = _mm256_set_m128i(_mm_setzero_si128(), rb_tail_128);

    // 转换：结果低8个float有效，高8个float是0.0
    // _mm512_cvtph_ps 需要 AVX512F，PolarDB 肯定支持
    __m512 fa_tail = _mm512_cvtph_ps(ra_tail_256);
    __m512 fb_tail = _mm512_cvtph_ps(rb_tail_256);
    
    __m512 diff_tail = _mm512_sub_ps(fa_tail, fb_tail);
    
    // 累加到 sum2 (高位0.0累加无影响)
    sum2 = _mm512_fmadd_ps(diff_tail, diff_tail, sum2);

    // ==================== 最终归约 ====================
    // 合并4个累加器
    __m512 sum01 = _mm512_add_ps(sum0, sum1);
    __m512 sum23 = _mm512_add_ps(sum2, sum3);
    __m512 sum_total = _mm512_add_ps(sum01, sum23);
    
    return _mm512_reduce_add_ps(sum_total);
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
