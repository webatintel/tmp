/*
*  Note that SM6 is not supported in the integrated shader compiling functionality in Viusal Studio 2017.
*  In the VS project file, we specify the shaders will be built via a custom build script, CompileShader_SM6.bat.
*  Please refer to CompileShader_SM6.bat to see compiling commands. 
*  
*  You may need to modify the SDK paths in CompileShader_SM6.bat to match the installed SDK. By default the path is pointing to 15063 SDK.
*/

float4 intel_sub_group_shuffle(float4 input, uint lane)
{
	return WaveReadLaneAt(input, lane);
}

float2 intel_sub_group_shuffle(float2 input, uint lane)
{
	return WaveReadLaneAt(input, lane);
}

float intel_sub_group_shuffle(float input, uint lane)
{
    return WaveReadLaneAt(input, lane);
}

cbuffer SceneConstantBuffer : register( b0 )
{
    int M;
    int K;
    int N;
    int TILE_K;
}

static uint3 gl_WorkGroupID = uint3(0, 0, 0);
static uint3 gl_LocalInvocationID = uint3(0, 0, 0);

struct CS_INPUT
{
    uint3 dx_WorkGroupID : SV_GroupID;
    uint3 dx_LocalInvocationID : SV_GroupThreadID;
};

void initGLBuiltins(CS_INPUT input)
{
    gl_WorkGroupID = input.dx_WorkGroupID;
    gl_LocalInvocationID = input.dx_LocalInvocationID;
};

#ifdef USE_SLM_8X8_4X16
StructuredBuffer<float4> src0 : register(t0);
StructuredBuffer<float4> src1 : register(t1);
RWStructuredBuffer<float4> dst : register(u0);

static int VEC_SIZE = 4;
static int TILE_M = 32;
static int TILE_N = 128;
static int ROWS_PER_WI = 8;
static int TILE_K0 = 64;

groupshared float4 atile[512];
[numthreads(16, 4, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);
    int local_y = int(gl_LocalInvocationID.y);

	// Result ctile is M rows x N columns
    // M = 32, we have 4 rows of work-items, so we need 32/4 8 results down
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 results across = 2 float4s across

    float4 dot00 = {0, 0, 0, 0};
    float4 dot01 = {0, 0, 0, 0};
    float4 dot02 = {0, 0, 0, 0};
    float4 dot03 = {0, 0, 0, 0};
    float4 dot04 = {0, 0, 0, 0};
    float4 dot05 = {0, 0, 0, 0};
    float4 dot06 = {0, 0, 0, 0};
    float4 dot07 = {0, 0, 0, 0};
    float4 dot10 = {0, 0, 0, 0};
    float4 dot11 = {0, 0, 0, 0};
    float4 dot12 = {0, 0, 0, 0};
    float4 dot13 = {0, 0, 0, 0};
    float4 dot14 = {0, 0, 0, 0};
    float4 dot15 = {0, 0, 0, 0};
    float4 dot16 = {0, 0, 0, 0};
    float4 dot17 = {0, 0, 0, 0};

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width1;

    // Src0 is used to load atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    // K = 64, we'll process four rows at a time
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 floats across = 2 float4s across
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );
    int src1_read1 = src1_read0 + ( TILE_N / 2 / VEC_SIZE );

    int slm = local_y * ( ROWS_PER_WI * TILE_K0 / VEC_SIZE );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do{
      // We want to load atile, which is M rows x K columns
      // M = 32, and we have 4 rows of work-items, so each work-item must load 32/4 = 8 rows.
      // K = 64, and we have 16 columns of work-items, so each work-item must load 64/16 = 4 columns = 1 float4.
      atile[slm + local_x + 0 * TILE_K0 / VEC_SIZE] = src0[src0_read + 0 * width0];
      atile[slm + local_x + 1 * TILE_K0 / VEC_SIZE] = src0[src0_read + 1 * width0];
      atile[slm + local_x + 2 * TILE_K0 / VEC_SIZE] = src0[src0_read + 2 * width0];
      atile[slm + local_x + 3 * TILE_K0 / VEC_SIZE] = src0[src0_read + 3 * width0];
      atile[slm + local_x + 4 * TILE_K0 / VEC_SIZE] = src0[src0_read + 4 * width0];
      atile[slm + local_x + 5 * TILE_K0 / VEC_SIZE] = src0[src0_read + 5 * width0];
      atile[slm + local_x + 6 * TILE_K0 / VEC_SIZE] = src0[src0_read + 6 * width0];
      atile[slm + local_x + 7 * TILE_K0 / VEC_SIZE] = src0[src0_read + 7 * width0];

      src0_read += TILE_K0 / VEC_SIZE;

      GroupMemoryBarrierWithGroupSync();

      int i = 0;
      do{
          // We get better performance by loading btile first.
          float4 brow00 = src1[src1_read0];   src1_read0 += width1;
          float4 brow01 = src1[src1_read0];   src1_read0 += width1;
          float4 brow02 = src1[src1_read0];   src1_read0 += width1;
          float4 brow03 = src1[src1_read0];   src1_read0 += width1;
          float4 brow10 = src1[src1_read1];   src1_read1 += width1;
          float4 brow11 = src1[src1_read1];   src1_read1 += width1;
          float4 brow12 = src1[src1_read1];   src1_read1 += width1;
          float4 brow13 = src1[src1_read1];   src1_read1 += width1;

          float4 a0 = atile[slm + i + 0 * TILE_K0 / VEC_SIZE ];
          dot00 = brow00*a0.x + dot00;
          dot00 = brow01*a0.y + dot00;
          dot00 = brow02*a0.z + dot00;
          dot00 = brow03*a0.w + dot00;
          dot10 = brow10*a0.x + dot10;
          dot10 = brow11*a0.y + dot10;
          dot10 = brow12*a0.z + dot10;
          dot10 = brow13*a0.w + dot10;

          float4 a1 = atile[slm + i + 1 * TILE_K0 / VEC_SIZE ];
          dot01 = brow00*a1.x + dot01;
          dot01 = brow01*a1.y + dot01;
          dot01 = brow02*a1.z + dot01;
          dot01 = brow03*a1.w + dot01;
          dot11 = brow10*a1.x + dot11;
          dot11 = brow11*a1.y + dot11;
          dot11 = brow12*a1.z + dot11;
          dot11 = brow13*a1.w + dot11;

          float4 a2 = atile[slm + i + 2 * TILE_K0 / VEC_SIZE ];
          dot02 = brow00*a2.x + dot02;
          dot02 = brow01*a2.y + dot02;
          dot02 = brow02*a2.z + dot02;
          dot02 = brow03*a2.w + dot02;
          dot12 = brow10*a2.x + dot12;
          dot12 = brow11*a2.y + dot12;
          dot12 = brow12*a2.z + dot12;
          dot12 = brow13*a2.w + dot12;

          float4 a3 = atile[slm + i + 3 * TILE_K0 / VEC_SIZE ];
          dot03 = brow00*a3.x + dot03;
          dot03 = brow01*a3.y + dot03;
          dot03 = brow02*a3.z + dot03;
          dot03 = brow03*a3.w + dot03;
          dot13 = brow10*a3.x + dot13;
          dot13 = brow11*a3.y + dot13;
          dot13 = brow12*a3.z + dot13;
          dot13 = brow13*a3.w + dot13;

          float4 a4 = atile[slm + i + 4 * TILE_K0 / VEC_SIZE ];
          dot04 = brow00*a4.x + dot04;
          dot04 = brow01*a4.y + dot04;
          dot04 = brow02*a4.z + dot04;
          dot04 = brow03*a4.w + dot04;
          dot14 = brow10*a4.x + dot14;
          dot14 = brow11*a4.y + dot14;
          dot14 = brow12*a4.z + dot14;
          dot14 = brow13*a4.w + dot14;

          float4 a5 = atile[slm + i + 5 * TILE_K0 / VEC_SIZE ];
          dot05 = brow00*a5.x + dot05;
          dot05 = brow01*a5.y + dot05;
          dot05 = brow02*a5.z + dot05;
          dot05 = brow03*a5.w + dot05;
          dot15 = brow10*a5.x + dot15;
          dot15 = brow11*a5.y + dot15;
          dot15 = brow12*a5.z + dot15;
          dot15 = brow13*a5.w + dot15;

          float4 a6 = atile[slm + i + 6 * TILE_K0 / VEC_SIZE ];
          dot06 = brow00*a6.x + dot06;
          dot06 = brow01*a6.y + dot06;
          dot06 = brow02*a6.z + dot06;
          dot06 = brow03*a6.w + dot06;
          dot16 = brow10*a6.x + dot16;
          dot16 = brow11*a6.y + dot16;
          dot16 = brow12*a6.z + dot16;
          dot16 = brow13*a6.w + dot16;

          float4 a7 = atile[slm + i + 7 * TILE_K0 / VEC_SIZE ];
          dot07 = brow00*a7.x + dot07;
          dot07 = brow01*a7.y + dot07;
          dot07 = brow02*a7.z + dot07;
          dot07 = brow03*a7.w + dot07;
          dot17 = brow10*a7.x + dot17;
          dot17 = brow11*a7.y + dot17;
          dot17 = brow12*a7.z + dot17;
          dot17 = brow13*a7.w + dot17;

          i++;
      }
      while( i < TILE_K0 / VEC_SIZE );

      GroupMemoryBarrierWithGroupSync();

      w += TILE_K0 / VEC_SIZE;
    }
    while( w < width0 );

    int dst_write1 = dst_write0 + ( TILE_N / 2 / VEC_SIZE );

    dst[dst_write0] = dot00;  dst_write0 += width1;
    dst[dst_write0] = dot01;  dst_write0 += width1;
    dst[dst_write0] = dot02;  dst_write0 += width1;
    dst[dst_write0] = dot03;  dst_write0 += width1;
    dst[dst_write0] = dot04;  dst_write0 += width1;
    dst[dst_write0] = dot05;  dst_write0 += width1;
    dst[dst_write0] = dot06;  dst_write0 += width1;
    dst[dst_write0] = dot07;  dst_write0 += width1;

    dst[dst_write1] = dot10;  dst_write1 += width1;
    dst[dst_write1] = dot11;  dst_write1 += width1;
    dst[dst_write1] = dot12;  dst_write1 += width1;
    dst[dst_write1] = dot13;  dst_write1 += width1;
    dst[dst_write1] = dot14;  dst_write1 += width1;
    dst[dst_write1] = dot15;  dst_write1 += width1;
    dst[dst_write1] = dot16;  dst_write1 += width1;
    dst[dst_write1] = dot17;  dst_write1 += width1;
}
#endif  // USE_SLM_8X8_4X16

#ifdef USE_SIMD_8X4_1X8
StructuredBuffer<float4> src0 : register(t0);
StructuredBuffer<float4> src1 : register(t1);
RWStructuredBuffer<float4> dst : register(u0);

static int VEC_SIZE = 4;
static int TILE_M = 8;
static int TILE_N = 32;

[numthreads(8, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);

    // Result ctile is M rows x N columns
    // M = 8, we have 1 rows of work-items, so we need 8/1 = 8 results down
    // N = 32, we have 8 columns of work-items, so we need 32/8 = 4 results across = 1 float4s across

    float4 dot00 = (float4)(0.f);
    float4 dot01 = (float4)(0.f);
    float4 dot02 = (float4)(0.f);
    float4 dot03 = (float4)(0.f);
    float4 dot04 = (float4)(0.f);
    float4 dot05 = (float4)(0.f);
    float4 dot06 = (float4)(0.f);
    float4 dot07 = (float4)(0.f);

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( group_y * TILE_M ) * width1;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( group_y * TILE_M ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
		float4 arow0 = (float4)(0.f);
		float4 arow1 = (float4)(0.f);
		float4 arow2 = (float4)(0.f);
		float4 arow3 = (float4)(0.f);
		float4 arow4 = (float4)(0.f);
		float4 arow5 = (float4)(0.f);
		float4 arow6 = (float4)(0.f);
		float4 arow7 = (float4)(0.f);
        arow0 = src0[src0_read + 0 * width0 ];
        arow1 = src0[src0_read + 1 * width0 ];
        arow2 = src0[src0_read + 2 * width0 ];
        arow3 = src0[src0_read + 3 * width0 ];
        arow4 = src0[src0_read + 4 * width0 ];
        arow5 = src0[src0_read + 5 * width0 ];
        arow6 = src0[src0_read + 6 * width0 ];
        arow7 = src0[src0_read + 7 * width0 ];
		float4 a0 = (float4)(0.f);
		float4 a1 = (float4)(0.f);
		float4 a2 = (float4)(0.f);
		float4 a3 = (float4)(0.f);
		float4 a4 = (float4)(0.f);
		float4 a5 = (float4)(0.f);
		float4 a6 = (float4)(0.f);
		float4 a7 = (float4)(0.f);

#ifdef PREVENT_LOOP_UNROLLING
        // We use uniform to prevent loop unrolling. This is because loop unrolling will
        // hit a fxc bug which reuslts the shader compilation is really slow.
        for (int _index = 0; _index < TILE_K / VEC_SIZE; _index++)
        {
            a0 = intel_sub_group_shuffle( arow0, _index );
            a1 = intel_sub_group_shuffle( arow1, _index );
            a2 = intel_sub_group_shuffle( arow2, _index );
            a3 = intel_sub_group_shuffle( arow3, _index );
            a4 = intel_sub_group_shuffle( arow4, _index );
            a5 = intel_sub_group_shuffle( arow5, _index );
            a6 = intel_sub_group_shuffle( arow6, _index );
            a7 = intel_sub_group_shuffle( arow7, _index );
            const float4 brow00 = src1[src1_read0];   src1_read0 += width1;
            const float4 brow01 = src1[src1_read0];   src1_read0 += width1;
            const float4 brow02 = src1[src1_read0];   src1_read0 += width1;
            const float4 brow03 = src1[src1_read0];   src1_read0 += width1;
            dot00 = mad(brow00, (float4) a0.x, dot00);
            dot00 = mad(brow01, (float4) a0.y, dot00);
            dot00 = mad(brow02, (float4) a0.z, dot00);
            dot00 = mad(brow03, (float4) a0.w, dot00);
            dot01 = mad(brow00, (float4) a1.x, dot01);
            dot01 = mad(brow01, (float4) a1.y, dot01);
            dot01 = mad(brow02, (float4) a1.z, dot01);
            dot01 = mad(brow03, (float4) a1.w, dot01);
            dot02 = mad(brow00, (float4) a2.x, dot02);
            dot02 = mad(brow01, (float4) a2.y, dot02);
            dot02 = mad(brow02, (float4) a2.z, dot02);
            dot02 = mad(brow03, (float4) a2.w, dot02);
            dot03 = mad(brow00, (float4) a3.x, dot03);
            dot03 = mad(brow01, (float4) a3.y, dot03);
            dot03 = mad(brow02, (float4) a3.z, dot03);
            dot03 = mad(brow03, (float4) a3.w, dot03);
            dot04 = mad(brow00, (float4) a4.x, dot04);
            dot04 = mad(brow01, (float4) a4.y, dot04);
            dot04 = mad(brow02, (float4) a4.z, dot04);
            dot04 = mad(brow03, (float4) a4.w, dot04);
            dot05 = mad(brow00, (float4) a5.x, dot05);
            dot05 = mad(brow01, (float4) a5.y, dot05);
            dot05 = mad(brow02, (float4) a5.z, dot05);
            dot05 = mad(brow03, (float4) a5.w, dot05);
            dot06 = mad(brow00, (float4) a6.x, dot06);
            dot06 = mad(brow01, (float4) a6.y, dot06);
            dot06 = mad(brow02, (float4) a6.z, dot06);
            dot06 = mad(brow03, (float4) a6.w, dot06);
            dot07 = mad(brow00, (float4) a7.x, dot07);
            dot07 = mad(brow01, (float4) a7.y, dot07);
            dot07 = mad(brow02, (float4) a7.z, dot07);
            dot07 = mad(brow03, (float4) a7.w, dot07);
        }
#else
#define ITERATION( _index ) \
        {   \
            a0 = intel_sub_group_shuffle( arow0, _index ); \
            a1 = intel_sub_group_shuffle( arow1, _index ); \
            a2 = intel_sub_group_shuffle( arow2, _index ); \
            a3 = intel_sub_group_shuffle( arow3, _index ); \
            a4 = intel_sub_group_shuffle( arow4, _index ); \
            a5 = intel_sub_group_shuffle( arow5, _index ); \
            a6 = intel_sub_group_shuffle( arow6, _index ); \
            a7 = intel_sub_group_shuffle( arow7, _index ); \
            const float4 brow00 = src1[src1_read0];   src1_read0 += width1;    \
            const float4 brow01 = src1[src1_read0];   src1_read0 += width1;    \
            const float4 brow02 = src1[src1_read0];   src1_read0 += width1;    \
            const float4 brow03 = src1[src1_read0];   src1_read0 += width1;    \
            dot00 = mad(brow00, (float4) a0.x, dot00);  \
            dot00 = mad(brow01, (float4) a0.y, dot00);  \
            dot00 = mad(brow02, (float4) a0.z, dot00);  \
            dot00 = mad(brow03, (float4) a0.w, dot00);  \
            dot01 = mad(brow00, (float4) a1.x, dot01);  \
            dot01 = mad(brow01, (float4) a1.y, dot01);  \
            dot01 = mad(brow02, (float4) a1.z, dot01);  \
            dot01 = mad(brow03, (float4) a1.w, dot01);  \
            dot02 = mad(brow00, (float4) a2.x, dot02);  \
            dot02 = mad(brow01, (float4) a2.y, dot02);  \
            dot02 = mad(brow02, (float4) a2.z, dot02);  \
            dot02 = mad(brow03, (float4) a2.w, dot02);  \
            dot03 = mad(brow00, (float4) a3.x, dot03);  \
            dot03 = mad(brow01, (float4) a3.y, dot03);  \
            dot03 = mad(brow02, (float4) a3.z, dot03);  \
            dot03 = mad(brow03, (float4) a3.w, dot03);  \
            dot04 = mad(brow00, (float4) a4.x, dot04);  \
            dot04 = mad(brow01, (float4) a4.y, dot04);  \
            dot04 = mad(brow02, (float4) a4.z, dot04);  \
            dot04 = mad(brow03, (float4) a4.w, dot04);  \
            dot05 = mad(brow00, (float4) a5.x, dot05);  \
            dot05 = mad(brow01, (float4) a5.y, dot05);  \
            dot05 = mad(brow02, (float4) a5.z, dot05);  \
            dot05 = mad(brow03, (float4) a5.w, dot05);  \
            dot06 = mad(brow00, (float4) a6.x, dot06);  \
            dot06 = mad(brow01, (float4) a6.y, dot06);  \
            dot06 = mad(brow02, (float4) a6.z, dot06);  \
            dot06 = mad(brow03, (float4) a6.w, dot06);  \
            dot07 = mad(brow00, (float4) a7.x, dot07);  \
            dot07 = mad(brow01, (float4) a7.y, dot07);  \
            dot07 = mad(brow02, (float4) a7.z, dot07);  \
            dot07 = mad(brow03, (float4) a7.w, dot07);  \
        }

        // We need K/VEC_SIZE iterations.
        // K = 32, VEC_SIZE = 4
        // So, 32/4 = 8 iterations.
        ITERATION( 0 );
        ITERATION( 1 );
        ITERATION( 2 );
        ITERATION( 3 );
        ITERATION( 4 );
        ITERATION( 5 );
        ITERATION( 6 );
        ITERATION( 7 );
#undef ITERATION
#endif  // PREVENT_LOOP_UNROLLING
        src0_read += TILE_K / VEC_SIZE;
        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    dst[dst_write0] = dot00;  dst_write0 += width1;
    dst[dst_write0] = dot01;  dst_write0 += width1;
    dst[dst_write0] = dot02;  dst_write0 += width1;
    dst[dst_write0] = dot03;  dst_write0 += width1;
    dst[dst_write0] = dot04;  dst_write0 += width1;
    dst[dst_write0] = dot05;  dst_write0 += width1;
    dst[dst_write0] = dot06;  dst_write0 += width1;
    dst[dst_write0] = dot07;  dst_write0 += width1;
}
#endif  // USE_SIMD_8X4_1X8

#ifdef USE_SIMD_16x2_1x8
StructuredBuffer<float2> src0 : register(t0);
StructuredBuffer<float2> src1 : register(t1);
RWStructuredBuffer<float2> dst : register(u0);

static int VEC_SIZE = 2;
static int TILE_M = 16;
static int TILE_N = 16;

[numthreads(8, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);

    // Result ctile is M rows x N columns
    // M = 16, we have 1 row of work-items, so we need 16/1 = 16 results down
    // N = 16, we have 8 columns of work-items, so we need 16/8 = 2 result across

    float2  dot[16];
    for (int i = 0; i < 16; i++)
    {
        dot[i] = 0.f;
    }

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( group_y * TILE_M ) * width1;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( group_y * TILE_M ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        // We want to load atile, which is M rows x K columns
        // M = 16, we have 1 row of work-items, so each work-item must load 16/1 = 16 rows
        // K = 16, we have 8 columns of work-items, so each work-item must load 16/8 = 2 columns
        float2  arow;

        // Now load btile, which is K rows x N columns
        // K = 16, we have 1 row of work-items, so each work-item must load 16/1 = 16 rows
        // N = 16, we have 8 columns of work-items, so each work-item must load 16/8 = 2 columns
        float2  brow0 = src1[src1_read0];  src1_read0 += width1;
        float2  brow1 = src1[src1_read0];  src1_read0 += width1;
        float2  brow2 = src1[src1_read0];  src1_read0 += width1;
        float2  brow3 = src1[src1_read0];  src1_read0 += width1;
        float2  brow4 = src1[src1_read0];  src1_read0 += width1;
        float2  brow5 = src1[src1_read0];  src1_read0 += width1;
        float2  brow6 = src1[src1_read0];  src1_read0 += width1;
        float2  brow7 = src1[src1_read0];  src1_read0 += width1;
        float2  brow8 = src1[src1_read0];  src1_read0 += width1;
        float2  brow9 = src1[src1_read0];  src1_read0 += width1;
        float2  browa = src1[src1_read0];  src1_read0 += width1;
        float2  browb = src1[src1_read0];  src1_read0 += width1;
        float2  browc = src1[src1_read0];  src1_read0 += width1;
        float2  browd = src1[src1_read0];  src1_read0 += width1;
        float2  browe = src1[src1_read0];  src1_read0 += width1;
        float2  browf = src1[src1_read0];  src1_read0 += width1;

#ifdef PREVENT_LOOP_UNROLLING
        for (int i = 0; i < TILE_K; i++)
        {
            arow = src0[src0_read + i * width0 ];
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 0 ).x), brow0, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 0 ).y), brow1, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 1 ).x), brow2, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 1 ).y), brow3, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 2 ).x), brow4, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 2 ).y), brow5, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 3 ).x), brow6, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 3 ).y), brow7, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 4 ).x), brow8, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 4 ).y), brow9, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 5 ).x), browa, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 5 ).y), browb, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 6 ).x), browc, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 6 ).y), browd, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 7 ).x), browe, dot[i] );
            dot[i] = mad( (float2)(intel_sub_group_shuffle( arow, 7 ).y), browf, dot[i] );
        }
#else
#define MM_DOT_PRODUCT( _row, _dot )   \
        arow = src0[src0_read + _row * width0 ];                           \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 0 ).x), brow0, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 0 ).y), brow1, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 1 ).x), brow2, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 1 ).y), brow3, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 2 ).x), brow4, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 2 ).y), brow5, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 3 ).x), brow6, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 3 ).y), brow7, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 4 ).x), brow8, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 4 ).y), brow9, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 5 ).x), browa, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 5 ).y), browb, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 6 ).x), browc, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 6 ).y), browd, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 7 ).x), browe, _dot ); \
        _dot = mad( (float2)(intel_sub_group_shuffle( arow, 7 ).y), browf, _dot );

        MM_DOT_PRODUCT( 0x0, dot[0] );
        MM_DOT_PRODUCT( 0x1, dot[1] );
        MM_DOT_PRODUCT( 0x2, dot[2] );
        MM_DOT_PRODUCT( 0x3, dot[3] );
        MM_DOT_PRODUCT( 0x4, dot[4] );
        MM_DOT_PRODUCT( 0x5, dot[5] );
        MM_DOT_PRODUCT( 0x6, dot[6] );
        MM_DOT_PRODUCT( 0x7, dot[7] );
        MM_DOT_PRODUCT( 0x8, dot[8] );
        MM_DOT_PRODUCT( 0x9, dot[9] );
        MM_DOT_PRODUCT( 0xa, dot[10] );
        MM_DOT_PRODUCT( 0xb, dot[11] );
        MM_DOT_PRODUCT( 0xc, dot[12] );
        MM_DOT_PRODUCT( 0xd, dot[13] );
        MM_DOT_PRODUCT( 0xe, dot[14] );
        MM_DOT_PRODUCT( 0xf, dot[15] );

#undef MM_DOT_PRODUCT
#endif  //PREVENT_LOOP_UNROLLING
        src0_read += TILE_K / VEC_SIZE;
        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    for (int i = 0; i < TILE_K; i++)
    {
        dst[dst_write0] = dot[i];  dst_write0 += width1;
    }
}
#endif  // USE_SIMD_16x2_1x8

#ifdef USE_SIMD_4x1_1x8
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

#define TILE_M          4
#define TILE_K0          8
#define TILE_N          8

[numthreads(8, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K;
    int width1 = N;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);

    // Result ctile is M rows x N columns
    // M = 4, we have 1 row of work-items, so we need 4/1 = 4 results down
    // N = 8, we have 8 columns of work-items, so we need 8/8 = 1 result across

    float   dot00 = 0.f;
    float   dot01 = 0.f;
    float   dot02 = 0.f;
    float   dot03 = 0.f;

    int dst_write0 = local_x + ( group_x * ( TILE_N ) ) + ( group_y * TILE_M ) * width1;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( group_y * TILE_M ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int src1_read0 = local_x + ( group_x * ( TILE_N ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        // We want to load atile, which is M rows x K columns
        // M = 4, we have 1 row of work-items, so each work-item must load 4/1 = 4 rows
        // K = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        float   arow;

        // Now load btile, which is K rows x N columns
        // K = 8, we have 1 row of work-items, so each work-item must load 8/1 = 8 rows
        // N = 8, we have 8 columns of work-items, so each work-item must load 8/8 = 1 column
        float   brow0 = src1[src1_read0];  src1_read0 += width1;
        float   brow1 = src1[src1_read0];  src1_read0 += width1;
        float   brow2 = src1[src1_read0];  src1_read0 += width1;
        float   brow3 = src1[src1_read0];  src1_read0 += width1;
        float   brow4 = src1[src1_read0];  src1_read0 += width1;
        float   brow5 = src1[src1_read0];  src1_read0 += width1;
        float   brow6 = src1[src1_read0];  src1_read0 += width1;
        float   brow7 = src1[src1_read0];  src1_read0 += width1;

        arow = src0[src0_read + 0 * width0 ];
        dot00 = mad(intel_sub_group_shuffle( arow, 0 ), brow0, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 1 ), brow1, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 2 ), brow2, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 3 ), brow3, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 4 ), brow4, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 5 ), brow5, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 6 ), brow6, dot00);
        dot00 = mad(intel_sub_group_shuffle( arow, 7 ), brow7, dot00);

        arow = src0[src0_read + 1 * width0 ];
        dot01 = mad(intel_sub_group_shuffle( arow, 0 ), brow0, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 1 ), brow1, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 2 ), brow2, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 3 ), brow3, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 5 ), brow5, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 4 ), brow4, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 6 ), brow6, dot01);
        dot01 = mad(intel_sub_group_shuffle( arow, 7 ), brow7, dot01);

        arow = src0[src0_read + 2 * width0 ];
        dot02 = mad(intel_sub_group_shuffle( arow, 0 ), brow0, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 1 ), brow1, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 2 ), brow2, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 3 ), brow3, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 4 ), brow4, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 5 ), brow5, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 6 ), brow6, dot02);
        dot02 = mad(intel_sub_group_shuffle( arow, 7 ), brow7, dot02);

        arow = src0[src0_read + 3 * width0 ];
        dot03 = mad(intel_sub_group_shuffle( arow, 0 ), brow0, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 1 ), brow1, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 2 ), brow2, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 3 ), brow3, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 4 ), brow4, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 5 ), brow5, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 6 ), brow6, dot03);
        dot03 = mad(intel_sub_group_shuffle( arow, 7 ), brow7, dot03);

        src0_read += TILE_K0;
        w += TILE_K0;
    }
    while( w < width0 );

    dst[dst_write0] = dot00;  dst_write0 += width1;
    dst[dst_write0] = dot01;  dst_write0 += width1;
    dst[dst_write0] = dot02;  dst_write0 += width1;
    dst[dst_write0] = dot03;  dst_write0 += width1;
}
#endif  // USE_SIMD_4x1_1x8

#ifdef USE_SIMD_16x1_1x16
// 16x1_1x16
StructuredBuffer<float> src0 : register(t0);
StructuredBuffer<float> src1 : register(t1);
RWStructuredBuffer<float> dst : register(u0);

static int VEC_SIZE = 1;
static int TILE_M = 16;
static int TILE_N = 16;
// static int TILE_K = 16;

[numthreads(16, 1, 1)]
void main(CS_INPUT input)
{
    initGLBuiltins(input);
    int width0 = K / VEC_SIZE;
    int width1 = N / VEC_SIZE;

    int group_x = int(gl_WorkGroupID.x);
    int group_y = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);

    // Result ctile is M rows x N columns
    // M = 16, we have 1 row of work-items, so we need 16/1 = 16 results down
    // N = 16, we have 16 columns of work-items, so we need 16/16 = 1 result across

    float  dot[16];
    for (int i = 0; i < 16; i++)
    {
        dot[i] = 0.f;
    }

    int dst_write0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( group_y * TILE_M ) * width1;

    // Src0 is directly used as atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    int src0_read = local_x + ( group_y * TILE_M ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    int src1_read0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        // We want to load atile, which is M rows x K columns
        // M = 16, we have 1 row of work-items, so each work-item must load 16/1 = 16 rows
        // K = 16, we have 16 columns of work-items, so each work-item must load 16/16 = 1 columns
        float  arow;

        // Now load btile, which is K rows x N columns
        // K = 16, we have 1 row of work-items, so each work-item must load 16/1 = 16 rows
        // N = 16, we have 16 columns of work-items, so each work-item must load 16/16 = 1 columns
        float  brow0 = src1[src1_read0];  src1_read0 += width1;
        float  brow1 = src1[src1_read0];  src1_read0 += width1;
        float  brow2 = src1[src1_read0];  src1_read0 += width1;
        float  brow3 = src1[src1_read0];  src1_read0 += width1;
        float  brow4 = src1[src1_read0];  src1_read0 += width1;
        float  brow5 = src1[src1_read0];  src1_read0 += width1;
        float  brow6 = src1[src1_read0];  src1_read0 += width1;
        float  brow7 = src1[src1_read0];  src1_read0 += width1;
        float  brow8 = src1[src1_read0];  src1_read0 += width1;
        float  brow9 = src1[src1_read0];  src1_read0 += width1;
        float  browa = src1[src1_read0];  src1_read0 += width1;
        float  browb = src1[src1_read0];  src1_read0 += width1;
        float  browc = src1[src1_read0];  src1_read0 += width1;
        float  browd = src1[src1_read0];  src1_read0 += width1;
        float  browe = src1[src1_read0];  src1_read0 += width1;
        float  browf = src1[src1_read0];  src1_read0 += width1;

		for (int i = 0; i < 16; i++)
		{
        arow = src0[src0_read + i * width0 ];
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 0 )), brow0, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 1 )), brow1, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 2 )), brow2, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 3 )), brow3, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 4 )), brow4, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 5 )), brow5, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 6 )), brow6, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 7 )), brow7, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 8 )), brow8, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 9 )), brow9, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 10 )), browa, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 11 )), browb, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 12 )), browc, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 13 )), browd, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 14 )), browe, dot[i] );
        dot[i] = mad( (float)(intel_sub_group_shuffle( arow, 15 )), browf, dot[i] );
		}

        src0_read += TILE_K / VEC_SIZE;
        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    dst[dst_write0] = dot[0];  dst_write0 += width1;
    dst[dst_write0] = dot[1];  dst_write0 += width1;
    dst[dst_write0] = dot[2];  dst_write0 += width1;
    dst[dst_write0] = dot[3];  dst_write0 += width1;
    dst[dst_write0] = dot[4];  dst_write0 += width1;
    dst[dst_write0] = dot[5];  dst_write0 += width1;
    dst[dst_write0] = dot[6];  dst_write0 += width1;
    dst[dst_write0] = dot[7];  dst_write0 += width1;
    dst[dst_write0] = dot[8];  dst_write0 += width1;
    dst[dst_write0] = dot[9];  dst_write0 += width1;
    dst[dst_write0] = dot[10];  dst_write0 += width1;
    dst[dst_write0] = dot[11];  dst_write0 += width1;
    dst[dst_write0] = dot[12];  dst_write0 += width1;
    dst[dst_write0] = dot[13];  dst_write0 += width1;
    dst[dst_write0] = dot[14];  dst_write0 += width1;
    dst[dst_write0] = dot[15];  dst_write0 += width1;
}
#endif  // USE_SIMD_16x1_1x16
