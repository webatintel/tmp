import glslangModule from "https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js";

const dimAOuter = 1024;
const dimInner = 1024;
const dimBOuter = 1024;
const localSizeX = 16;
const localSizeY = 16;
const workPerThread = [4, 4];
var device, computePipeline, bindGroup;
var gpuReadBuffer;
var firstMatrix, secondMatrix;
var iteration = 50;
var commandQueue = [];

function recordCommands()
{
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(dimBOuter / (localSizeX * workPerThread[0]) /* x */,
                       dimAOuter / (localSizeY * workPerThread[1])  /* y */);
  passEncoder.endPass();
  commandQueue.push(commandEncoder);
}

function submitQueue() {
  device.defaultQueue.submit(commandQueue.map(enc => enc.finish()));
  commandQueue = [];
}

(async () => {
  if (!navigator.gpu) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  // Uniform Buffer
  const uniformData = new Int32Array([
    dimAOuter /* A rows */,
    dimInner /* A columns */,
    dimInner /* B rows */,
    dimBOuter /* B columns */
  ]);

  const uniformBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM
  });
  new Int32Array(uniformBuffer.getMappedRange()).set(uniformData);
  uniformBuffer.unmap();

  // First Matrix
  firstMatrix = new Float32Array(dimAOuter * dimInner);
  for(var i = 0; i < dimAOuter * dimInner; i++){
    firstMatrix[i] = Math.random();
  }

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(gpuBufferFirstMatrix.getMappedRange()).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  secondMatrix = new Float32Array(dimInner * dimBOuter);
  for(var i = 0; i < dimInner * dimBOuter; i++){
    secondMatrix[i] = Math.random();
  }

  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(gpuBufferSecondMatrix.getMappedRange()).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix
  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (uniformData[0] * uniformData[3]);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "uniform-buffer"
      }
    ]
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer
        }
      },
      {
        binding: 3,
        resource: {
          buffer: uniformBuffer
        }
      }
    ]
  });

  // Compute shader code (GLSL)
  const matmulPackedCode = `#version 450
  layout (local_size_x = ${localSizeX},
    local_size_y = ${localSizeY},
    local_size_z = 1) in;
  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
  } resultMatrix;

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    ivec2 aShape; ivec2 bShape;
  };

  float mm_readA(int row, int col) {
    float result = firstMatrix.numbers[row * aShape[1] + col];
    return result;
  }

  float mm_readB(int row, int col) {
    float result = secondMatrix.numbers[row * bShape[1] + col];
    return result;
  }

  void mm_write(int row, int col, float value) {
    resultMatrix.numbers[row * bShape[1] + col] = value;
  }

  const int RowPerThread = ${workPerThread[1]};
  const int ColPerThread = ${workPerThread[0]};
  const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
  const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
  const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

  shared float mm_Asub[TileAOuter][TileInner];
  shared float mm_Bsub[TileInner][TileBOuter];

  void mm_matMul() {
    int dimAOuter = aShape[0];
    int dimInner = aShape[1];
    int dimBOuter = bShape[1];
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

    int numTiles = (dimInner - 1) / TileInner + 1;

    float acc[RowPerThread][ColPerThread];
    float ACached;
    float BCached[ColPerThread];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
        acc[innerRow][innerCol] = 0.0;
      }
    }

    const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
    int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
    const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
    int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

    // Loop over shared dimension.
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileColA + innerCol;

          mm_Asub[inputRow][inputCol] = mm_readA(
              globalRow + innerRow,
              t * TileInner + inputCol);
        }
      }
      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol + innerCol;

          mm_Bsub[inputRow][inputCol] = mm_readB(
            t * TileInner + inputRow,
            globalCol + innerCol);;
        }
      }

      barrier();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner; k++) {
        for (int inner = 0; inner < ColPerThread; inner++) {
          BCached[inner] = mm_Bsub[k][tileCol + inner];
        }

        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          ACached = mm_Asub[tileRow + innerRow][k];
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            acc[innerRow][innerCol] += ACached * BCached[innerCol];
          }
        }
      }

      barrier();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

        if ((globalCol + innerCol) < dimBOuter &&
            (globalRow + innerRow) < dimAOuter) {
          mm_write(globalRow + innerRow,
                   globalCol + innerCol,
                   acc[innerRow][innerCol]);
        }
      }
    }
  }

    void main() {
      mm_matMul();
    }
  `;

  // Pipeline setup

  const glslang = await glslangModule();

  computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: glslang.compileGLSL(matmulPackedCode, "compute")
      }),
      entryPoint: "main"
    }
  });

  recordCommands();


  // Get a GPU buffer for reading in an unmapped state.
  gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultMatrixBufferSize /* size */
  );
  commandQueue.push(commandEncoder);

  // Submit GPU commands.
  submitQueue();

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

  let acc = 0, m = Math.floor(dimAOuter*Math.random()),  n = Math.floor(dimBOuter*Math.random())
  for(let k=0; k<dimInner; k++) acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];
  console.log(`result[${m}, ${n}] = ${arrayBuffer[m * dimBOuter + n]}, expectedResult = ${acc}`);
  document.getElementById('output').innerText =
    'Finished the warmup. press "Run" button to see the result data.';
  /*
  for (var i = 0; i < dimAOuter; i++)
  for (var j = 0; j< dimBOuter; j++)
  {
    let test = 0;
    for (var k =0; k < dimInner; k++)
    {
      test += firstMatrix[i * dimInner + k] * secondMatrix[k * dimBOuter + j];
    }
    console.log(`result[${i}, ${j}] = ${arrayBuffer[i * dimBOuter + j]}, expectedResult = ${test}`);
  }
  */
  gpuReadBuffer.unmap();
})();

export function handleChange(e)
{
  iteration = parseInt(e.target.value);
}

export async function run(){
  const computeFence = device.defaultQueue.createFence();
  var start = performance.now()
  for (var i = 0; i < iteration; i++)
  {
    recordCommands();
  }
  submitQueue();
  device.defaultQueue.signal(computeFence, 1);
  await computeFence.onCompletion(1);
  var end = performance.now();

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

  let acc = 0, m = Math.floor(dimAOuter*Math.random()),  n = Math.floor(dimBOuter*Math.random())
  for(let k=0; k<dimInner; k++) acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];

  const meanTime = (end - start) / iteration;
  document.getElementById('output').innerText =
    `Mean time = ${(meanTime).toFixed(3)}ms, GFLOPS=${Math.round(2*dimAOuter*dimBOuter*dimInner/meanTime/10000)/100}
     result[${m}, ${n}] = ${arrayBuffer[m * dimBOuter + n]}, expectedResult = ${acc}`;
  gpuReadBuffer.unmap();
}