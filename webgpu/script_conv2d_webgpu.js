import glslangModule from "https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js";


/*
  it('x=[1,4,4,1] f=[1,1,1,3] s=2 d=1 p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const stride: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);

    expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]);
  });
*/

//ivec4 xShape; ivec4 wShape; ivec4 outShape; ivec2 filterDims, pad, stride;

const arrayProduct = (arr) => {
  let product = 1;
  for (let i = 0; i < arr.length; i++) {
    product *= arr[i];
  }
  return product;
};

function computeDispatch(
  layout, outputShape,
  workGroupSize = [1, 1, 1],
  elementsPerThread =
      [1, 1, 1]) {
return [
  Math.ceil(
      arrayProduct(layout.x.map(d => outputShape[d])) /
      (workGroupSize[0] * elementsPerThread[0])),
  layout.y ? Math.ceil(
                 arrayProduct(layout.y.map(d => outputShape[d])) /
                 (workGroupSize[1] * elementsPerThread[1])) :
             1,
  layout.z ? Math.ceil(
                 arrayProduct(layout.z.map(d => outputShape[d])) /
                 (workGroupSize[2] * elementsPerThread[2])) :
             1
];
}

const inChannels = 1;
const filterHeight = 1;
const filterWidth = 1;
const strideHeight = 2;
const strideWidth = 2;
const dilationHeight = 1;
const dilationWidth = 1;
const pad = [0, 0];


const xShape = [1, 4, 4, inChannels];
const wShape = [filterHeight, filterWidth, inChannels, 3];
const outputShape = [1, 2, 2, 3]; // ouputShape.length must be 4

const localSizeX = 16;
const localSizeY = 16;
const workPerThread = [2, 2];

const dispatchLayout = {x: [3], y: [1, 2], z: [0]};

const dispatch = computeDispatch(
  dispatchLayout, outputShape, [localSizeX, localSizeY, 1],
  [workPerThread[0], workPerThread[1], 1]);

var device;
var gpuReadBuffer;
var firstMatrix, secondMatrix;
var gpuCommands;

(async () => {
  if (!navigator.gpu) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  let dimUniforms = [];
  const bufferShapes = [xShape, wShape, outputShape];
    let currentOffset = 0;
    bufferShapes.forEach((d, i) => {
      // Uniforms.
      if (d.length === 0) {
        d = [1];
      }
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment;
      switch (d.length) {
        case 0:
          baseAlignment = 1;
          break;
        case 1:
          baseAlignment = 1;
          break;
        case 2:
          baseAlignment = 2;
          break;
        case 3:
          baseAlignment = 4;
          break;
        case 4:
          baseAlignment = 4;
          break;
        default:
          console.log('unsupported shape');
      }

      const padding =
          Math.ceil(currentOffset / baseAlignment) * baseAlignment -
          currentOffset;
      for (let p = 0; p < padding; ++p) {
        dimUniforms.push(0);
      }
      dimUniforms.push(...d);
      currentOffset += d.length + padding;
    });

    const dimensions = [
      filterHeight, filterWidth, ...pad,
      strideHeight, strideWidth, dilationHeight,
      dilationWidth
    ];

    dimUniforms = dimUniforms.concat(dimensions);

  // Uniform Buffer
  const uniformData = new Int32Array(dimUniforms);

  const [uniformBuffer, arrayBufferData] = device.createBufferMapped({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM
  });
  new Int32Array(arrayBufferData).set(uniformData);
  uniformBuffer.unmap();

  // First Matrix
  const xSize = arrayProduct(xShape);
  firstMatrix = new Float32Array(xSize);
  for(var i = 0; i < xSize; i++){
    firstMatrix[i] = Math.random();
  }
/*
  firstMatrix[0] = 10;
  firstMatrix[1] = 30;
  firstMatrix[2] = 50;
  firstMatrix[3] = 70;
  firstMatrix[4] = 20;
  firstMatrix[5] = 40;
  firstMatrix[6] = 60;
  firstMatrix[7] = 80; 
  firstMatrix[8] = -10;
  firstMatrix[9] = -30;
  firstMatrix[10] = -50;
  firstMatrix[11] = -70; 
  firstMatrix[12] = -20;
  firstMatrix[13] = -40;
  firstMatrix[14] = -60;
  firstMatrix[15] = -80; 
*/
  const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  const wSize = arrayProduct(wShape);
  secondMatrix = new Float32Array(wSize);
  for(var i = 0; i < wSize; i++){
    secondMatrix[i] = Math.random();
  }
 /*
  secondMatrix[0] = 1;
  secondMatrix[1] = 0.5;
  secondMatrix[2] = 1;
  */

  const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix
  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (arrayProduct(outputShape));
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
        type: "storage-buffer"
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "uniform-buffer"
      }
    ]
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: resultMatrixBuffer
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gpuBufferSecondMatrix
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

const sampleA =
  `coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0`;
const sampleB =
  `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
  W[row * dimBOuter + col] : 0`;
  
  // Compute shader code (GLSL)
  const matmulPackedCode = `#version 450
  layout (local_size_x = ${localSizeX},
    local_size_y = ${localSizeY},
    local_size_z = 1) in;

    // Checks whether coordinates lie within the bounds of the shape.
    bool coordsInBounds(ivec4 coord, ivec4 shape) {
      return all(greaterThanEqual(coord, ivec4(0))) &&
          all(lessThan(coord, shape));
    }
  
    bool coordsInBounds(ivec2 coord, ivec2 shape) {
      return all(greaterThanEqual(coord, ivec2(0))) &&
          all(lessThan(coord, shape));
    }

    int getFlatIndex(int coord, int shape) {
      return coord;
    }
  
    int getFlatIndex(ivec2 coords, ivec2 shape) {
      return int(dot(coords, ivec2(shape.y, 1.)));
    }
  
    int getFlatIndex(ivec3 coords, ivec3 shape) {
      return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
    }
  
    int getFlatIndex(ivec4 coords, ivec4 shape) {
      return int(dot(coords, ivec4(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
    }

    layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
      float result[];
    };


      layout(std430, set = 0, binding = 1) readonly buffer ssbx {
        float x[];
      };


      layout(std430, set = 0, binding = 2) readonly buffer ssbW {
        float W[];
      };


    layout(std140, set = 0, binding = 3) uniform Uniforms {
      ivec4 xShape; ivec4 wShape; ivec4 outShape; ivec2 filterDims, pad, stride, dilation;
    };

    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * xShape[3];

  float mm_readA(int row, int col) {
    int r = int(row), c = int(col);
    int outRow = r / outShape[2];
    int outCol = r % outShape[2];

    int WRow = c / (filterDims[1] * xShape[3]);
    int WCol = (c / xShape[3]) % filterDims[1];

    ivec4 coord = ivec4(
        batch,
        outRow * stride[0] + dilation[0] * WRow - pad[0],
        outCol * stride[1] + dilation[1] * WCol - pad[1],
        c % xShape[3]);
    return ${sampleA};
  }

  float mm_readB(int row, int col) {
    return ${sampleB};
  }

  void mm_write(int row, int col, float value) {
    ivec4 outCoord = ivec4(
        batch,
        row / outShape[2],
        row % outShape[2],
        col);
    result[getFlatIndex(outCoord, outShape)] = value;
  }

  const int RowPerThread = ${workPerThread[1]};
  const int ColPerThread = ${workPerThread[0]};
  const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
  const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
  const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

  shared float mm_Asub[TileAOuter][TileInner];
  shared float mm_Bsub[TileInner][TileBOuter];

  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
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
    batch = int(gl_GlobalInvocationID.z);

    mm_matMul(dimAOuter, dimInner, dimBOuter);
  }
  `;

  // Pipeline setup

  const glslang = await glslangModule();

  const computePipeline = device.createComputePipeline({
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

  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(dispatch[0], dispatch[1], dispatch[2]);
  passEncoder.endPass();

  // Get a GPU buffer for reading in an unmapped state.
  gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultMatrixBufferSize /* size */
  );

  // Submit GPU commands.
  gpuCommands = commandEncoder.finish();
  device.defaultQueue.submit([gpuCommands]);

  // Read buffer.
  const arrayBuffer = new Float32Array(await gpuReadBuffer.mapReadAsync());

  console.log(arrayBuffer);
  document.getElementById('output').innerText =
    'Finished the warmup. press "MatMul" button to see the result data.';

  gpuReadBuffer.unmap();
})();

export async function run(){
  const computeFence = device.defaultQueue.createFence();

  var start = performance.now()
  device.defaultQueue.submit([gpuCommands]);
  device.defaultQueue.signal(computeFence, 1);
  await computeFence.onCompletion(1);
  var end = performance.now();

  document.getElementById('output').innerText =
    `time = ${(end - start).toFixed(3)}ms`;
  gpuReadBuffer.unmap();
}