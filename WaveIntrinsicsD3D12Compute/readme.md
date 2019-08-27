Requirement:
*) OS: Windows 10 Creators Update 15063+.
*) Windows SDK: Install Windows 10 SDK 15063+ to obtain DXC, the new shader compiler that supports SM6. Note that you might need to modify the paths in CompileShader_SM6.bat if you are using a SDK that is newer than 15063. For simplicity we point the dxc.exe path to 15063 SDK folder.

By default, the wave intrinsics(SIMD_8X4_1X8) algorithm will be executed with DispatchCountPerFrame = 20.

To switch to shared memory(SLM_8X8_4X16) algorithm, comment out '#define USE_SIMD_8X4_1X8' in D3D12SM6WaveIntrinsics.cpp. And run again.

To change dispatch count, modify the value of DispatchCountPerFrame in D3D12SM6WaveIntrinsics.h.
