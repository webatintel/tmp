Requirement:
*) OS: Windows 10 Creators Update 15063+.
*) Windows SDK: Install Windows 10 SDK 15063+ to obtain DXC, the new shader compiler that supports SM6. Note that you might need to modify the paths in CompileShader_SM6.bat if you are using a SDK that is newer than 15063. For simplicity we point the dxc.exe path to 15063 SDK folder.

By default, the wave intrinsics(SIMD_8X4_1X8) algorithm will be executed with DispatchCountPerFrame = 20.

-h, --help     Show this help text and exit.
-k, --kernel SIMD_8X4_1X8 | SIMD_16x2_1x8 | SIMD_4x1_1x8 | SLM_8X8_4X16
    Determines which algorithm you use for matrix multiplication. By default, SIMD_8X4_1X8 will be run.
--num-dispatch int_value     Determines how many dispatch commands will be executed per command list.
--num-frame int_value        Determines how many command lists will be executed.