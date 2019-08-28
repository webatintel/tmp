@ECHO OFF
:: Search dxc.exe on current PATH. 
where dxc.exe >nul 2>nul
IF %ERRORLEVEL%==0 (
    SET dxcCmd=dxc.exe
	GOTO COMPILE_SHADER
)

:: Search dxc.exe on 15063 SDK installtion path.
dir "%PROGRAMFILES(x86)%\Windows Kits\10\bin\10.0.15063.0\x86\dxc.exe" >nul 2>nul
IF %ERRORLEVEL%==0 (
	SET dxcCmd="%PROGRAMFILES(x86)%\Windows Kits\10\bin\10.0.15063.0\x86\dxc.exe"
	GOTO COMPILE_SHADER
)

:DXC_NOT_FOUND
ECHO Error: dxc.exe does not exist somewhere on PATH or on %PROGRAMFILES(x86)%\Windows Kits\10\bin\10.0.15063.0\x86\
EXIT /b 1

:COMPILE_SHADER
ECHO DXC Path: %dxcCmd%
ECHO Start compiling shaders...
ECHO ON
%dxcCmd% /D USE_SLM_8X8_4X16 /Zi /E"main" /Vn"g_SLM_8X8_4X16_CS" /Tcs_6_0 /Fh"SLM_8X8_4X16_cs.hlsl.h" /nologo BasicCompute12.hlsl
@IF %ERRORLEVEL% NEQ 0 (EXIT /b %ERRORLEVEL%)
%dxcCmd% /D USE_SIMD_4x1_1x8 /Zi /E"main" /Vn"g_SIMD_4x1_1x8_CS" /Tcs_6_0 /Fh"SIMD_4x1_1x8_cs.hlsl.h" /nologo BasicCompute12.hlsl
@IF %ERRORLEVEL% NEQ 0 (EXIT /b %ERRORLEVEL%)
%dxcCmd% /D USE_SIMD_16x2_1x8 /Zi /E"main" /Vn"g_SIMD_16x2_1x8_CS" /Tcs_6_0 /Fh"SIMD_16x2_1x8_cs.hlsl.h" /nologo BasicCompute12.hlsl
@IF %ERRORLEVEL% NEQ 0 (EXIT /b %ERRORLEVEL%)
%dxcCmd% /D USE_SIMD_8X4_1X8 /Zi /E"main" /Vn"g_SIMD_8X4_1X8_CS" /Tcs_6_0 /Fh"SIMD_8X4_1X8_cs.hlsl.h" /nologo BasicCompute12.hlsl
@IF %ERRORLEVEL% NEQ 0 (EXIT /b %ERRORLEVEL%)
ECHO Done.
