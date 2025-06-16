@echo off
setlocal enabledelayedexpansion

REM Default to Release if no configuration specified
set BUILD_CONFIG=Release
set BUILD_ALL=0

REM Parse command line arguments
:parse_args
if "%1"=="" goto done_parsing
if /i "%1"=="debug" set BUILD_CONFIG=Debug
if /i "%1"=="Debug" set BUILD_CONFIG=Debug
if /i "%1"=="release" set BUILD_CONFIG=Release
if /i "%1"=="Release" set BUILD_CONFIG=Release
if /i "%1"=="all" set BUILD_ALL=1
if /i "%1"=="All" set BUILD_ALL=1
shift
goto parse_args
:done_parsing

echo Build Configuration: %BUILD_CONFIG%

REM Check if premake5 is available
where premake5 >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: premake5 not found. Please ensure premake5 is installed and in your PATH.
    pause
    exit /b 1
)

REM Run premake5 with the vs2022 argument
premake5 vs2022
if %ERRORLEVEL% neq 0 (
    echo Error: premake5 failed to generate the solution.
    pause
    exit /b 1
)

REM Check if MSBuild is available
where msbuild >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: MSBuild not found. Please ensure Visual Studio is installed and the Developer Command Prompt is in your PATH.
    pause
    exit /b 1
)

if %BUILD_ALL%==1 (
    echo Building both Debug and Release configurations...
    
    echo Building Debug configuration...
    msbuild MultiLayerPerception.sln /p:Configuration=Debug /p:Platform=x64
    if !ERRORLEVEL! neq 0 (
        echo Debug build failed!
        pause
        exit /b 1
    )
    
    echo Building Release configuration...
    msbuild MultiLayerPerception.sln /p:Configuration=Release /p:Platform=x64
    if !ERRORLEVEL! neq 0 (
        echo Release build failed!
        pause
        exit /b 1
    )
    
    echo Both configurations built successfully!
) else (
    echo Building %BUILD_CONFIG% configuration...
    msbuild MultiLayerPerception.sln /p:Configuration=%BUILD_CONFIG% /p:Platform=x64
    if !ERRORLEVEL! neq 0 (
        echo Build failed!
        pause
        exit /b 1
    )
    echo %BUILD_CONFIG% configuration built successfully!
)

echo.
echo Build completed successfully.
echo The executables can be found in bin/%BUILD_CONFIG%/
pause
