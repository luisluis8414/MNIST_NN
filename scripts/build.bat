@echo off
setlocal enabledelayedexpansion

REM Default to Release if no configuration specified
set BUILD_CONFIG=Release
set BUILD_ALL=0

REM Parse command line arguments
:parse_args
if "%1"=="" goto done_parsing
if /i "%1"=="debug" set BUILD_CONFIG=Debug
if /i "%1"=="release" set BUILD_CONFIG=Release
if /i "%1"=="all" set BUILD_ALL=1
shift
goto parse_args
:done_parsing

echo Build Configuration: %BUILD_CONFIG%

REM Navigate to the directory containing premake5
cd /d "%~dp0\vendor\premake5"

REM Run premake5 with the vs2022 argument
premake5.exe vs2022

REM Return to the original directory
cd /d "%~dp0"
cd ..

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
