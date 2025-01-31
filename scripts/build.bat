@echo off
REM Navigate to the directory containing premake5
cd /d "%~dp0\vendor\premake5"

REM Run premake5 with the vs2022 argument
premake5.exe vs2022

REM Return to the original directory
cd /d "%~dp0"

@echo Premake5 has generated the Visual Studio solution.
pause
