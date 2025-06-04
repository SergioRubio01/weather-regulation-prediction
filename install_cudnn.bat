@echo off
echo Installing cuDNN files to CUDA directories...
echo.

set CUDNN_PATH=%1
if "%CUDNN_PATH%"=="" (
    echo Usage: install_cudnn.bat ^<path_to_extracted_cudnn^>
    echo Example: install_cudnn.bat C:\Downloads\cudnn-windows-x86_64-9.5.1.17_cuda12
    exit /b 1
)

set CUDA_PATH_V12_5=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5
set CUDA_PATH_V12_6=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

echo Copying to CUDA 12.5...
xcopy /y /s "%CUDNN_PATH%\bin\*" "%CUDA_PATH_V12_5%\bin\"
xcopy /y /s "%CUDNN_PATH%\include\*" "%CUDA_PATH_V12_5%\include\"
xcopy /y /s "%CUDNN_PATH%\lib\*" "%CUDA_PATH_V12_5%\lib\"

echo.
echo Copying to CUDA 12.6...
xcopy /y /s "%CUDNN_PATH%\bin\*" "%CUDA_PATH_V12_6%\bin\"
xcopy /y /s "%CUDNN_PATH%\include\*" "%CUDA_PATH_V12_6%\include\"
xcopy /y /s "%CUDNN_PATH%\lib\*" "%CUDA_PATH_V12_6%\lib\"

echo.
echo cuDNN installation completed!
echo Please restart your terminal and run check_cuda.py again.
