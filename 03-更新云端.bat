@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 上传本地改动到云端

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo [错误] 当前文件夹不是 Git 仓库
    pause
    exit /b 1
)

git add -A
git diff --cached --quiet
if not errorlevel 1 (
    echo 没有新的改动，跳过提交
) else (
    git commit -m "update %date% %time%"
)

git push
if errorlevel 1 (
    echo.
    echo [失败] 上传失败。常见原因：
    echo   - 云端有你本地没有的提交：先双击 02-更新本地.bat
    echo   - SSH key 未加载 / 网络问题：看上面的具体报错
) else (
    echo.
    echo [完成] 已推送到云端
)
pause