@echo off
title Aqua Scope - Push to GitHub
echo ======================================================================
echo                PUSHING CAPSTONE PROJECT TO GITHUB
echo    Repository: https://github.com/meghanamakkena03/Capstone_Project
echo ======================================================================
echo.
echo Make sure you have created the empty repository on GitHub:
echo https://github.com/new (Repository name: Capstone_Project)
echo.
echo Pushing branch 'main' to origin...
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo [SUCCESS] Repository pushed to GitHub successfully!
    echo Next step: Deploy to Vercel at https://vercel.com/new
    echo ======================================================================
) else (
    echo ======================================================================
    echo [NOTE] If this is your first push, please sign in via the browser
    echo or check that the repository 'Capstone_Project' exists on GitHub.
    echo ======================================================================
)
pause

