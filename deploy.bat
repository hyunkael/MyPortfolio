@echo off
echo ========================================
echo  Deep Learning Portfolio - Deploy Script
echo ========================================
echo.

echo [1/4] Cleaning previous build...
jupyter-book clean .
echo.

echo [2/4] Building Jupyter Book...
jupyter-book build .
echo.

echo [3/4] Deploying to GitHub Pages...
ghp-import -n -p -f _build/html
echo.

echo [4/4] Committing source changes...
git add .
git commit -m "Update portfolio content"
git push origin main
echo.

echo ========================================
echo  Deployment Complete!
echo ========================================
echo Your portfolio has been updated.
echo Visit: https://hyunkael.github.io/Deep-Learning-Portfolio/
echo.
echo Note: Changes may take 1-2 minutes to appear on GitHub Pages.
echo.
pause
