# PowerShell helper to install hmmlearn in the current Python environment
Write-Host "Attempting to install hmmlearn via pip..."
python -m pip install --user --upgrade pip
python -m pip install --user hmmlearn
Write-Host "Done. Please re-run your scripts."