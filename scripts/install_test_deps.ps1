# PowerShell helper to install testing dependencies in the current Python environment
Write-Host "Installing pytest and optional packages (hmmlearn, pytest-cov)..."
python -m pip install --user --upgrade pip
python -m pip install --user pytest pytest-cov
# Optional: install hmmlearn for HMM features
python -m pip install --user hmmlearn
Write-Host "Done. Run `python -m pytest -q` to run tests."