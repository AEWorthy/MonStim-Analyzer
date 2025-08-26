# PowerShell script to update your Conda environment using requirements.txt
# Usage: Run from an Anaconda Prompt or a PowerShell session where conda is available

param(
    [string]$envName = "alv_lab" # Update this to your environment name
)


Write-Host "Backing up current environment package lists..."
conda run -n $envName conda list --export | Out-File -Encoding utf8 old_requirements.txt
conda run -n $envName conda env export | Out-File -Encoding utf8 old_environment.yml

Write-Host "Updating conda packages from requirements.txt (where possible)..."
conda run -n $envName conda install --file requirements.txt --yes
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Conda install failed or some packages could not be resolved. Continuing with pip..."
}

Write-Host "Updating pip-only packages and ensuring all requirements are met..."
conda run -n $envName pip install --upgrade --requirement requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pip install failed."
    exit 1
}

Write-Host "Environment update complete."
