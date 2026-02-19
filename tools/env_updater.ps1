# PowerShell script to update your Conda environment using requirements.txt
# Usage: Run from an Anaconda Prompt or a PowerShell session where conda is available

param(
    [string]$envName = "monstim" # Update this to your environment name
)

Write-Host "Backing up current environment..."
conda run -n $envName conda env export | Out-File -Encoding utf8 old_environment.yml

Write-Host "Updating environment from environment.yml..."
conda env update -n $envName -f environment.yml --prune
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Conda update may have encountered issues. Review the output above for details."
}

Write-Host "Environment update complete."
