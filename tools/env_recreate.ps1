param(
    [string]$envName = "monstim" # Environment name to recreate
)

Write-Host "Checking for existing conda environments..."
$envList = conda env list | Out-String

if ($envList -match $envName) {
    Write-Host "Backing up current environment to 'old_environment.yml'..."
    try {
        conda run -n $envName conda env export | Out-File -Encoding utf8 old_environment.yml
    } catch {
        Write-Warning "Unable to export existing environment; it may not be fully functional. Continuing."
    }

    Write-Host "Removing existing environment '$envName'..."
    conda env remove -n $envName -y
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "`nWarning: removing environment '$envName' returned non-zero exit code. Proceeding to attempt a fresh create anyway.`n"
    }

    Write-Host "Cleaning conda caches..."
    conda clean --all --yes
} else {
    Write-Host "Environment '$envName' not found. Creating a fresh environment from 'environment.yml'."
}

Write-Host "Creating environment from 'environment.yml'..."
conda env create -n $envName -f environment.yml
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Conda environment creation failed. Inspect output above for solver errors."
} else {
    Write-Host "Environment '$envName' created successfully."
}

Write-Host "If you need a reproducible rebuild that also removes untracked package caches, run: `conda clean --all --yes` manually.`n"
