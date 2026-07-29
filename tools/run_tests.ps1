<#
Run pytest for MonStim Analyzer.

Runs:
  python -m pytest tests -vv -s

Streams pytest output live.

Usage:
    conda activate monstim
    .\run_tests.ps1
#>

param(
    [switch]$NoColor
)

# -----------------------------
# Optional ANSI colors
# -----------------------------
$useColor = -not $NoColor

function C {
    param(
        [string]$Text,
        [string]$Color
    )

    if (-not $useColor) {
        return $Text
    }

    $map = @{
        red   = 31
        green = 32
        cyan  = 36
        gray  = 90
    }

    if ($map.ContainsKey($Color)) {
        return "`e[$($map[$Color])m$Text`e[0m"
    }

    return $Text
}

# -----------------------------
# Environment check
# -----------------------------
if ($env:CONDA_DEFAULT_ENV -ne "monstim") {
    Write-Host (C "[warn] Active environment: $($env:CONDA_DEFAULT_ENV) (expected monstim)" "red")
}

Write-Host ""
Write-Host (C "== Running pytest ==" "cyan")
Write-Host (C "> python -m pytest tests -v -s" "gray")
Write-Host ""

# -----------------------------
# Run pytest directly
# -----------------------------
python -m pytest tests -v -s

$exitCode = $LASTEXITCODE

Write-Host ""

if ($exitCode -eq 0) {
    Write-Host (C "pytest passed" "green")
}
else {
    Write-Host (C "pytest failed (exit $exitCode)" "red")
}

exit $exitCode