<#
Minimal formatter/linter runner for MonStim Analyzer.
Runs (in order):
  1. isort monstim_signals monstim_gui tests
  2. black .
  3. flake8 .
Streams raw tool output directly. No summaries, no diff analysis.

Usage:
  conda activate alv_lab
  ./precommit_cleanup.ps1            # format (in-place) + lint
  ./precommit_cleanup.ps1 -Check     # check only (no file modifications)

Exit code: first non‑zero tool exit code (isort > black > flake8).
#>
param(
    [switch]$Check,
    [switch]$NoColor
)

# --- Simple color helper (optional) ---
$useColor = -not $NoColor
function C($text, $color) {
    if (-not $useColor) { return $text }
    $map = @{ red=31; green=32; yellow=33; blue=34; magenta=35; cyan=36; gray=90 }
    if ($map.ContainsKey($color)) { return "`e[$($map[$color])m$text`e[0m" } else { return $text }
}

if ($env:CONDA_DEFAULT_ENV -ne 'alv_lab') {
    Write-Host (C "[warn] Active env: $($env:CONDA_DEFAULT_ENV) (expected alv_lab)" 'yellow')
}

function Run-Cmd {
    param(
        [string]$Label,
        [string]$Cmd
    )
    Write-Host (C "== $Label ==" 'cyan')
    Write-Host (C "> $Cmd" 'gray')
    & powershell -NoLogo -NoProfile -Command $Cmd
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host (C "$Label exit 0" 'green')
    } else {
        Write-Host (C "$Label exit $code" 'red')
    }
    return $code
}

# Build commands
$isortArgs = if ($Check) { '--check-only --diff' } else { '--profile black' }
$blackArgs = if ($Check) { '--check --diff .' } else { '.' }
$flakeArgs = '.'  # flake8 always just checks

$exitCode = 0
if ($exitCode -eq 0) { $exitCode = Run-Cmd -Label 'isort' -Cmd "isort monstim_signals monstim_gui tests $isortArgs" }
if ($exitCode -eq 0) { $exitCode = Run-Cmd -Label 'black' -Cmd "black $blackArgs" }
if ($exitCode -eq 0) { $exitCode = Run-Cmd -Label 'flake8' -Cmd "flake8 $flakeArgs" }

exit $exitCode
