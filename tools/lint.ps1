<#
Minimal Ruff formatter/linter runner for MonStim Analyzer.

Runs:
  1. ruff check (lint + import sorting)
  2. ruff format (formatter)

Streams raw tool output directly.

Usage:
    conda activate monstim

    .\precommit_cleanup.ps1
        Format in-place, then lint.

    .\precommit_cleanup.ps1 -Check
        Check only (no file modifications).

Exit code:
    First non-zero exit code encountered.
#>

param(
    [switch]$Check,
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
        red     = 31
        green   = 32
        yellow  = 33
        blue    = 34
        magenta = 35
        cyan    = 36
        gray    = 90
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
    Write-Host (C "[warn] Active environment: $($env:CONDA_DEFAULT_ENV) (expected monstim)" "yellow")
}

# -----------------------------
# Tool runner
# -----------------------------
function Invoke-Tool {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host (C "== $Name ==" "cyan")
    Write-Host (C "> $Executable $($Arguments -join ' ')" "gray")

    & $Executable @Arguments

    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Write-Host (C "$Name exit 0" "green")
    }
    else {
        Write-Host (C "$Name exit $code" "red")
    }

    return $code
}

# -----------------------------
# Arguments
# -----------------------------
if ($Check) {
    $ruffCheckArgs = @("check", ".")
    $ruffFormatArgs = @("format", "--check", ".")
}
else {
    $ruffCheckArgs = @("check", "--fix", ".")
    $ruffFormatArgs = @("format", ".")
}

# -----------------------------
# Run Ruff
# -----------------------------
$tools = @(
    @{
        Name = "ruff check"
        Exe  = "ruff"
        Args = $ruffCheckArgs
    },
    @{
        Name = "ruff format"
        Exe  = "ruff"
        Args = $ruffFormatArgs
    }
)

$exitCode = 0

foreach ($tool in $tools) {
    $code = Invoke-Tool `
        -Name $tool.Name `
        -Executable $tool.Exe `
        -Arguments $tool.Args

    if ($exitCode -eq 0 -and $code -ne 0) {
        $exitCode = $code
    }
}

exit $exitCode