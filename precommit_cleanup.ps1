<#
Pre-commit cleanup / QA script for MonStim Analyzer

Runs (in order):
  1. isort monstim_signals monstim_gui tests
  2. black .
  3. flake8 .

Produces a concise report of:
  - Any formatting changes introduced
  - flake8 issues (errors & warnings)
  - Exit status summary

USAGE (PowerShell):
  # (Recommended) Ensure the correct conda env is active first
  conda activate alv_lab
  ./precommit_cleanup.ps1               # normal run (formats then lints)
  ./precommit_cleanup.ps1 -CheckOnly    # do not modify files, just report
  ./precommit_cleanup.ps1 -FailOnChange # exit non-zero if formatting changes were made
  ./precommit_cleanup.ps1 -Verbose      # more detailed output

PARAMETERS:
  -CheckOnly     : Run tools in check mode (no modifications). isort & black use --check.
  -FailOnChange  : If formatting changes occurred (or would occur in CheckOnly), exit with code 2.
  -NoColor       : Disable ANSI color output (for CI logs that don't support color).
  -LogDir        : Directory to write detailed logs (default: build/qa_logs). Created if missing.

EXIT CODES:
   0 = Success (no flake8 errors, no (or allowed) formatting changes)
   1 = flake8 reported errors
   2 = Formatting changes occurred (only if -FailOnChange specified)

NOTES:
  - This script purposefully does not stage or add files; you decide what to commit.
  - Requires isort, black, flake8 to be installed in the active environment (alv_lab).
  - Warns if the active conda environment is not 'alv_lab'.
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$FailOnChange,
    [switch]$NoColor,
    [switch]$ShowToolOutput,
    [string]$LogDir = 'build/qa_logs'
)

# ----------------------------- Utility / Styling ---------------------------------
$script:UseColor = -not $NoColor
function Color($text, $color) {
    if (-not $script:UseColor) { return $text }
    $map = @{ 'red' = 31; 'green' = 32; 'yellow' = 33; 'blue' = 34; 'magenta' = 35; 'cyan' = 36; 'gray' = 90 }
    if ($map.ContainsKey($color)) { return "`e[${($map[$color])}m$text`e[0m" } else { return $text }
}
function Section($title) { Write-Host (Color "==== $title ====" 'cyan') }
function Sub($title) { Write-Host (Color "--- $title" 'blue') }

# Ensure log directory
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# ----------------------------- Environment Check ---------------------------------
if ($env:CONDA_DEFAULT_ENV -ne 'alv_lab') {
    Write-Warning "Active conda environment is '$($env:CONDA_DEFAULT_ENV)' (expected 'alv_lab'). Activate it first: conda activate alv_lab"
}

# ----------------------------- Pre-run State --------------------------------------
# Capture current modified files for later diffing. We no longer warn if there are existing changes;
# instead we will show both: (1) files newly modified by formatting, (2) all modified files now.
$initialModified = (git diff --name-only) 2>$null
$hadInitialChanges = $initialModified.Count -gt 0

# Helper to run a tool and capture output + status
function Run-Tool {
    param(
        [string]$Name,
        [string]$Command,
        [string]$LogFile,
        [switch]$AllowFailure
    )
    Sub $Name
    Write-Host (Color "> $Command" 'gray')
    $output = & powershell -NoLogo -NoProfile -Command $Command 2>&1
    $exitCode = $LASTEXITCODE
    $output | Out-File -FilePath $LogFile -Encoding UTF8
    if ($exitCode -eq 0) {
        Write-Host (Color "$Name succeeded (exit 0)" 'green')
    } else {
        $msg = "$Name exited with code $exitCode"
        if ($AllowFailure) {
            Write-Host (Color $msg 'yellow')
        } else {
            Write-Host (Color $msg 'red')
        }
    }
    # Echo tool output if requested or on non-zero exit for visibility.
    if ($ShowToolOutput -or $exitCode -ne 0) {
        Write-Host (Color "--- $Name output begin ---" 'magenta')
        $output | ForEach-Object { Write-Host $_ }
        Write-Host (Color "--- $Name output end ---" 'magenta')
    }
    return [pscustomobject]@{ Name = $Name; ExitCode = $exitCode; Output = $output; LogFile = $LogFile }
}

$modeNote = if ($CheckOnly) { 'CHECK MODE (no modifications)' } else { 'FORMAT MODE (will modify files)' }
Section "Pre-commit Cleanup ($modeNote)"

# ----------------------------- Run isort ------------------------------------------
$IsortArgs = @()
if ($CheckOnly) { $IsortArgs += '--check-only'; $IsortArgs += '--diff' }
# Allow repo config (pyproject.toml) to drive profile—only add profile if not defined there.
if (-not $CheckOnly) { $IsortArgs += '--profile' ; $IsortArgs += 'black' }
$isortCmd = "isort monstim_signals monstim_gui tests $($IsortArgs -join ' ')"
$isortResult = Run-Tool -Name 'isort' -Command $isortCmd -LogFile (Join-Path $LogDir 'isort.log') -AllowFailure:$CheckOnly

# ----------------------------- Run black ------------------------------------------
$BlackArgs = @('.')
if ($CheckOnly) { $BlackArgs += '--check'; $BlackArgs += '--diff' }
$blackCmd = "black $($BlackArgs -join ' ')"
$blackResult = Run-Tool -Name 'black' -Command $blackCmd -LogFile (Join-Path $LogDir 'black.log') -AllowFailure:$CheckOnly

# ----------------------------- Run flake8 -----------------------------------------
$flakeCmd = 'flake8 .'
$flakeResult = Run-Tool -Name 'flake8' -Command $flakeCmd -LogFile (Join-Path $LogDir 'flake8.log') -AllowFailure

# ----------------------------- Post-run Analysis ----------------------------------
Section 'Post-run Analysis'
$modifiedAfter = (git diff --name-only) 2>$null

# Files newly added to the modified set because of formatting (heuristic diff vs initial list)
$formattedModified = (Compare-Object -ReferenceObject $initialModified -DifferenceObject $modifiedAfter | Where-Object { $_.SideIndicator -eq '=>' } | ForEach-Object { $_.InputObject }) | Sort-Object -Unique
$formattedModifiedCount = $formattedModified.Count
if ($formattedModifiedCount -gt 0) {
    Write-Host (Color "Files newly modified by formatting: $formattedModifiedCount" 'yellow')
    $formattedModified | ForEach-Object { Write-Host (Color "  - $_" 'yellow') }
} else {
    Write-Host (Color 'No additional files modified beyond those already changed.' 'green')
}

# Full current modified file list (regardless of cause)
$allCurrentModified = $modifiedAfter | Sort-Object -Unique
$allCurrentCount = $allCurrentModified.Count
Write-Host (Color "All currently modified files (working tree): $allCurrentCount" ($allCurrentCount -gt 0 ? 'cyan' : 'green'))
if ($allCurrentCount -gt 0) {
    $allCurrentModified | ForEach-Object { Write-Host (Color "  - $_" 'cyan') }
}

# flake8 issues summary
$flakeErrors = @($flakeResult.Output | Where-Object { $_ -match '^[^ ]+:[0-9]+:[0-9]+' })
if ($flakeErrors.Count -gt 0) {
    Write-Host (Color "flake8 reported $($flakeErrors.Count) issue(s)." 'red')
    $preview = $flakeErrors | Select-Object -First 30
    $preview | ForEach-Object { Write-Host (Color "  $_" 'red') }
    if ($flakeErrors.Count -gt $preview.Count) {
        Write-Host (Color "  ... ($($flakeErrors.Count - $preview.Count) more, see $(Join-Path $LogDir 'flake8.log'))" 'red')
    }
} else {
    Write-Host (Color 'flake8: no issues.' 'green')
}

# ----------------------------- Summary --------------------------------------------
Section 'Summary'
$summaryTable = @(
    [pscustomobject]@{ Tool = 'isort';  ExitCode = $isortResult.ExitCode }
    [pscustomobject]@{ Tool = 'black';  ExitCode = $blackResult.ExitCode }
    [pscustomobject]@{ Tool = 'flake8'; ExitCode = $flakeResult.ExitCode }
)
$summaryTable | ForEach-Object {
    $color = if ($_.ExitCode -eq 0) { 'green' } elseif ($_.Tool -eq 'flake8') { 'red' } else { 'yellow' }
    Write-Host (Color ("{0,-8} -> exit {1}" -f $_.Tool, $_.ExitCode) $color)
}
Write-Host (Color ("Newly formatted files: $formattedModifiedCount") ($formattedModifiedCount -gt 0 ? 'yellow' : 'green'))
Write-Host (Color ("Currently modified (all): $allCurrentCount") ($allCurrentCount -gt 0 ? 'cyan' : 'green'))

$finalExit = 0
if ($flakeResult.ExitCode -ne 0 -and $flakeErrors.Count -gt 0) { $finalExit = 1 }
if ($FailOnChange -and $formattedModifiedCount -gt 0) { if ($finalExit -eq 0) { $finalExit = 2 } }

if ($finalExit -eq 0) {
    Write-Host (Color 'All checks passed.' 'green')
} elseif ($finalExit -eq 1) {
    Write-Host (Color 'Exiting due to flake8 errors.' 'red')
} elseif ($finalExit -eq 2) {
    Write-Host (Color 'Exiting due to new formatting changes (FailOnChange enabled).' 'yellow')
}

Write-Host "Logs written to: $LogDir"

exit $finalExit
