<#
.SYNOPSIS
Prepares the signed update catalog for one already-built MonStim ZIP.

.DESCRIPTION
This is the maintainer-friendly wrapper around tools.publish_release_catalog.
It never copies, prints, or stores the private key in the repository.  First
upload the exact ZIP to a GitHub draft release, then run this command locally.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]+)?$')]
    [string]$Version,

    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$Archive,

    [string]$ReleaseTag = "v$Version",
    [ValidateSet('beta', 'stable')]
    [string]$Channel = 'beta',
    [string]$PrivateKey = (Join-Path $env:USERPROFILE '.monstim\update-catalog-ed25519.key'),
    [string]$ChecksumsOutput,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$assetName = Split-Path -Leaf $archivePath
$releaseBaseUrl = "https://github.com/AEWorthy/MonStim-Analyzer/releases"
$assetUrl = "$releaseBaseUrl/download/$ReleaseTag/$assetName"
$notesUrl = "$releaseBaseUrl/tag/$ReleaseTag"

$arguments = @(
    '-n', 'monstim', 'python', '-m', 'tools.publish_release_catalog',
    '--version', $Version,
    '--archive', $archivePath,
    '--asset-url', $assetUrl,
    '--notes-url', $notesUrl,
    '--private-key', $PrivateKey,
    '--channel', $Channel
)
if ($ChecksumsOutput) {
    $arguments += @('--checksums-output', $ChecksumsOutput)
}
if ($DryRun) {
    $arguments += '--dry-run'
}

& conda run @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Release catalog preparation failed (exit code $LASTEXITCODE). No release should be published."
}
