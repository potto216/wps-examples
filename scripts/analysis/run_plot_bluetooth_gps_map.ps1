<#
.SYNOPSIS
Runs plot_bluetooth_gps_map_cli.py in automated mode for each .pcapng file in a directory tree.

.DESCRIPTION
Recursively scans a user-specified root directory for .pcapng files. For each .pcapng file,
the script looks for a .gpx file with the same base filename in the same directory. The
plot_bluetooth_gps_map_cli.py script is invoked with automated plotting options. The output
base path is set to the input file's full path without extension, producing .png and .json
outputs. The script verifies both outputs are created and errors if not.

You can skip processing if the output files already exist and ignore specific directory
names (such as "notes") anywhere in the directory tree.

.PARAMETER RootPath
Root directory to recursively scan for .pcapng files.

.PARAMETER IgnoreDirName
Directory names to ignore anywhere in the path (case-insensitive). Example: -IgnoreDirName notes,tmp

.PARAMETER SkipIfOutputsExist
If set, skips processing when both expected output files already exist.

.PARAMETER PythonPath
Python executable to invoke. Defaults to "python".

.PARAMETER PcapTimeOffset
Value for --pcap-time-offset passed to the CLI. Defaults to "+05:00".

.PARAMETER BaseMap
Value for --basemap passed to the CLI. Defaults to "osm".

.EXAMPLE
.\scripts\analysis\run_plot_bluetooth_gps_map.ps1 -RootPath "D:\Dropbox\lib\data\bluetooth" `
  -IgnoreDirName notes -SkipIfOutputsExist
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [string[]]$IgnoreDirName = @(),

    [switch]$SkipIfOutputsExist,

    [string]$PythonPath = "python",

    [string]$PcapTimeOffset = "+05:00",

    [string]$BaseMap = "osm"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

function Test-PathHasIgnoredDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string[]]$IgnoredNames
    )

    if (-not $IgnoredNames -or $IgnoredNames.Count -eq 0) {
        return $false
    }

    # Start from the containing directory if $Path is a file path
    $dir = Split-Path -Path $Path -Parent
    if (-not $dir) {
        return $false
    }

    while ($true) {
        $leaf = Split-Path -Path $dir -Leaf
        foreach ($ignored in $IgnoredNames) {
            if ($leaf.Equals($ignored, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }

        $parent = Split-Path -Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) {
            break
        }

        $dir = $parent
    }

    return $false
}

$resolvedRoot = Resolve-Path -Path $RootPath
$repoRoot = Resolve-Path -Path (Join-Path $PSScriptRoot "..\..")
$cliPath = Join-Path $repoRoot "scripts\analysis\plot_bluetooth_gps_map_cli.py"

if (-not (Test-Path $cliPath)) {
    throw "Could not find plot_bluetooth_gps_map_cli.py at $cliPath"
}

Write-Log "Scanning for .pcapng files under $resolvedRoot"
Write-Log "Ignoring directory names: $($IgnoreDirName -join ', ')"
Write-Log "Using python: $PythonPath"
Write-Log "Using CLI path: $cliPath"

$pcapFiles = Get-ChildItem -Path $resolvedRoot -Filter "*.pcapng" -File -Recurse
if (-not $pcapFiles) {
    Write-Log "No .pcapng files found."
    return
}

foreach ($pcap in $pcapFiles) {
    if (Test-PathHasIgnoredDirectory -Path $pcap.FullName -IgnoredNames $IgnoreDirName) {
        Write-Log "Skipping ignored path: $($pcap.FullName)"
        continue
    }

    $basePath = [System.IO.Path]::ChangeExtension($pcap.FullName, $null)
    if ($basePath.EndsWith('.')) {
        $basePath = $basePath.TrimEnd('.')
    }
    $gpsPath = "$basePath.gpx"
    $pngPath = "$basePath.png"
    $jsonPath = "$basePath.json"

    Write-Log "Processing: $($pcap.FullName)"

    if (-not (Test-Path $gpsPath)) {
        throw "Missing matching GPS file: $gpsPath"
    }

    if ($SkipIfOutputsExist -and (Test-Path $pngPath) -and (Test-Path $jsonPath)) {
        Write-Log "Skipping because outputs exist: $pngPath and $jsonPath"
        continue
    }

    $arguments = @(
        $cliPath,
        "--pcapng", $pcap.FullName,
        "--gps", $gpsPath,
        "--verbose",
        "--use-pcap-range",
        "--pcap-time-offset", $PcapTimeOffset,
        "--color-by-packet-type",
        "--basemap", $BaseMap,
        "--density-line",
        "--automated",
        "--output-base", $basePath
    )

    Write-Log "Running: $PythonPath $($arguments -join ' ')"
    & $PythonPath @arguments

    if ($LASTEXITCODE -ne 0) {
        throw "plot_bluetooth_gps_map_cli.py failed for $($pcap.FullName) with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $pngPath)) {
        throw "Expected output PNG not found: $pngPath"
    }

    if (-not (Test-Path $jsonPath)) {
        throw "Expected output JSON not found: $jsonPath"
    }

    Write-Log "Outputs verified: $pngPath, $jsonPath"
}

Write-Log "Processing complete."
