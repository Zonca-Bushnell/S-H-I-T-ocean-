param(
  [string]$DestRoot = 'F:\Tide\FES2014',
  [string]$Python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe',
  [string]$Components = 'elevations currents load extrapolated'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logDir = Join-Path $DestRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdout = Join-Path $logDir 'download_fes2014.stdout.log'
$stderr = Join-Path $logDir 'download_fes2014.stderr.log'

$script = Join-Path $repoRoot 'Detection_for_OFES\tools\download_fes2014.py'
$componentArgs = $Components -split '\s+' | Where-Object { $_ }
$argList = @(
  '-m', 'Detection_for_OFES.datasets.download_fes2014',
  '--dest-root', $DestRoot,
  '--components'
) + $componentArgs

$env:PYTHONNOUSERSITE = '1'
$proc = Start-Process -FilePath $Python `
  -ArgumentList $argList `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

[pscustomobject]@{
  ProcessId = $proc.Id
  DestRoot = $DestRoot
  Stdout = $stdout
  Stderr = $stderr
  Note = 'Set AVISO_USERNAME and AVISO_PASSWORD before running if they are not already in the environment.'
}
