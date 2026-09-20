param(
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012',
  [int]$StartYear = 1993,
  [int]$EndYear = 2012,
  [string]$WorkerName = 'single',
  [switch]$DownloadOnly,
  [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_climatology_pydap\python.exe'
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logStem = "ofes2_monthly_climatology_$WorkerName"
$stdout = Join-Path $logDir "$logStem.stdout.log"
$stderr = Join-Path $logDir "$logStem.stderr.log"

$env:PYTHONNOUSERSITE = '1'
$arguments = @(
  '-m', 'Detection_for_OFES.tools.download_ofes2_monthly_climatology',
  '--output-root', $OutputRoot,
  '--start-year', $StartYear,
  '--end-year', $EndYear,
  '--worker-name', $WorkerName
)
if ($DownloadOnly) { $arguments += '--download-only' }
if ($BuildOnly) { $arguments += '--build-only' }
if (-not $DownloadOnly) { $arguments += '--preview' }
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$pidPath = Join-Path $OutputRoot "launcher_$WorkerName.pid"
$process.Id | Set-Content -Path $pidPath -NoNewline

[pscustomobject]@{
  ProcessId = $process.Id
  OutputRoot = $OutputRoot
  Stdout = $stdout
  Stderr = $stderr
  ProcessIdFile = $pidPath
  Monitor = "Get-Content '$stdout' -Wait -Tail 40"
}
