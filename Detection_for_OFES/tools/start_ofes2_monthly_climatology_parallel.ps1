param(
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONNOUSERSITE = '1'

function Start-DownloadWorker([string]$Name, [int]$StartYear, [int]$EndYear) {
  $stdout = Join-Path $logDir "ofes2_monthly_climatology_$Name.stdout.log"
  $stderr = Join-Path $logDir "ofes2_monthly_climatology_$Name.stderr.log"
  $arguments = @('-m', 'Detection_for_OFES.tools.download_ofes2_monthly_climatology', '--output-root', $OutputRoot,
    '--start-year', $StartYear, '--end-year', $EndYear, '--worker-name', $Name, '--download-only')
  Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
}

$first = Start-DownloadWorker 'part_1993_2002' 1993 2002
$second = Start-DownloadWorker 'part_2003_2012' 2003 2012
Wait-Process -Id $first.Id, $second.Id
if ($first.ExitCode -ne 0 -or $second.ExitCode -ne 0) {
  throw "A climatology download worker failed. Inspect $logDir before restarting."
}

$stdout = Join-Path $logDir 'ofes2_monthly_climatology_aggregate.stdout.log'
$stderr = Join-Path $logDir 'ofes2_monthly_climatology_aggregate.stderr.log'
$arguments = @('-m', 'Detection_for_OFES.tools.download_ofes2_monthly_climatology', '--output-root', $OutputRoot,
  '--start-year', 1993, '--end-year', 2012, '--worker-name', 'aggregate', '--build-only', '--preview')
$aggregate = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Wait-Process -Id $aggregate.Id
if ($aggregate.ExitCode -ne 0) {
  throw "Climatology aggregation failed. Inspect $stderr."
}
