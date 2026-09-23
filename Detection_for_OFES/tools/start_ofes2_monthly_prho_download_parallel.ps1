param(
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_climatology_pydap\python.exe'
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Start-PrhoWorker([string]$Name, [int]$StartYear, [int]$EndYear) {
  $stdout = Join-Path $logDir "ofes2_monthly_prho_$Name.stdout.log"
  $stderr = Join-Path $logDir "ofes2_monthly_prho_$Name.stderr.log"
  $arguments = @('-m', 'Detection_for_OFES.tools.download_ofes2_monthly_prho', '--output-root', $OutputRoot,
    '--start-year', $StartYear, '--end-year', $EndYear, '--worker-name', $Name,
    '--lat-block-rows', 20)
  Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
}

$first = Start-PrhoWorker 'prho_part_1993_2002' 1993 2002
$second = Start-PrhoWorker 'prho_part_2003_2012' 2003 2012
Write-Output "Started pydap/DAP2 prho workers: $($first.Id), $($second.Id)"
Write-Output "Logs: $logDir"
Wait-Process -Id $first.Id, $second.Id
if ($first.ExitCode -ne 0 -or $second.ExitCode -ne 0) {
  throw "A prho download worker failed. Inspect $logDir."
}
