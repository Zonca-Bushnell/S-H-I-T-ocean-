param(
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_climatology_pydap\python.exe'
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Start-PrhoWorker([string]$Name, [int]$StartYear, [int]$EndYear) {
  $stdout = Join-Path $logDir "ofes2_annual_prho_$Name.stdout.log"
  $stderr = Join-Path $logDir "ofes2_annual_prho_$Name.stderr.log"
  $arguments = @('-m', 'Detection_for_OFES.datasets.download_ofes2_annual_prho', '--output-root', $OutputRoot,
    '--start-year', $StartYear, '--end-year', $EndYear, '--worker-name', $Name,
    '--lat-block-rows', 10, '--depth-block-layers', 20,
    '--block-retries', 5, '--retry-seconds', 15, '--retry-max-seconds', 300,
    '--retry-jitter-seconds', 7)
  Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
}

$first = Start-PrhoWorker 'annual_prho_1993_2002' 1993 2002
$second = Start-PrhoWorker 'annual_prho_2003_2012' 2003 2012
Write-Output "Started server-side annual prho workers: $($first.Id), $($second.Id)"
Write-Output "Logs: $logDir"
Wait-Process -Id $first.Id, $second.Id
if ($first.ExitCode -ne 0 -or $second.ExitCode -ne 0) {
  throw "An annual prho worker failed. Inspect $logDir."
}

& $python -m Detection_for_OFES.datasets.download_ofes2_annual_prho `
  --output-root $OutputRoot --start-year 1993 --end-year 2012 --finalize-only `
  1> (Join-Path $logDir 'ofes2_annual_prho_finalize.stdout.log') `
  2> (Join-Path $logDir 'ofes2_annual_prho_finalize.stderr.log')
if ($LASTEXITCODE -ne 0) {
  throw "Annual prho climatology finalization failed. Inspect $logDir."
}
