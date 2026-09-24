param(
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012',
  [int]$StallMinutes = 45
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_climatology_pydap\python.exe'
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$supervisorLog = Join-Path $logDir 'ofes2_monthly_prho_supervisor.log'

function Write-SupervisorLog([string]$Message) {
  $line = "$(Get-Date -Format s) $Message"
  Add-Content -LiteralPath $supervisorLog -Value $line
  Write-Output $line
}

function Start-DownloadWorker([string]$Name, [int]$StartYear, [int]$EndYear) {
  $stdout = Join-Path $logDir "ofes2_monthly_prho_$Name.stdout.log"
  $stderr = Join-Path $logDir "ofes2_monthly_prho_$Name.stderr.log"
  $arguments = @('-m', 'Detection_for_OFES.tools.download_ofes2_monthly_prho', '--output-root', $OutputRoot,
    '--start-year', $StartYear, '--end-year', $EndYear, '--worker-name', $Name, '--lat-block-rows', 20)
  $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  Write-SupervisorLog "started $Name pid=$($process.Id) period=$StartYear-$EndYear"
  return [pscustomobject]@{ Name=$Name; StartYear=$StartYear; EndYear=$EndYear; Process=$process; Stdout=$stdout; RestartCount=0 }
}

function WorkerComplete($Worker) {
  $manifest = Join-Path $OutputRoot ("workers\{0}\monthly_prho_manifest.json" -f $Worker.Name)
  if (-not (Test-Path -LiteralPath $manifest)) { return $false }
  try { return ((Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).status -eq 'complete') }
  catch { return $false }
}

$workers = @(
  (Start-DownloadWorker 'prho_part_1993_2002' 1993 2002),
  (Start-DownloadWorker 'prho_part_2003_2012' 2003 2012)
)

while ($true) {
  $complete = 0
  for ($index = 0; $index -lt $workers.Count; $index++) {
    $worker = $workers[$index]
    if (WorkerComplete $worker) { $complete++; continue }
    $stale = $false
    if (Test-Path -LiteralPath $worker.Stdout) {
      $age = ((Get-Date) - (Get-Item -LiteralPath $worker.Stdout).LastWriteTime).TotalMinutes
      $stale = $age -ge $StallMinutes
    }
    if ($worker.Process.HasExited -or $stale) {
      if (-not $worker.Process.HasExited) {
        Write-SupervisorLog "stalled $($worker.Name) pid=$($worker.Process.Id); stopping after $StallMinutes minutes without stdout progress"
        Stop-Process -Id $worker.Process.Id -Force
      } else {
        Write-SupervisorLog "exited $($worker.Name) pid=$($worker.Process.Id) exit=$($worker.Process.ExitCode); resuming cache"
      }
      Start-Sleep -Seconds 15
      $replacement = Start-DownloadWorker $worker.Name $worker.StartYear $worker.EndYear
      $replacement.RestartCount = $worker.RestartCount + 1
      $workers[$index] = $replacement
    }
  }
  if ($complete -eq $workers.Count) { break }
  Start-Sleep -Seconds 60
}

Write-SupervisorLog 'all monthly prho workers complete; building day-weighted annual MSS'
& $python -m Detection_for_OFES.tools.build_ofes2_prho_annual_mss --output-root $OutputRoot *>> (Join-Path $logDir 'ofes2_prho_annual_mss.log')
if ($LASTEXITCODE -ne 0) { throw "Annual prho MSS build failed with exit code $LASTEXITCODE" }
Write-SupervisorLog 'annual prho MSS complete'
$fieldOutput = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\nh_cyclonic_strict_core_raw_fields_19910101_19910119'
Write-SupervisorLog 'starting final strict-core density-anomaly family panels'
& $python -u -m Detection_for_OFES.tools.render_multiday_strict_core_raw_fields --mode with-anomaly --output-root $fieldOutput *>> (Join-Path $fieldOutput 'logs\annual_mss_family_panels.log')
if ($LASTEXITCODE -ne 0) { throw "Density-anomaly family panel render failed with exit code $LASTEXITCODE" }
Write-SupervisorLog 'final strict-core density-anomaly family panels complete'
