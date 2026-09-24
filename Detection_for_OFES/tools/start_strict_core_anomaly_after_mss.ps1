param(
  [string]$MssPath = 'E:\DATA\01_Eddy_correspond\02_OFES\ofes2_monthly_climatology_1993_2012\climatology\ofes2_prho_annual_mss_1993_2012.npy',
  [string]$OutputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\nh_cyclonic_strict_core_raw_fields_19910101_19910119'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = 'D:\Util\lever\02_miniforge\envs\OFES_climatology_pydap\python.exe'
$log = Join-Path $OutputRoot 'logs\annual_mss_family_panels.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $log) -Force | Out-Null
while (-not (Test-Path -LiteralPath $MssPath)) {
  Add-Content -LiteralPath $log -Value "$(Get-Date -Format s) waiting for annual prho MSS: $MssPath"
  Start-Sleep -Seconds 300
}
& $python -u -m Detection_for_OFES.tools.render_multiday_strict_core_raw_fields --mode with-anomaly --output-root $OutputRoot *>> $log
if ($LASTEXITCODE -ne 0) { throw "Density-anomaly family panel render failed with exit code $LASTEXITCODE" }
