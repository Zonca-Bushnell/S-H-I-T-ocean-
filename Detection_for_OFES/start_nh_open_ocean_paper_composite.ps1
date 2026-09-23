param(
    [int]$PollSeconds = 60,
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$verticalRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19'
$rawTable = Join-Path $verticalRoot 'raw_detection\raw_detection\daily_runs\19910101\centers_hua_style.csv'
$outputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\nh_open_ocean_cyclonic_dipole_native_w_pointwise_unrotated_eta_19910101'
$qcRoot = Join-Path $outputRoot 'surface_geometry_qc'
$objectTable = Join-Path $qcRoot 'daily_runs\19910101\centers_hua_style.csv'
$logs = Join-Path $outputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$waiter = @"
`$ErrorActionPreference = 'Stop'
while (-not (Test-Path -LiteralPath '$rawTable')) { Start-Sleep -Seconds $PollSeconds }
& '$python' -m Detection_for_OFES.tools.postprocess_ofes_eddy_qc --source-root '$(Join-Path $verticalRoot 'raw_detection\raw_detection')' --filter-root '$(Join-Path $verticalRoot 'surface_inputs_highpass_500km')' --output-root '$qcRoot' --day 1991-01-01 --skip-persistence --accept-ssh-primary-without-streamline
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
& '$python' -m Detection_for_OFES.tools.run_north_pacific_dipole_native_w_composite --object-table '$objectTable' --output-root '$outputRoot' --selection-scope nh_open_ocean --day 1991-01-01 --workers $Workers
exit `$LASTEXITCODE
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($waiter))
$process = Start-Process -FilePath 'powershell.exe' -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -ArgumentList "-NoProfile -EncodedCommand $encoded" `
    -RedirectStandardOutput (Join-Path $logs 'launcher.stdout.log') `
    -RedirectStandardError (Join-Path $logs 'launcher.stderr.log')

[pscustomobject]@{
    Pid = $process.Id
    WaitingFor = $rawTable
    OutputRoot = $outputRoot
} | ConvertTo-Json -Compress
