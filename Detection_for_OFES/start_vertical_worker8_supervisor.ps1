param(
    [int]$Workers = 8,
    [int]$MaxAttempts = 3,
    [switch]$LegacyMatlabBridge
)

$ErrorActionPreference = 'Stop'
if (-not $LegacyMatlabBridge) {
    throw 'This MATLAB bridge supervisor is legacy-only. Use start_ofes_default_run.ps1 for the official workflow, or pass -LegacyMatlabBridge for historical reproduction.'
}
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$root = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$stdout = Join-Path $logs 'worker8_supervisor.stdout.log'
$stderr = Join-Path $logs 'worker8_supervisor.stderr.log'
$started = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"===== scheduled supervisor start $started workers=$Workers =====" | Add-Content -LiteralPath $stdout

Push-Location $repo
try {
    & $python -m Detection_for_OFES.tools.resume_vertical_with_matlab_bridge `
        --start 1991-01-01 --end 1991-01-19 `
        --workers $Workers --max-attempts $MaxAttempts --retry-delay-seconds 30 `
        1>> $stdout 2>> $stderr
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$finished = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"===== scheduled supervisor finish $finished exit=$exitCode =====" | Add-Content -LiteralPath $stdout
exit $exitCode
