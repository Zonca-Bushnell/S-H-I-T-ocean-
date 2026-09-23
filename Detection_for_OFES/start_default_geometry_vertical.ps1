param(
    [string]$Start = '1991-01-01',
    [string]$End = '1991-01-19',
    [int]$Workers = 19
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$root = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$process = Start-Process -FilePath $python -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -ArgumentList "-m Detection_for_OFES.tools.run_default_geometry_vertical --start $Start --end $End --workers $Workers --resume" `
    -RedirectStandardOutput (Join-Path $logs 'launcher.stdout.log') `
    -RedirectStandardError (Join-Path $logs 'launcher.stderr.log')

[pscustomobject]@{ Pid = $process.Id; OutputRoot = $root } | ConvertTo-Json -Compress
