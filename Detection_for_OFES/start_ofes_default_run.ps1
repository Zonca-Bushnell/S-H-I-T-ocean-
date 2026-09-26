param(
    [string]$Profile = 'eta_hp500_geometry_vertical_v1',
    [string]$Start = '1991-01-01',
    [string]$End = '1991-01-19',
    [string]$Stages = 'all',
    [switch]$Resume,
    [switch]$StageRaw,
    [switch]$WithTracking,
    [switch]$WithShape
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$token = "{0}_{1}" -f $Start.Replace('-', ''), $End.Replace('-', '')
$root = "E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\runs\$Profile\layout_v2\$token"
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$arguments = "-m Detection_for_OFES run --profile $Profile --start $Start --end $End --stages $Stages"
if ($Resume) { $arguments += ' --resume' }
if ($StageRaw) { $arguments += ' --stage-raw' }
if ($WithTracking) { $arguments += ' --with-tracking' }
if ($WithShape) { $arguments += ' --with-shape' }

$process = Start-Process -FilePath $python -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -ArgumentList $arguments `
    -RedirectStandardOutput (Join-Path $logs 'launcher.stdout.log') `
    -RedirectStandardError (Join-Path $logs 'launcher.stderr.log')

[pscustomobject]@{
    Pid = $process.Id
    Profile = $Profile
    OutputRoot = $root
    Stages = $Stages
} | ConvertTo-Json -Compress
