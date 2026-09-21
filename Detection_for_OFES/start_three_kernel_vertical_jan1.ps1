param(
    [int]$WaitSeconds = 30,
    [double]$WaitTimeoutHours = 6
)

$ErrorActionPreference = 'Stop'
$env:PYTHONNOUSERSITE = '1'
$env:HDF5_USE_FILE_LOCKING = 'FALSE'
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$mamba = 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe'
$root = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_eta_mss_three_kernel_surface_jan01_jan19'
$logRoot = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stdout = Join-Path $logRoot 'vertical_jan1_launcher.stdout.log'
$stderr = Join-Path $logRoot 'vertical_jan1_launcher.stderr.log'
$arguments = @(
    'run', '-n', 'OFES_detection', 'python', '-m',
    'Detection_for_OFES.tools.run_three_kernel_vertical_jan1',
    '--wait-seconds', $WaitSeconds,
    '--wait-timeout-hours', $WaitTimeoutHours
)
$process = Start-Process -FilePath $mamba -ArgumentList $arguments -WorkingDirectory $repo `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

[pscustomobject]@{
    PID = $process.Id
    Manifest = (Join-Path $root 'three_kernel_vertical_jan01_manifest.json')
    Stdout = $stdout
    Stderr = $stderr
    Monitor = "Get-Content '$root\three_kernel_vertical_jan01_manifest.json' -Raw"
} | Format-List
