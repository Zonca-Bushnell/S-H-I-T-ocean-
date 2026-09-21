param(
    [string]$Start = '1991-01-01',
    [string]$End = '1991-01-19',
    [int]$Workers = 19
)

$ErrorActionPreference = 'Stop'
$env:PYTHONNOUSERSITE = '1'
$env:HDF5_USE_FILE_LOCKING = 'FALSE'
$repo = 'D:\01_Eddy\01_Vertical_asymmetric\S-H-I-T-ocean-'
$mamba = 'D:\Util\lever\02_miniforge\Library\bin\mamba.exe'
$root = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_eta_mss_three_kernel_surface_jan01_jan19'
$logRoot = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stdout = Join-Path $logRoot 'launcher.stdout.log'
$stderr = Join-Path $logRoot 'launcher.stderr.log'
$arguments = @(
    'run', '-n', 'OFES_detection', 'python', '-m',
    'Detection_for_OFES.tools.run_ofes_e_path_three_kernels',
    '--start', $Start, '--end', $End, '--workers', $Workers
)
$process = Start-Process -FilePath $mamba -ArgumentList $arguments -WorkingDirectory $repo `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -PassThru

[pscustomobject]@{
    PID = $process.Id
    OutputRoot = $root
    Manifest = (Join-Path $root 'three_kernel_run_manifest.json')
    Stdout = $stdout
    Stderr = $stderr
    Monitor = "Get-Content '$root\three_kernel_run_manifest.json' -Raw"
} | Format-List
