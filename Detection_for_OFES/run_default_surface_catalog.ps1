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
$filterRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_compatible_filter_rossby_lower_upper180'
$outputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\origin_unified_ssh_primary_target_all_open_ocean_recovery_surface_jan01_jan19'
$logRoot = Join-Path $outputRoot 'logs'
$log = Join-Path $logRoot 'default_surface_catalog.log'

Push-Location $repo
try {
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $arguments = @(
        'run', '-n', 'OFES_detection', 'python', '-m', 'Detection_for_OFES.tools.build_unified_eddy_catalog',
        '--filter-output-root', $filterRoot,
        '--output-root', $outputRoot,
        '--start', $Start, '--end', $End, '--max-depth-m', '3', '--workers', $Workers,
        '--skip-filter', '--open-ocean-no-streamline-gate', '--resume'
    )
    & $mamba @arguments *> $log
} finally {
    Pop-Location
}
