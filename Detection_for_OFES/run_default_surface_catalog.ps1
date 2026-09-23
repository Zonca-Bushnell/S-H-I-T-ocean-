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
$outputRoot = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\04_Vertical\eta_highpass_500km_no_tilecap_ssh_geometry_section_bipolar_jan01_jan19'
$logRoot = Join-Path $outputRoot 'logs'
$log = Join-Path $logRoot 'default_surface_catalog.log'

Push-Location $repo
try {
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $arguments = @(
        'run', '-n', 'OFES_detection', 'python', '-m', 'Detection_for_OFES.tools.run_default_geometry_vertical',
        '--start', $Start, '--end', $End, '--workers', $Workers, '--resume'
    )
    & $mamba @arguments *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Default eta 500-km geometry-vertical workflow failed with exit code $LASTEXITCODE. See $log"
    }
} finally {
    Pop-Location
}
