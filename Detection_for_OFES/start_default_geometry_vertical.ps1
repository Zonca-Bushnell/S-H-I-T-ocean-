param(
    [string]$Start = '1991-01-01',
    [string]$End = '1991-01-19',
    [int]$Workers = 2,
    [int]$CompositeWorkers = 8
)

Write-Warning 'Deprecated launcher: forwarding to eta_hp500_geometry_vertical_v1. Worker overrides are now profile-controlled (surface=8, velocity/vertical=2, W=8).'
& (Join-Path $PSScriptRoot 'start_ofes_default_run.ps1') -Start $Start -End $End -Resume
