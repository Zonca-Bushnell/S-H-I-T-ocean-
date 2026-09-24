param(
    [string]$Start = '1991-01-01',
    [string]$End = '1991-01-19',
    [int]$Workers = 8
)

Write-Warning 'Deprecated launcher: the canonical workflow owns surface stages. Forwarding to the official profile; -Workers is ignored.'
& (Join-Path $PSScriptRoot 'start_ofes_default_run.ps1') -Start $Start -End $End -Stages 'surface-inputs,surface-filter,raw-detection,geometry-qc' -Resume
