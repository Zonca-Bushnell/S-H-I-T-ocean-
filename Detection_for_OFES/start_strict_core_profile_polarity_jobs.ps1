param(
    [switch]$NoResume
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $packageRoot
$python = 'D:\Util\lever\02_miniforge\envs\OFES_detection\python.exe'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repo;$env:PYTHONPATH" } else { $repo }
$output = 'E:\DATA\01_Eddy_correspond\02_OFES\Fromthebeginning\05_TEMP\strict_core_profile_polarity_native_fields_19910101_19910119'
$logs = Join-Path $output 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$arguments = @('-u', '-m', 'Detection_for_OFES.tools.run_strict_core_profile_polarity_jobs')
if ($NoResume) { $arguments += '--no-resume' }
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'strict_core_profile_polarity.stdout.log') -RedirectStandardError (Join-Path $logs 'strict_core_profile_polarity.stderr.log') -PassThru
@{ pid = $process.Id; output_root = $output; status = (Join-Path $output 'job_status.json') } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logs 'strict_core_profile_polarity.launch.json')
Write-Output "PID: $($process.Id)"
Write-Output "Status: $(Join-Path $output 'job_status.json')"
Write-Output "Log: $(Join-Path $logs 'strict_core_profile_polarity.stdout.log')"
