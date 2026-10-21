# CPU monitor for the N-sweep timing runs: every 5 s, total CPU % and the CPU % (of the whole machine) of every
# process using > 1 %, tagged SWEEP for mpc_testbed_N_sweep.py, OTHER for anything else.  Appends to cpu_log.csv.
param([string]$Log = "$PSScriptRoot\sweep_logs\cpu_log.csv")
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$ncpu = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
if (-not (Test-Path $Log)) { "time,total,sweep,other,top_other" | Out-File -Encoding utf8 $Log }
while ($true) {
    $s = Get-Counter '\Processor(_Total)\% Processor Time', '\Process(*)\% Processor Time', '\Process(*)\ID Process' -SampleInterval 5 -MaxSamples 1
    $tot = ($s.CounterSamples | Where-Object { $_.Path -like '*processor(_total)*' }).CookedValue
    $pc = @{}; $pid_ = @{}
    foreach ($c in $s.CounterSamples) {                              # key = counter path minus the counter name
        $inst = $c.InstanceName                                     # (instance names repeat: python, python ...)
        $key = $c.Path.Substring(0, $c.Path.LastIndexOf('\'))
        if ($c.Path -like '*\process(*)\% processor time') { if ($inst -notin '_total', 'idle') { $pc[$key] = $c.CookedValue / $ncpu } }
        elseif ($c.Path -like '*\process(*)\id process') { $pid_[$key] = [int]$c.CookedValue }
    }
    $sweepPids = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*mpc_testbed_N_sweep*' } | ForEach-Object { $_.ProcessId })
    $sw = 0.0; $ot = 0.0; $tops = @()
    foreach ($k in $pc.Keys) {
        if ($pc[$k] -lt 1.0) { continue }
        if ($sweepPids -contains $pid_[$k]) { $sw += $pc[$k] } else { $ot += $pc[$k]; $tops += ("{0}[{1}]:{2:N1}" -f ($k -replace '.*\\process\(|\)$', ''), $pid_[$k], $pc[$k]) }
    }
    "{0},{1:N1},{2:N1},{3:N1},{4}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $tot, $sw, $ot, ($tops -join ' ') | Out-File -Append -Encoding utf8 $Log
}
