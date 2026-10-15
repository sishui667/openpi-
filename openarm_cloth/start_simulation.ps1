$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$candidates = @((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'))
if ($env:CONDA_PREFIX) {
    $candidates += Join-Path $env:CONDA_PREFIX 'python.exe'
}
$candidates += 'D:\miniconda3\envs\openpi\python.exe'
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
    $candidates += $pythonCommand.Source
}
$simulationPython = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$simulationPython) {
    throw 'No Python environment found. Create .venv and install requirements_simulation.txt first.'
}
& $simulationPython -c 'import mujoco, cv2, numpy, scipy, glfw, PIL'
if ($LASTEXITCODE -ne 0) {
    throw 'Missing simulation dependencies. Install requirements_simulation.txt in the selected environment.'
}
& $simulationPython -u (Join-Path $PSScriptRoot 'demo_grasp_viewer.py') --interactive-place-s 86400 --preview-initial-s 5 --post-lift-s 1.5
exit $LASTEXITCODE
