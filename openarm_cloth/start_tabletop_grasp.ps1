$taskRoot = $PSScriptRoot
$taskParent = Split-Path -Parent $taskRoot
$taskRuntime = Join-Path $taskParent 'venv_member_ab_live\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $taskRuntime)) {
    throw '请在项目相邻目录配置 venv_member_ab_live，或用装好仿真依赖的 Python 运行 launch_tabletop_grasp.pyw。'
}
$taskLauncher = Join-Path $taskRoot 'launch_tabletop_grasp.pyw'
Start-Process -FilePath $taskRuntime -ArgumentList @('"' + $taskLauncher + '"') -WorkingDirectory $taskRoot -WindowStyle Hidden
