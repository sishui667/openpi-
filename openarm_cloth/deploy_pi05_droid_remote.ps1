[CmdletBinding()]
param(
    [string]$RemoteHost = "100.122.120.76",
    [string]$RemoteUser = "jnu",
    [string]$ContainerName = "openpi-droid",
    [string]$ModelConfig = "pi05_droid",
    [int]$Port = 8000,
    [switch]$FollowLogs
)

$ErrorActionPreference = "Stop"
$target = "$RemoteUser@$RemoteHost"
$image = "docker-openpi_server:latest"
$checkpoint = "gs://openpi-assets/checkpoints/$ModelConfig"

function Invoke-Remote([string]$Command) {
    & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[1/5] Checking SSH and Docker on $target"
Invoke-Remote "hostname; docker version --format '{{.Server.Version}}'"

Write-Host "[2/5] Stopping the previous ALOHA server, if present"
Invoke-Remote "docker stop openpi-aloha-towel 2>/dev/null || true"

Write-Host "[3/5] Removing only the managed DROID container, if present"
Invoke-Remote "docker rm -f $ContainerName 2>/dev/null || true"

$checkpointCheck = & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "test -f /home/jnu/.cache/openpi/openpi-assets/checkpoints/$ModelConfig/params/_METADATA && test -d /home/jnu/.cache/openpi/openpi-assets/checkpoints/$ModelConfig/assets"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[3b/5] Downloading $ModelConfig into the shared cache"
    Invoke-Remote "docker rm -f openpi-checkpoint-download 2>/dev/null || true"
    Invoke-Remote "docker run -d --rm --name openpi-checkpoint-download --network host -v /home/jnu/.cache/openpi:/openpi_assets --entrypoint /bin/sh $image -c 'sleep 3600'"

    $downloadPy = @'
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import gcsfs

remote_root = "__REMOTE_ROOT__"
local_root = Path("/openpi_assets/openpi-assets/checkpoints/__MODEL_CONFIG__")
local_root.mkdir(parents=True, exist_ok=True)
fs = gcsfs.GCSFileSystem(token="anon")
files = fs.find(remote_root)

def download_one(remote_path: str) -> str:
    relative = remote_path[len(remote_root):].lstrip("/")
    local_path = local_root / relative
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_name(local_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    fs.get(remote_path, str(partial_path))
    os.replace(partial_path, local_path)
    return relative

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(download_one, path) for path in files]
    for index, future in enumerate(as_completed(futures), start=1):
        print(f"[{index}/{len(futures)}] {future.result()}", flush=True)
print("checkpoint download complete", flush=True)
'@
    $downloadPy = $downloadPy.Replace("__REMOTE_ROOT__", "openpi-assets/checkpoints/$ModelConfig")
    $downloadPy = $downloadPy.Replace("__MODEL_CONFIG__", $ModelConfig)
    $downloadPy | & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker exec -i openpi-checkpoint-download /.venv/bin/python3.11 -"
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint download failed"
    }
    Invoke-Remote "docker rm -f openpi-checkpoint-download 2>/dev/null || true"
}

Write-Host "[4/5] Starting $ModelConfig on port $Port"
$run = @(
    "docker run -d",
    "--name $ContainerName",
    "--gpus all",
    "--network host",
    "--restart unless-stopped",
    "-e OPENPI_DATA_HOME=/openpi_assets",
    "-e IS_DOCKER=true",
    "-e OPENPI_FORCE_FLOAT32=1",
    "-e XLA_PYTHON_CLIENT_PREALLOCATE=false",
    "-e XLA_PYTHON_CLIENT_ALLOCATOR=platform",
    "-v /home/jnu/openpi:/app",
    "-v /home/jnu/.cache/openpi:/openpi_assets",
    $image,
    "/.venv/bin/python3.11 /app/scripts/serve_policy.py",
    "--port $Port",
    "policy:checkpoint",
    "--policy.config=$ModelConfig",
    "--policy.dir=$checkpoint"
) -join " "
Invoke-Remote $run

Write-Host "[5/5] Waiting for the policy server to load the checkpoint"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    $state = & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker inspect -f '{{.State.Status}}' $ContainerName 2>/dev/null"
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot inspect remote container $ContainerName"
    }
    $state = ($state | Out-String).Trim()
    if ($state -eq "running") {
        & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker logs $ContainerName 2>&1 | grep -q 'server listening'"
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            Write-Host "OpenPI server is listening on port $Port."
            break
        }
    } elseif ($state -in @("exited", "dead")) {
        & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker logs --tail 80 $ContainerName"
        throw "Container $ContainerName exited while loading the model"
    }
    if ($i % 5 -eq 0) {
        Write-Host "  still loading ($state)..."
    }
    Start-Sleep -Seconds 5
}
if (-not $ready) {
    & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker logs --tail 80 $ContainerName"
    throw "Timed out waiting for OpenPI server readiness"
}

if ($ready) {
    Write-Host "Container is running and ready."
} else {
    Write-Host "Container state: $state"
}

& ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker ps --filter name=$ContainerName --format 'table {{.Names}}\t{{.Status}}'; docker logs --tail 30 $ContainerName"
if ($FollowLogs) {
    & ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $target "docker logs -f $ContainerName"
}
