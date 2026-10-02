param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$appDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $appDir "..\..")).Path
$resultsDir = Join-Path $appDir "results"
$buildDir = Join-Path $appDir "build"
$exe = Join-Path $buildDir "ddpm_shapes.exe"
$rawLog = Join-Path $resultsDir "babyctorch_raw.log"

New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Get-ChildItem $resultsDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found. Run this from a shell with the CUDA toolkit on PATH."
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "cl.exe is not on PATH. On Windows, launch an x64 Native Tools Command Prompt for Visual Studio 2022, then run PowerShell from there."
}

$python = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $python = "py"
} else {
    throw "Python was not found on PATH."
}

$steps = if ($Quick) { 600 } else { 4000 }
$batch = 16
$seed = 7
$lr = 0.10

$includeDir = Join-Path $repoRoot "c\code\include"
$src = @(
    (Join-Path $appDir "ddpm_shapes.c"),
    (Join-Path $repoRoot "c\code\src\tensor.cu"),
    (Join-Path $repoRoot "c\code\src\cuda_utils.cu"),
    (Join-Path $repoRoot "c\code\src\params.cu"),
    (Join-Path $repoRoot "c\code\src\model.c"),
    (Join-Path $repoRoot "c\code\src\linear.c"),
    (Join-Path $repoRoot "c\code\src\loss.c"),
    (Join-Path $repoRoot "c\code\src\activation.c"),
    (Join-Path $repoRoot "c\code\src\activation_cpu.c"),
    (Join-Path $repoRoot "c\code\src\activation_cuda.cu"),
    (Join-Path $repoRoot "c\code\src\ops_add_sub.c"),
    (Join-Path $repoRoot "c\code\src\ops_add_sub_cpu.c"),
    (Join-Path $repoRoot "c\code\src\ops_add_sub_cuda.cu"),
    (Join-Path $repoRoot "c\code\src\ops_mul_div.c"),
    (Join-Path $repoRoot "c\code\src\ops_mul_div_cpu.c"),
    (Join-Path $repoRoot "c\code\src\ops_mul_div_cuda.cu"),
    (Join-Path $repoRoot "c\code\src\ops_matmul.c"),
    (Join-Path $repoRoot "c\code\src\ops_matmul_cpu.c"),
    (Join-Path $repoRoot "c\code\src\ops_matmul_cuda.cu")
)

Write-Host "[1/4] Building babyCTorch DDPM application..."
$nvccArgs = @("-O2", "-std=c++14", "-I$includeDir") + $src + @("-o", $exe)
& nvcc @nvccArgs
if ($LASTEXITCODE -ne 0) {
    throw "nvcc build failed with exit code $LASTEXITCODE"
}

Write-Host "[2/4] Training babyCTorch DDPM..."
& $exe --steps $steps --batch $batch --seed $seed --lr $lr --out-dir $resultsDir 1> $rawLog
if ($LASTEXITCODE -ne 0) {
    throw "babyCTorch DDPM run failed with exit code $LASTEXITCODE. See $rawLog"
}

Write-Host "[3/4] Training matched PyTorch reference..."
if ($python -eq "py") {
    & py -3 (Join-Path $appDir "reference.py") --steps $steps --batch $batch --seed $seed --lr $lr --out-dir $resultsDir
} else {
    & python (Join-Path $appDir "reference.py") --steps $steps --batch $batch --seed $seed --lr $lr --out-dir $resultsDir
}
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch reference failed with exit code $LASTEXITCODE"
}

Write-Host "[4/4] Verifying trainability..."
if ($python -eq "py") {
    & py -3 (Join-Path $appDir "verify.py")
} else {
    & python (Join-Path $appDir "verify.py")
}
$verifyExit = $LASTEXITCODE

Write-Host ""
Write-Host "babyCTorch summary"
Write-Host "------------------"
Get-Content (Join-Path $resultsDir "babyctorch_summary.txt")
Write-Host ""
Write-Host "Full outputs: $resultsDir"

exit $verifyExit
