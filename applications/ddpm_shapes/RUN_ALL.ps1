param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

function Import-VisualStudioBuildEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vsDevCmd = $null
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

    if (Test-Path $vswhere) {
        $installationPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1)
        if ($installationPath) {
            $candidate = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $candidate) {
                $vsDevCmd = $candidate
            }
        }
    }

    if (-not $vsDevCmd) {
        $roots = @(
            (Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022"),
            (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022")
        )
        foreach ($root in $roots) {
            if (-not (Test-Path $root)) { continue }
            $candidate = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { Join-Path $_.FullName "Common7\Tools\VsDevCmd.bat" } |
                Where-Object { Test-Path $_ } |
                Select-Object -First 1
            if ($candidate) {
                $vsDevCmd = $candidate
                break
            }
        }
    }

    if (-not $vsDevCmd) {
        throw @"
Visual Studio C++ build tools were not found.
Install Visual Studio 2022 or Build Tools 2022 with the workload 'Desktop development with C++', then rerun this script.
"@
    }

    Write-Host "Detected Visual Studio build environment: $vsDevCmd"

    # Import the environment produced by VsDevCmd.bat into this PowerShell process.
    # A temporary cmd file avoids fragile quoting around 'Program Files' paths.
    $tempCmd = Join-Path $env:TEMP ("babyctorch_vsenv_{0}.cmd" -f $PID)
    try {
        @"
@echo off
call "$vsDevCmd" -no_logo -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
set
"@ | Set-Content -LiteralPath $tempCmd -Encoding ASCII

        $envLines = & $env:ComSpec /d /c $tempCmd
        if ($LASTEXITCODE -ne 0) {
            throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
        }

        foreach ($line in $envLines) {
            if ($line -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $tempCmd -Force -ErrorAction SilentlyContinue
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Visual Studio was detected, but cl.exe is still unavailable after importing VsDevCmd.bat. Make sure the MSVC C++ x64/x86 build tools component is installed."
    }

    $clPath = (Get-Command cl.exe).Source
    Write-Host "Using MSVC compiler: $clPath"
}

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
    throw "nvcc was not found. Make sure the CUDA toolkit is installed and its bin directory is on PATH."
}

Import-VisualStudioBuildEnvironment

Write-Host "Using CUDA compiler: $((Get-Command nvcc).Source)"

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
