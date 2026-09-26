param(
    [string]$Config = "$PSScriptRoot/config.json",
    [string]$Python = "python",
    [string]$Node = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
)
$ErrorActionPreference = "Stop"
$repoDirectory = (Resolve-Path "$PSScriptRoot/../..").Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
# JSON contains physically distinct delta/Delta keys. Windows PowerShell's
# case-insensitive ConvertFrom-Json must not parse that scientific config.
$analysisOutput = & $Python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['output'])" $configPath
if ($LASTEXITCODE -ne 0) { throw "Cannot read the analysis configuration." }
$dependencyPath = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
if (-not (Test-Path -LiteralPath "$PSScriptRoot/node_modules")) {
    if (-not (Test-Path -LiteralPath $dependencyPath)) {
        throw "Bundled workbook dependencies are unavailable. Configure a Windows Node runtime with @oai/artifact-tool."
    }
    New-Item -ItemType Junction -Path "$PSScriptRoot/node_modules" -Target $dependencyPath | Out-Null
}
Push-Location -LiteralPath $repoDirectory
try {
    & $Python -m analysis.adaptive_protocol.run --config $configPath
    if ($LASTEXITCODE -ne 0) { throw "Numerical analysis failed; workbook was not replaced." }
    & $Node --max-old-space-size=10000 "$PSScriptRoot/export_workbook.mjs" $analysisOutput
    if ($LASTEXITCODE -ne 0) { throw "Workbook export failed; CSV results remain available." }
}
finally { Pop-Location }
