# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# ============================================================================
#  install.ps1 - clawcodex PowerShell installer
# ----------------------------------------------------------------------------
#  Builds on source hydration and adds pinned, checksum-verified dependency
#  installation, command registration, and update. Independently mergeable
#  diagnostics and uninstall functions land in the final staged PR.
#
#  Usage:
#      .\install.ps1 install
#      .\install.ps1 hydrate
#      .\install.ps1 status
#      .\install.ps1 doctor
#      .\install.ps1 verify
#      .\install.ps1 update
#      .\install.ps1 uninstall
#      .\install.ps1 help
# ============================================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('', 'install', 'hydrate', 'status', 'doctor', 'verify', 'update', 'uninstall', 'help')]
    [string]$Subcommand = '',

    # ---- Option flags (long form is preferred; aliases for parity with install.sh) ----
    [string]$Ref,
    [string]$InstallDir,
    [Alias('ConfigDir')]
    [string]$ConfigPath,
    [switch]$NoVenv,
    [switch]$NoSetup,
    [switch]$DryRun,
    [switch]$ForceSrc,
    [Alias('Yes')]
    [switch]$Force,
    [string]$LogFile,
    [switch]$Uninstall,
    [switch]$Help,
    [switch]$HelpZh,
    [switch]$Version
)

# ============================================================================
#  Strict mode + error preferences
# ============================================================================
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'
$WarningPreference     = 'Continue'

# Ensure TLS 1.2 is used for all network calls.  On Windows PowerShell 5.1 the
# default is TLS 1.0 which causes Invoke-WebRequest / Invoke-RestMethod to
# fail when connecting to GitHub / api.github.com.  PowerShell 7+ already
# defaults to TLS 1.2, but setting it explicitly does no harm.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

# Ensure console speaks UTF-8 so non-ASCII (e.g. Chinese help) renders correctly.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding           = [System.Text.Encoding]::UTF8
} catch {
    Write-Warning 'Could not set console encoding to UTF-8; non-ASCII output may not render correctly.'
}

# ============================================================================
#  Config (read-only defaults)
# ============================================================================
# --- Version and repository pins ---
# Versioning scheme — mirrored from install.sh:65-66.  The version is the CalVer
# date of the release.  To install a different clawcodex version, fetch the
# install.ps1 that ships with that release tag — same rule as the bash installer.
$script:InstallerVersion  = '2026.7.28'
$script:ClawCodexVersion  = '2026.7.28'
$script:RepoRef           = 'tech_v26.2.0'
$script:RepoUrl           = 'https://gitcode.com/Ascend/AgentSDK.git'
$script:ProductSubdir     = 'clawcodex-ascend'

# --- Source hydration pins ---
# Upstream source for src/ directory (Claude Code upstream fork).
# When src/ is not present in the repo, the installer pulls it from the
# upstream source at the pinned commit and applies the corresponding patches.
$script:UpstreamUrl      = 'https://github.com/agentforce314/clawcodex.git'
# UpstreamRef is NOT readonly — update this on each version sync to match the
# patches/upstream/<commit>/ directory.
$script:UpstreamRef      = '398b44f08f9de6dd36ab590d7d83799b34a28b3c'
$script:PatchSetId       = '398b44f'
$script:ExpectedPatchCount = 584
$script:UvVersion        = '0.11.16'
$script:UvWindowsSha256  = @{
    'aarch64' = 'e4f8e70eb21f0f4efd2eeb159ab289f9a16057d59881a4475758be4ce39bc8c5'
    'i686'    = '7417090298bf202395b9b3d6eefb9230332d8d6c94a5616e531148a0b041c8e2'
    'x86_64'  = 'dd9d6d6554bfab265bfa98aa8e8a406c5c3a7b97582f93de1f4d48d9154a0395'
}

# --- Installation paths and runtime prerequisites ---
# Overridable paths.  Resolved from $env:USERPROFILE so we work under both
# the SYSTEM and the interactive user context (the latter is the common case).
$script:DefaultInstallDir = Join-Path $env:USERPROFILE '.clawcodex\clawcodex'
$script:DefaultConfigDir  = Join-Path $env:USERPROFILE '.clawcodex'
$script:LocalBin          = Join-Path $env:USERPROFILE '.local\bin'
$script:PythonMinVersion  = '3.11'
$script:PythonMaxSupported = '3.13'
$script:EntryPoint        = 'clawcodex-dev'   # the single registered entry in pyproject.toml
$script:RcMarker          = '# clawcodex installer — managed by install.ps1'

# --- Invocation metadata ---
$invocationPath = $null
$pathProperty = $MyInvocation.MyCommand.PSObject.Properties['Path']
if ($pathProperty) {
    $invocationPath = [string]$pathProperty.Value
}
$script:SponsorScript     = if ($invocationPath -and
                                (Test-Path -LiteralPath $invocationPath) -and
                                $invocationPath -notlike "$env:TEMP\*" -and
                                $invocationPath -notlike "$env:LOCALAPPDATA\Temp\*") {
    $invocationPath
} else {
    'install.ps1'
}

# --- Runtime state ---
# Effective (post-override) paths.  Resolved in Initialize-Config below.
$script:ClawCodexHome      = $null
$script:ClawCodexParentDir = $null
$script:ProductHome        = $null
$script:ConfigDir          = $null
$script:UseVenv            = -not $NoVenv.IsPresent
$script:RunSetup           = -not $NoSetup.IsPresent
$script:AssumeYes          = $Force.IsPresent
$script:OS                 = 'unknown'
$script:ScriptStartTs      = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$script:StderrLogBuffer    = [System.Collections.Generic.List[string]]::new()

# ============================================================================
#  UI helpers
# ============================================================================
if ($Host.UI.SupportsVirtualTerminal) {
    # PowerShell 5.1 does not recognize the PowerShell 6+ `e escape sequence.
    $esc = [char]27
    $script:C_Red    = "${esc}[0;31m"
    $script:C_Green  = "${esc}[0;32m"
    $script:C_Yellow = "${esc}[1;33m"
    $script:C_Blue   = "${esc}[0;34m"
    $script:C_Bold   = "${esc}[1m"
    $script:C_Reset  = "${esc}[0m"
} else {
    $script:C_Red = ''; $script:C_Green = ''; $script:C_Yellow = ''
    $script:C_Blue = ''; $script:C_Bold = ''; $script:C_Reset = ''
}

# Agent-friendly line prefix.  Emitted only when stdout/stderr is not a TTY
# (i.e. when the script is being driven by another process, an agent, a CI
# runner, or a piped tee).  Interactive users see clean output.

function script:ScriptP1 { if ([Console]::IsOutputRedirected) { Write-Host '[install.ps1] ' -NoNewline } }

function script:Write-StderrLine {
    param(
        [AllowEmptyString()][string]$Message = '',
        [switch]$Prefix
    )
    $line = if ($Prefix -and [Console]::IsErrorRedirected) { "[install.ps1] $Message" } else { $Message }
    [Console]::Error.WriteLine($line)
    if ($LogFile) {
        $StderrLogBuffer.Add($line)
    }
}

function script:Flush-StderrLogBuffer {
    if (-not $LogFile -or $StderrLogBuffer.Count -eq 0) { return }

    $lines = $StderrLogBuffer.ToArray()
    $StderrLogBuffer.Clear()
    try {
        $lines | Add-Content -LiteralPath $LogFile -Encoding UTF8 -ErrorAction Stop
    } catch {
        # Do not recurse through Write-StderrLine when the log itself is broken.
        [Console]::Error.WriteLine("[install.ps1] Could not append stderr to log file: $LogFile")
    }
}

function script:Log-Info { param($Msg) ScriptP1; Write-Host "${C_Blue}==>${C_Reset} ${C_Bold}$Msg${C_Reset}" }

function script:Log-Ok   { param($Msg) ScriptP1; Write-Host "  ${C_Green}✓${C_Reset} $Msg" }

function script:Log-Warn { param($Msg) ScriptP1; Write-Host "  ${C_Yellow}!${C_Reset} $Msg" }

function script:Log-Err  { param($Msg) Write-StderrLine -Message "${C_Red}✗${C_Reset} $Msg" -Prefix }

function script:Log-Step { param($Msg) ScriptP1; Write-Host "`n${C_Bold}${C_Blue}>>>${C_Reset} ${C_Bold}$Msg${C_Reset}" }

function script:Write-ExitSummary {
    param([int]$Rc)
    $elapsed = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $ScriptStartTs
    if ($Rc -eq 0) {
        Write-StderrLine -Message "DONE: SUCCESS (exit 0) after ${elapsed}s" -Prefix
        if ($LogFile) {
            Write-StderrLine -Message "DONE: full log at: $LogFile" -Prefix
        }
    } else {
        Write-StderrLine -Message "DONE: FAILED (exit $Rc) after ${elapsed}s" -Prefix
        if ($LogFile) {
            Write-StderrLine -Message "DONE: failure log saved to: $LogFile" -Prefix
        } else {
            Write-StderrLine -Message 'DONE: re-run with -LogFile <path> to capture full output.' -Prefix
        }
    }
}

function script:Run-OrDry {
    param([scriptblock]$Block, [string]$WhatIfText)
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would run: $WhatIfText"
        return
    }
    & $Block
}

function script:Die-With-Help {
    param(
        [Parameter(Mandatory, Position = 0)][string]$Header,
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$NextSteps
    )
    Log-Err $Header
    if ($NextSteps -and $NextSteps.Count -gt 0) {
        Write-StderrLine
        Write-StderrLine -Message '  Next steps to try:'
        foreach ($step in $NextSteps) { Write-StderrLine -Message "    -> $step" }
    }
    Write-StderrLine
    Write-StderrLine -Message "  For diagnosis, run:    $($SponsorScript) doctor"
    Write-StderrLine -Message "  For full usage, run:    $($SponsorScript) -Help"
    $script:rc = 1
    $handled = [System.InvalidOperationException]::new($Header)
    $handled.Data['ClawCodexInstallerHandled'] = $true
    throw $handled
}

function script:Detect-OS {
    # $IsWindows / $IsLinux / $IsMacOS are PowerShell 6+ automatic vars.
    # We probe them defensively so the script also runs on 5.1.
    $isWindows = $false
    $isLinux   = $false

    try {
        if (Get-Variable -Name 'IsWindows' -ErrorAction SilentlyContinue) { $isWindows = [bool]$IsWindows }
        if (Get-Variable -Name 'IsLinux'   -ErrorAction SilentlyContinue) { $isLinux   = [bool]$IsLinux }
    } catch { }

    if (-not $isWindows -and -not $isLinux) {
        # PowerShell 5.1 fallback — check the OS environment variable.
        $osName = $env:OS
        if ($osName -eq 'Windows_NT') { $isWindows = $true }
    }

    if ($isWindows) {
        # Detect WSL: the WSL interop exposes C:\Windows\System32\wsl.exe and
        # the WSL_DISTRO_NAME env var is set inside WSL-hosted PowerShell.
        if ($env:WSL_DISTRO_NAME) { return 'wsl' }
        if (Test-Path 'C:\Windows\System32\wsl.exe') {
            # WSL binary is present — could be native Win with WSL optional feature.
            # We only flip to 'wsl' if the WSL_DISTRO_NAME is set, otherwise treat
            # as plain windows.
        }
        return 'windows'
    }
    if ($isLinux) { return 'linux' }
    return 'unknown'
}

function script:Get-OsInstallHint {
    param([string]$OsType)
    switch ($OsType) {
        'windows' {
            '    Install Git for Windows:   https://git-scm.com/download/win'
            '    Or use winget:             winget install Git.Git'
            '    Or use Chocolatey:         choco install git'
        }
        'wsl' {
            '    You are inside WSL — install Git in your Linux distro:'
            '        Debian/Ubuntu : sudo apt update && sudo apt install -y git'
            '        Fedora/RHEL   : sudo dnf install -y git'
            '        Arch          : sudo pacman -S --noconfirm git'
        }
        default { '    install git via your package manager' }
    }
}

function script:Get-OsInstallHintOneLiner {
    param([string]$OsType)
    switch ($OsType) {
        'windows' { 'winget install Git.Git   (or: https://git-scm.com/download/win)' }
        'wsl'     { 'sudo apt install -y git   (or your distro package manager)' }
        default   { 'install git via your package manager' }
    }
}

function script:Check-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        $installHints = @(Get-OsInstallHint $OS)
        Die-With-Help 'Git is not installed.' $installHints
    }
    $version = & git --version
    Log-Ok $version
}

function script:Install-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        $uvVer = (& uv --version) -replace '^uv\s+', ''
        Log-Ok "uv $uvVer already installed"
        return
    }

    Log-Info 'Installing uv (user-local, no admin) ...'

    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would install uv $UvVersion via winget / verified GitHub binary"
        return
    }

    # Strategy 1: winget (available on Windows 10/11 by default)
    Log-Info 'Trying winget install ...'
    $wingetOk = $false
    try {
        $wingetCmd = Get-Command winget -ErrorAction Stop
        $wingetOk = $true
    } catch {
        Log-Warn 'winget not available — skipping to next method'
    }

    if ($wingetOk) {
        try {
            $env:Path = "$LocalBin;$((Join-Path $env:USERPROFILE '.cargo\bin'));$env:Path"
            $installArgs = @('install', '--id', 'AstralIndustries.uv', '--version', $UvVersion,
                             '--accept-package-agreements', '--scope', 'user', '--source', 'winget', '-e')
            & winget $installArgs 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $env:Path = "$LocalBin;$((Join-Path $env:USERPROFILE '.cargo\bin'));$env:Path"
                $uv = Get-Command uv -ErrorAction SilentlyContinue
                if ($uv) {
                    $uvVer = (& uv --version) -replace '^uv\s+', ''
                    Log-Ok "uv $uvVer installed via winget"
                    return
                }
            }
        } catch {
            Log-Warn "winget install failed: $_"
        }
    }

    # Strategy 2: Direct binary download from GitHub releases
    Log-Info 'Downloading uv from GitHub releases ...'
    $uvInstallDir = Join-Path $env:USERPROFILE '.local'
    $zipPath = $null
    try {
        # Detect architecture
        $arch = 'x86_64'
        if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $arch = 'aarch64' }
        elseif ($env:PROCESSOR_ARCHITECTURE -eq 'ARM') { $arch = 'aarch64' }
        elseif ($env:PROCESSOR_ARCHITECTURE -eq 'x86') { $arch = 'i686' }

        $assetName = "uv-${arch}-pc-windows-msvc.zip"
        $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$assetName"
        $zipPath = Join-Path $env:TEMP "uv-$UvVersion-${arch}.zip"
        $extractDir = Join-Path $uvInstallDir 'bin'

        # Download the zip
        Log-Info "  URL: $url"
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing -ErrorAction Stop | Out-Null

        $expectedHash = $UvWindowsSha256[$arch]
        $actualHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        if (-not $expectedHash -or $actualHash -ne $expectedHash) {
            throw "SHA256 mismatch for $assetName (expected $expectedHash, got $actualHash)"
        }
        Log-Ok "Verified uv $UvVersion $assetName SHA256"

        # Extract
        if (-not (Test-Path -LiteralPath $extractDir)) {
            New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
        }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        # Cleanup
        Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue

        # Verify
        $uvExe = Join-Path $extractDir 'uv.exe'
        if (Test-Path -LiteralPath $uvExe -PathType Leaf) {
            $env:Path = "${extractDir};$env:Path"
            $uvVer = (& $uvExe --version) -replace '^uv\s+', ''
            if ($uvVer -ne $UvVersion) {
                throw "Downloaded uv version mismatch (expected $UvVersion, got $uvVer)"
            }
            Log-Ok "uv $uvVer installed (GitHub binary)"
            return
        }
    } catch {
        Log-Warn "GitHub binary download failed: $_"
        # Cleanup partial files (zipPath may be null if failure occurred early)
        if ($zipPath -and (Test-Path -LiteralPath $zipPath)) {
            try { Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue } catch { }
        }
    }

    # If all strategies fail, give the user a clear error with manual install steps
    Die-With-Help 'All uv installation methods failed.' `
        'Manual install (recommended):  iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex' `
        "Or download:     https://github.com/astral-sh/uv/releases" `
        "Then add to PATH: $LocalBin" `
        "Retry:           $SponsorScript"
}

function script:Ensure-Python {
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would check for Python $PythonMinVersion+ via uv"
        return
    }

    $py = $null
    try {
        $py = & uv python find $PythonMinVersion 2>$null
    } catch { $py = $null }

    if ($py -and (Test-Path $py)) {
        $ver = & $py --version 2>&1
        Log-Ok "$ver"
        return
    }

    Log-Info "Python $PythonMinVersion+ not found — provisioning via uv (no admin) ..."
    & uv python install $PythonMinVersion
    if ($LASTEXITCODE -ne 0) {
        Die-With-Help "Failed to install Python $PythonMinVersion via uv." `
            "Retry:    $SponsorScript" `
            "Manual:   uv python install $PythonMinVersion" `
            'Or:       install Python 3.11+ from https://python.org'
    }

    $py = & uv python find $PythonMinVersion 2>$null
    if (-not $py -or -not (Test-Path $py)) {
        Die-With-Help "Python $PythonMinVersion still not found after uv install." `
            "Retry:    $SponsorScript" `
            "Diagnose: $SponsorScript doctor"
    }
    $ver = & $py --version 2>&1
    Log-Ok "$ver"
}

function script:Ensure-Python-Pyo3Compat {
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would check Python <= $PythonMaxSupported for pyo3-ffi compatibility"
        return
    }

    $py = $null
    try { $py = & uv python find $PythonMinVersion 2>$null } catch { $py = $null }
    if (-not $py -or -not (Test-Path $py)) { return }

    $verStr = & $py --version 2>&1
    if ($verStr -match 'Python\s+(\d+)\.(\d+)') {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
    } else {
        return
    }

    # No mitigation needed for Python <= 3.13
    if ($major -lt 3 -or ($major -eq 3 -and $minor -le 13)) { return }

    Log-Warn "Python $major.$minor detected, but pyo3-ffi (used by outlines-core) only supports up to $PythonMaxSupported"
    Log-Info "Searching for a compatible Python ($PythonMinVersion - $PythonMaxSupported)..."

    foreach ($target in @('3.13', '3.12', '3.11')) {
        $compatPy = $null
        try { $compatPy = & uv python find $target 2>$null } catch { $compatPy = $null }
        if ($compatPy -and (Test-Path $compatPy)) {
            $compatVer = & $compatPy --version 2>&1
            Log-Ok "Found compatible Python: $compatPy ($compatVer)"
            $env:UV_PYTHON = $compatPy
            return
        }
    }

    # No compatible interpreter locally — try uv provisioning
    Log-Info "No compatible Python found locally — provisioning Python $PythonMaxSupported via uv..."
    $installed = $false
    try {
        & uv python install $PythonMaxSupported 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $installed = $true }
    } catch { $installed = $false }

    if ($installed) {
        $compatPy = $null
        try { $compatPy = & uv python find $PythonMaxSupported 2>$null } catch { $compatPy = $null }
        if ($compatPy -and (Test-Path $compatPy)) {
            $compatVer = & $compatPy --version 2>&1
            Log-Ok "Python $PythonMaxSupported installed via uv ($compatPy)"
            $env:UV_PYTHON = $compatPy
            return
        }
    }

    # Auto-mitigation failed — defensive error with clear next steps
    Die-With-Help "Python $major.$minor is incompatible with clawcodex dependencies" `
        "The detected Python ($major.$minor) is too new for pyo3-ffi (max $PythonMaxSupported)." `
        "outlines-core will fail to compile with Rust errors." `
        "" `
        "Recommended fixes (pick one):" `
        "  1. uv python install 3.13   (then re-run install.ps1)" `
        "  2. pyenv install 3.13 && pyenv local 3.13" `
        "  3. Install Python 3.11-3.13 from https://python.org" `
        "  4. Set `$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = 1` and retry (risky)" `
        "The installer attempted to auto-provision a compatible Python but failed."
}

function script:Write-GitFailureOutput {
    param([object[]]$CommandOutput)
    $lines = @($CommandOutput | Select-Object -Last 10)
    if ($lines.Count -eq 0) { return }
    Write-StderrLine -Message 'Git output:'
    foreach ($line in $lines) {
        Write-StderrLine -Message "  $line"
    }
}

function script:Get-CanonicalPath {
    param([Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'Path cannot be empty or whitespace.'
    }
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPath = [System.IO.Path]::GetPathRoot($fullPath)
    $trimChars = [char[]]@('\', '/')
    if ($fullPath.TrimEnd($trimChars) -ieq $rootPath.TrimEnd($trimChars)) {
        return $rootPath
    }
    return $fullPath.TrimEnd($trimChars)
}

function script:Assert-SafeInstallTarget {
    if ($ClawCodexHome -match '^[A-Za-z]:$') {
        Die-With-Help "Refusing drive-relative install target: $ClawCodexHome" `
            'Use an absolute dedicated directory such as C:\Apps\clawcodex.'
    }
    $installFull = Get-CanonicalPath $ClawCodexHome
    $rootFull = Get-CanonicalPath ([System.IO.Path]::GetPathRoot($installFull))
    $userFull = Get-CanonicalPath $env:USERPROFILE
    $configFull = Get-CanonicalPath $ConfigDir
    $localBinFull = Get-CanonicalPath $LocalBin
    $separator = [System.IO.Path]::DirectorySeparatorChar
    if (
        $installFull -ieq $rootFull -or
        $installFull -ieq $userFull -or
        $userFull.StartsWith("$installFull$separator", [System.StringComparison]::OrdinalIgnoreCase) -or
        $configFull -ieq $installFull -or
        $configFull.StartsWith("$installFull$separator", [System.StringComparison]::OrdinalIgnoreCase) -or
        $localBinFull -ieq $installFull -or
        $localBinFull.StartsWith("$installFull$separator", [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        Die-With-Help "Refusing unsafe install target: $installFull" `
            'Choose a dedicated checkout directory with -InstallDir.' `
            'It must not contain the user profile, config directory, or local command directory.'
    }
}

function script:Get-InstallOwnershipMarker {
    return (Join-Path (Join-Path $ClawCodexHome '.git') 'clawcodex-installer.json')
}

function script:Write-InstallOwnershipMarker {
    $markerFile = Get-InstallOwnershipMarker
    [ordered]@{
        schema_version  = 2
        created_by     = 'clawcodex-ascend/install.ps1'
        install_dir    = (Get-CanonicalPath $ClawCodexHome)
        product_subdir = $ProductSubdir
        repo_url       = $RepoUrl
        user_path_added = $false
        created_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $markerFile -Encoding UTF8 -ErrorAction Stop
}

function script:Set-InstallUserPathOwnership {
    param([Parameter(Mandatory)][bool]$Added)

    $markerFile = Get-InstallOwnershipMarker
    try {
        $marker = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
        $marker.schema_version = 2
        $marker | Add-Member -NotePropertyName user_path_added -NotePropertyValue $Added -Force
        $marker | ConvertTo-Json | Set-Content -LiteralPath $markerFile -Encoding UTF8 -ErrorAction Stop
    } catch {
        Log-Warn "Could not record User PATH ownership in $markerFile; uninstall will preserve the PATH entry."
    }
}

function script:Test-InstallerOwnedCheckout {
    $markerFile = Get-InstallOwnershipMarker
    if (-not (Test-Path -LiteralPath $markerFile -PathType Leaf)) { return $false }
    try {
        $marker = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json
        $canonicalInstall = Get-CanonicalPath $ClawCodexHome
        $originUrl = (& git -c "safe.directory=$canonicalInstall" -C $ClawCodexHome remote get-url origin 2>$null) -join ''
        if ($LASTEXITCODE -ne 0) { return $false }
        return (
            $marker.created_by -eq 'clawcodex-ascend/install.ps1' -and
            $marker.install_dir -ieq (Get-CanonicalPath $ClawCodexHome) -and
            $marker.product_subdir -eq $ProductSubdir -and
            $marker.repo_url -eq $RepoUrl -and
            $originUrl -match '(?i)gitcode\.com[:/][^/]+/[^/]*AgentSDK[^/]*/?$'
        )
    } catch {
        return $false
    }
}

function script:Clone-OrUpdate-Repo {
    Assert-SafeInstallTarget
    $gitDir = Join-Path $ClawCodexHome '.git'
    if (Test-Path $gitDir) {
        Log-Info "Existing AgentSDK checkout found at $ClawCodexHome — validating ref $RepoRef ..."
        Push-Location $ClawCodexHome
        try {
            $originUrl = (& git remote get-url origin 2>$null) -join ''
            if ($originUrl -notmatch '(?i)gitcode\.com[:/][^/]+/[^/]*AgentSDK[^/]*/?$') {
                Die-With-Help "Existing checkout does not point to the supported AgentSDK origin: $originUrl" `
                    "Choose another -InstallDir, or move the existing checkout out of the way." `
                    "Expected origin: $RepoUrl"
            }
            if ($DryRun) {
                ScriptP1
                Write-Host "[DRY-RUN] would fetch and fast-forward exactly to: $RepoUrl $RepoRef"
                return
            }

            $gitFetchOutput = @(& git fetch --depth 100 $RepoUrl $RepoRef 2>&1)
            $gitFetchRc = $LASTEXITCODE
            if ($gitFetchRc -ne 0) {
                Write-GitFailureOutput $gitFetchOutput
                Die-With-Help "Failed to fetch AgentSDK ref '$RepoRef'." `
                    "Verify: git ls-remote $RepoUrl $RepoRef"
            }
            $targetSha = (& git rev-parse FETCH_HEAD 2>$null) -join ''
            $currentSha = (& git rev-parse HEAD 2>$null) -join ''
            if (-not $targetSha -or -not $currentSha) {
                Die-With-Help 'Cannot resolve the current or requested AgentSDK commit.'
            }
            if ($currentSha -ne $targetSha) {
                & git merge-base --is-ancestor $currentSha $targetSha 2>$null
                if ($LASTEXITCODE -ne 0) {
                    $isShallow = ((& git rev-parse --is-shallow-repository 2>$null) -join '') -eq 'true'
                    if ($isShallow) {
                        Log-Info 'Requested ref is beyond the shallow history; fetching full ancestry ...'
                        $deepenOutput = @(& git fetch --unshallow $RepoUrl $RepoRef 2>&1)
                        if ($LASTEXITCODE -ne 0) {
                            Write-GitFailureOutput $deepenOutput
                            Die-With-Help "Failed to deepen the AgentSDK checkout for '$RepoRef'."
                        }
                        $targetSha = (& git rev-parse FETCH_HEAD 2>$null) -join ''
                    }
                }
                $gitMergeOutput = @(& git merge --ff-only FETCH_HEAD 2>&1)
                $gitMergeRc = $LASTEXITCODE
                if ($gitMergeRc -ne 0) {
                    Write-GitFailureOutput $gitMergeOutput
                    Die-With-Help "Cannot fast-forward the install checkout to '$RepoRef'." `
                        "Local commits or edits may be present; choose a clean -InstallDir."
                }
            }
            $resolvedSha = (& git rev-parse HEAD 2>$null) -join ''
            if ($resolvedSha -ne $targetSha) {
                Die-With-Help "Install checkout did not resolve exactly to '$RepoRef'." `
                    "Expected: $targetSha" `
                    "Actual:   $resolvedSha"
            }
            Log-Ok "Resolved AgentSDK ref $RepoRef at $targetSha"
        } finally {
            Pop-Location
        }
        if (-not (Test-Path $ProductHome)) {
            Die-With-Help "Product directory is missing after update: $ProductHome" `
                "Verify that '$RepoRef' contains '$ProductSubdir/'." `
                "Retry with: $SponsorScript update"
        }
        return
    }

    if (Test-Path $ClawCodexHome) {
        Die-With-Help "Install target already exists and is not an AgentSDK Git checkout: $ClawCodexHome" `
            'Choose an empty/nonexistent -InstallDir.' `
            'The installer will not move or overwrite an existing directory.'
    }

    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would create parent dir: $ClawCodexParentDir"
        ScriptP1
        Write-Host "[DRY-RUN] would sparse-clone: $RepoUrl (ref: $RepoRef, path: $ProductSubdir) -> $ClawCodexHome"
        return
    }

    if (-not (Test-Path $ClawCodexParentDir)) {
        New-Item -ItemType Directory -Path $ClawCodexParentDir -Force -ErrorAction Stop | Out-Null
    }

    Log-Info "Sparse-cloning $RepoUrl (ref: $RepoRef, path: $ProductSubdir) -> $ClawCodexHome"

    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $cloneArgs = @(
        'clone', '--depth', '1', '--filter=blob:none', '--sparse',
        '--branch', $RepoRef, $RepoUrl, $ClawCodexHome
    )
    $cloneOut = @(& git @cloneArgs 2>&1)
    $cloneExit = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP
    if ($cloneExit -ne 0) {
        Write-GitFailureOutput $cloneOut
        Die-With-Help "Failed to clone AgentSDK ref '$RepoRef'." `
            'Check your network connection.' `
            "Verify:  git ls-remote $RepoUrl $RepoRef" `
            "Retry:   $SponsorScript" `
            "Diagnose: $SponsorScript doctor"
    }

    Push-Location $ClawCodexHome
    try {
        & git sparse-checkout set $ProductSubdir 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Die-With-Help "Failed to configure sparse checkout for '$ProductSubdir'." `
                "Retry after removing the incomplete checkout: $ClawCodexHome"
        }
    } finally {
        Pop-Location
    }

    if (-not (Test-Path $ProductHome)) {
        Die-With-Help "Product directory not found in AgentSDK ref '$RepoRef': $ProductSubdir" `
            "Verify: git ls-tree $RepoRef $ProductSubdir"
    }
    try {
        Write-InstallOwnershipMarker
    } catch {
        Die-With-Help 'AgentSDK was cloned, but installer ownership could not be recorded.' `
            "Checkout left in place for inspection: $ClawCodexHome" `
            "Error: $_"
    }
    Log-Ok "Cloned AgentSDK ref $RepoRef and materialized $ProductSubdir/"
}

function script:Get-ValidatedPatchSeries {
    $patchBase = Join-Path $ProductHome "patches/upstream/$PatchSetId"
    $patchDir = Join-Path $patchBase 'merged'
    $seriesFile = Join-Path $patchBase 'series'

    if (-not (Test-Path -LiteralPath $seriesFile -PathType Leaf)) {
        Die-With-Help "Patch series is missing: $seriesFile" `
            "The clawcodex-ascend patch migration is incomplete."
    }
    if (-not (Test-Path -LiteralPath $patchDir -PathType Container)) {
        Die-With-Help "Patch directory is missing: $patchDir" `
            "The clawcodex-ascend patch migration is incomplete."
    }

    $entries = [System.Collections.Generic.List[object]]::new()
    $seen = @{}
    foreach ($rawLine in Get-Content -LiteralPath $seriesFile -Encoding UTF8) {
        $name = $rawLine.Trim()
        if (-not $name -or $name.StartsWith('#')) { continue }
        $expectedPrefix = '{0:D4}.' -f ($entries.Count + 1)
        if (
            $name -notmatch '^[0-9]{4}\.[A-Za-z0-9_.-]+\.patch$' -or
            [System.IO.Path]::GetFileName($name) -ne $name -or
            -not $name.StartsWith($expectedPrefix)
        ) {
            Die-With-Help "Unsafe or invalid patch entry in series: $name" `
                "Series entries must be plain .patch filenames numbered continuously from 0001."
        }
        if ($seen.ContainsKey($name)) {
            Die-With-Help "Duplicate patch entry in series: $name"
        }
        $seen[$name] = $true
        $entries.Add([pscustomobject]@{
            Name = $name
            Path = Join-Path $patchDir $name
        })
    }

    if ($entries.Count -ne $ExpectedPatchCount) {
        Die-With-Help "Patch series count mismatch: expected $ExpectedPatchCount, found $($entries.Count)." `
            "Verify: $seriesFile"
    }

    $missing = @($entries | Where-Object { -not (Test-Path -LiteralPath $_.Path -PathType Leaf) })
    if ($missing.Count -gt 0) {
        $preview = ($missing | Select-Object -First 5 | ForEach-Object Name) -join ', '
        Die-With-Help "Patch payload is incomplete: $($missing.Count) series file(s) are missing." `
            "First missing entries: $preview" `
            "Wait for the remaining patch migration PRs, then retry."
    }

    $extra = @(
        Get-ChildItem -LiteralPath $patchDir -Filter '*.patch' -File |
            Where-Object { -not $seen.ContainsKey($_.Name) }
    )
    if ($extra.Count -gt 0) {
        $preview = ($extra | Select-Object -First 5 -ExpandProperty Name) -join ', '
        Die-With-Help "Patch directory contains $($extra.Count) file(s) not declared by series." `
            "First extra entries: $preview"
    }

    # Parse each payload without requiring a target tree.  Applicability is
    # checked later against the pinned upstream tree with git apply --check.
    foreach ($entry in $entries) {
        & git apply --numstat $entry.Path 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Die-With-Help "Malformed patch payload: $($entry.Name)" `
                "Regenerate the patch as a valid unified diff before retrying."
        }
    }

    return ,$entries.ToArray()
}

function script:Get-PatchPayloadDigest {
    param([object[]]$Patches)

    $manifest = New-Object System.Text.StringBuilder
    foreach ($patch in $Patches) {
        $hash = (Get-FileHash -LiteralPath $patch.Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        [void]$manifest.Append($patch.Name).Append("`n").Append($hash).Append("`n")
    }
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($manifest.ToString())
        return ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

function script:Setup-UpstreamSrc {
    param([switch]$RebuildIfStale)

    $srcDir = Join-Path $ProductHome 'src'
    $markerFile = Join-Path $ProductHome '.clawcodex-source.json'
    $seriesFile = Join-Path $ProductHome "patches/upstream/$PatchSetId/series"

    # A fresh dry-run does not clone AgentSDK, so there may be no local payload
    # to inspect.  When the checkout is already present, validate it even in
    # dry-run mode rather than reporting a misleading "would validate" result.
    $canValidateLocalPayload = Test-Path -LiteralPath $seriesFile -PathType Leaf
    if ($DryRun -and -not $canValidateLocalPayload) {
        Log-Info "Hydrating src/ from upstream $UpstreamRef + patch set $PatchSetId ..."
        Log-Warn 'Patch payload is not available locally; this fresh dry-run can only preview the hydration plan.'
        ScriptP1
        Write-Host "[DRY-RUN] would clone: $UpstreamUrl (ref: $UpstreamRef) -> temp"
        ScriptP1
        Write-Host "[DRY-RUN] would require exactly $ExpectedPatchCount entries from patches/upstream/$PatchSetId/series"
        ScriptP1
        Write-Host "[DRY-RUN] would apply patches in series order and atomically replace: $srcDir"
        return
    }

    $patches = Get-ValidatedPatchSeries
    $seriesHash = (Get-FileHash -LiteralPath $seriesFile -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
    $patchPayloadHash = Get-PatchPayloadDigest -Patches $patches

    if ($DryRun) {
        Log-Ok "Validated local patch payload: $($patches.Count) patches from $PatchSetId"
        ScriptP1
        Write-Host "[DRY-RUN] would clone: $UpstreamUrl (ref: $UpstreamRef) -> temp"
        ScriptP1
        Write-Host "[DRY-RUN] would apply patches in series order and atomically replace: $srcDir"
        return
    }

    if ((Test-Path $srcDir) -and -not $ForceSrc) {
        if (Test-Path -LiteralPath $markerFile -PathType Leaf) {
            try {
                $marker = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 | ConvertFrom-Json
                if (
                    $marker.upstream_commit -eq $UpstreamRef -and
                    $marker.patch_set -eq $PatchSetId -and
                    [int]$marker.patch_count -eq $ExpectedPatchCount -and
                    $marker.series_sha256 -eq $seriesHash -and
                    $marker.patch_payload_sha256 -eq $patchPayloadHash
                ) {
                    Log-Ok "src/ already hydrated from $PatchSetId ($ExpectedPatchCount patches)"
                    return
                }
            } catch {
                Log-Warn "Cannot read source marker: $markerFile"
            }
        }
        if (-not $RebuildIfStale) {
            Die-With-Help 'src/ exists but its upstream/patch identity cannot be verified.' `
                "Use -ForceSrc to rebuild it from the locked upstream commit."
        }
        Log-Warn 'Rebuilding src/ because update must revalidate the current patch queue.'
    }

    Log-Info "Hydrating src/ from upstream $UpstreamRef + patch set $PatchSetId ..."
    $upstreamTmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Path $upstreamTmp -Force | Out-Null
    try {
        Log-Info "Cloning $UpstreamUrl (ref: $UpstreamRef) ..."
        & git -C $upstreamTmp init 2>&1 | Out-Null
        & git -C $upstreamTmp remote add origin $UpstreamUrl 2>&1 | Out-Null
        & git -C $upstreamTmp fetch --depth 1 origin $UpstreamRef 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Die-With-Help 'Failed to fetch the locked upstream commit.' `
                'Check your network connection.' `
                "Verify:  git ls-remote $UpstreamUrl $UpstreamRef" `
                "Retry:   $SponsorScript"
        }
        & git -C $upstreamTmp checkout --detach FETCH_HEAD 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Die-With-Help "Failed to checkout upstream commit '$UpstreamRef'."
        }

        $upstreamSrc = Join-Path $upstreamTmp 'src'
        if (-not (Test-Path $upstreamSrc)) {
            Die-With-Help 'src/ directory not found in upstream source.' `
                "Verify:  Get-ChildItem $upstreamTmp" `
                'The upstream repo may have changed its layout.'
        }

        Log-Info "Checking $($patches.Count) patches in series order ..."
        foreach ($patch in $patches) {
            # git apply discovers the repository root even when invoked from
            # upstreamTmp/src.  Prefix paths explicitly so a/foo.py is written
            # to upstreamTmp/src/foo.py rather than upstreamTmp/foo.py.
            & git -C $upstreamTmp apply --check -p1 --directory=src -- $patch.Path 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Die-With-Help "Patch preflight failed: $($patch.Name)" `
                    "No changes were written to the install directory."
            }
            & git -C $upstreamTmp apply -p1 --directory=src -- $patch.Path 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Die-With-Help "Patch application failed after preflight: $($patch.Name)" `
                    "No changes were written to the install directory."
            }
        }

        # Copy the fully patched tree into a same-volume staging directory
        # before touching the currently installed src/.  The subsequent moves
        # are same-volume renames, so failure can restore the old tree.
        $transactionId = [guid]::NewGuid().ToString('N')
        $srcStaging = Join-Path $ProductHome ".src.staging.$transactionId"
        $srcBackup = Join-Path $ProductHome ".src.backup.$transactionId"
        $markerTmp = "$markerFile.tmp.$transactionId"
        $markerBackup = "$markerFile.backup.$transactionId"
        $oldSrcMoved = $false
        $oldMarkerMoved = $false
        $newSrcInstalled = $false
        $newMarkerInstalled = $false
        try {
            Copy-Item -LiteralPath $upstreamSrc -Destination $srcStaging -Recurse -ErrorAction Stop
            [ordered]@{
                schema_version       = 2
                upstream_url         = $UpstreamUrl
                upstream_commit      = $UpstreamRef
                patch_set            = $PatchSetId
                patch_count          = $patches.Count
                series_sha256        = $seriesHash
                patch_payload_sha256 = $patchPayloadHash
                hydrated_at_utc      = [DateTime]::UtcNow.ToString('o')
            } | ConvertTo-Json | Set-Content -LiteralPath $markerTmp -Encoding UTF8 -ErrorAction Stop

            if (Test-Path -LiteralPath $srcDir) {
                Move-Item -LiteralPath $srcDir -Destination $srcBackup -ErrorAction Stop
                $oldSrcMoved = $true
            }
            if (Test-Path -LiteralPath $markerFile) {
                Move-Item -LiteralPath $markerFile -Destination $markerBackup -ErrorAction Stop
                $oldMarkerMoved = $true
            }
            Move-Item -LiteralPath $srcStaging -Destination $srcDir -ErrorAction Stop
            $newSrcInstalled = $true
            Move-Item -LiteralPath $markerTmp -Destination $markerFile -ErrorAction Stop
            $newMarkerInstalled = $true
        } catch {
            if ($newMarkerInstalled -and (Test-Path -LiteralPath $markerFile)) {
                Remove-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue
            }
            if ($oldMarkerMoved -and (Test-Path -LiteralPath $markerBackup)) {
                Move-Item -LiteralPath $markerBackup -Destination $markerFile -ErrorAction SilentlyContinue
            }
            if ($newSrcInstalled -and (Test-Path -LiteralPath $srcDir)) {
                Remove-Item -LiteralPath $srcDir -Recurse -Force -ErrorAction SilentlyContinue
            }
            if ($oldSrcMoved -and (Test-Path -LiteralPath $srcBackup)) {
                Move-Item -LiteralPath $srcBackup -Destination $srcDir -ErrorAction SilentlyContinue
            }
            throw
        } finally {
            foreach ($leftover in @($srcStaging, $markerTmp)) {
                if (Test-Path -LiteralPath $leftover) {
                    Remove-Item -LiteralPath $leftover -Recurse -Force -ErrorAction SilentlyContinue
                }
            }
        }
        foreach ($backup in @($srcBackup, $markerBackup)) {
            if (Test-Path -LiteralPath $backup) {
                try {
                    Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction Stop
                } catch {
                    Log-Warn "Installed successfully, but could not remove backup: $backup"
                }
            }
        }
        Log-Ok "Hydrated src/ with $($patches.Count) patches from $PatchSetId"
    } finally {
        Remove-Item -LiteralPath $upstreamTmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function script:Assert-RuntimePayload {
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would require pyproject.toml, uv.lock, and clawcodex_ext/ in $ProductHome"
        return
    }

    $required = @(
        'pyproject.toml',
        'uv.lock',
        'clawcodex_ext'
    )
    $missing = @(
        $required | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $ProductHome $_))
        }
    )
    if ($missing.Count -gt 0) {
        Die-With-Help "clawcodex-ascend runtime payload migration is incomplete." `
            "Missing: $($missing -join ', ')" `
            "The installer is intentionally fail-closed until these files are migrated." `
            "Source-only verification remains available with: $SponsorScript hydrate"
    }
}

function script:Ensure-Local-EnvFile {
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would create $ProductHome\.env from .env.example if missing"
        return
    }

    $envFile     = Join-Path $ProductHome '.env'
    $envExample  = Join-Path $ProductHome '.env.example'

    if (Test-Path $envFile) {
        Log-Ok 'Local .env already exists (not modified)'
        return
    }

    if (Test-Path $envExample) {
        Copy-Item -LiteralPath $envExample -Destination $envFile -Force
    } else {
        $template = @(
            '# Local F-73 release credentials. Never commit real token values.'
            'GITCODE_TOKEN='
            'TEST_PYPI_TOKEN='
            '# PYPI_TOKEN='
            'GITCODE_OWNER='
            'GITCODE_REPO='
            'GITCODE_API_ROOT=https://api.gitcode.com'
        )
        Set-Content -LiteralPath $envFile -Value $template -Encoding UTF8
    }

    # 600 equivalent: remove inheritance and grant only the current user.
    try {
        $acl = Get-Acl -LiteralPath $envFile
        $acl.SetAccessRuleProtection($true, $false)
        foreach ($accessRule in @($acl.Access)) {
            [void]$acl.RemoveAccessRuleAll($accessRule)
        }
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $currentUser, 'FullControl', 'Allow')
        [void]$acl.AddAccessRule($rule)
        Set-Acl -LiteralPath $envFile -AclObject $acl
    } catch {
        Log-Warn "Could not restrict ACLs on $envFile; review its permissions before adding tokens."
    }

    Log-Ok 'Created local .env template (fill tokens before release publishing)'
}

function script:Create-Venv {
    if (-not $UseVenv) {
        Log-Info '-NoVenv specified — skipping venv creation (deps will install to system Python)'
        return
    }
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would run: uv venv --python $PythonMinVersion .venv   (in $ProductHome)"
        return
    }

    $venvDir = Join-Path $ProductHome '.venv'
    if (Test-Path $venvDir) {
        Log-Ok "Existing venv at $venvDir"
        return
    }

    Log-Info "Creating venv with Python $PythonMinVersion ..."
    Push-Location $ProductHome
    try {
        & uv venv --python $PythonMinVersion .venv
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Die-With-Help 'uv venv failed.' `
            'Check:    uv --version' `
            "Retry:    $SponsorScript" `
            "Diagnose: $SponsorScript doctor"
    }
    Log-Ok 'Venv created'
}

function script:Find-Venv-Python {
    param([string]$VenvDir)

    foreach ($candidate in @(
        (Join-Path $VenvDir 'Scripts\python.exe'),
        (Join-Path $VenvDir 'bin\python'))) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function script:Install-Deps {
    if ($DryRun) {
        ScriptP1
        if ($UseVenv) {
            Write-Host "[DRY-RUN] would run: uv sync --extra all   (in $ProductHome)"
        } else {
            Write-Host "[DRY-RUN] would run: uv pip install --system -e '.[all]'   (in $ProductHome)"
        }
        return
    }

    Push-Location $ProductHome
    try {
        Log-Info 'Installing project + [all] extra (lock-pinned to uv.lock when possible) ...'

        $venvDir = Join-Path $ProductHome '.venv'
        $venvPython = if ($UseVenv) { Find-Venv-Python -VenvDir $venvDir } else { $null }
        if (-not $UseVenv) {
            Log-Warn '-NoVenv uses a fresh system-Python resolve; uv sync has no --system target.'
            $systemOut = & uv pip install --system -e '.[all]' 2>&1
            if ($LASTEXITCODE -eq 0) {
                Log-Ok 'Dependencies installed into system Python (fresh-resolve, NOT lock-pinned)'
                return
            }
            $systemErr = ($systemOut -join "`n")
            if ($systemErr -match 'externally[ -]managed') {
                Log-Warn 'System Python is externally managed (PEP 668). Retrying with --break-system-packages.'
                $retryOut = & uv pip install --system --break-system-packages -e '.[all]' 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Log-Ok 'Dependencies installed into system Python (--break-system-packages)'
                    return
                }
                $systemErr = ($retryOut -join "`n")
            }
            Die-With-Help "uv pip install to system Python failed: $systemErr" `
                'Retry without -NoVenv for the lock-pinned virtual-environment install.'
        }

        if (-not $venvPython) {
            Die-With-Help "Venv Python missing at $venvDir — run without -NoVenv or re-clone." `
                "Retry:  $SponsorScript update"
        }

        # Stage 1 — `uv sync --extra all` honors uv.lock.
        $syncOut = & uv sync --extra all 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log-Ok "Dependencies installed (lock-pinned to uv.lock at $RepoRef)"
            return
        }

        # uv sync failed — inspect why and decide.
        $syncErr = ($syncOut -join "`n")

        $editableSpec = '.[all]'
        if ($syncErr -match 'Extra `?all`? is not defined') {
            Log-Warn 'This clawcodex version has no [all] extra — falling back to uv pip install.'
            Log-Warn '  Dependency versions will be resolved fresh (NOT lock-pinned).'
            Log-Warn '  For strict version pinning, use an install.ps1 whose'
            Log-Warn '  ClawCodexVersion matches a release that includes [all].'
            $editableSpec = '.'
        } else {
            Log-Warn 'uv sync failed; falling back to uv pip install.'
            Log-Warn "  Sync error was: $syncErr"
        }

        # Stage 2 — fallback to a fresh editable resolve in the selected venv.
        $pipArgs = @('--python', $venvPython)
        $pipErr = & uv pip install @pipArgs -e $editableSpec 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log-Ok 'Dependencies installed (fresh-resolve, NOT lock-pinned; target: .venv)'
            return
        }

        $pipErrText = ($pipErr -join "`n")

        # uv's PEP 668 message has changed wording across versions; match both
        # the structured error code ('externally-managed-environment') and the
        # human message ('externally managed') defensively.
        Log-Err "uv pip install failed: $pipErrText"
        Die-With-Help 'Both uv sync and uv pip install failed.' `
            "Re-run with -LogFile <path> to capture full output." `
            "Retry:    $SponsorScript" `
            "Diagnose: $SponsorScript doctor" `
            "Clean:    $SponsorScript uninstall ; $SponsorScript"
    } finally {
        Pop-Location
    }
}

function script:Find-Project-Python {
    if ($UseVenv) {
        return Find-Venv-Python -VenvDir (Join-Path $ProductHome '.venv')
    }
    foreach ($candidate in @('python3', 'python')) {
        $found = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($found) { return $found.Path }
    }
    return $null
}

function script:Install-Git-Hooks {
    Log-Info 'Installing local Git hooks (pre-commit, best-effort) ...'
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would run: python -m pre_commit install   (in $ClawCodexHome)"
        return
    }

    $gitDir     = Join-Path $ClawCodexHome '.git'
    $preCommit  = Join-Path $ClawCodexHome '.pre-commit-config.yaml'
    if (-not (Test-Path $gitDir) -or -not (Test-Path $preCommit)) {
        Log-Warn 'Skipping pre-commit hook install (not a Git worktree with .pre-commit-config.yaml).'
        return
    }

    $pythonBin = Find-Project-Python
    if (-not $pythonBin) {
        Log-Warn 'Skipping pre-commit hook install (project Python not found).'
        return
    }

    $preCommitAvailable = & $pythonBin -m pre_commit --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log-Warn 'Skipping pre-commit hook install (pre-commit is not available in the install environment).'
        return
    }

    Push-Location $ClawCodexHome
    try {
        $hookOut = & $pythonBin -m pre_commit install --hook-type pre-commit 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log-Ok 'Installed .git/hooks/pre-commit'
        } else {
            Log-Warn 'Could not install .git/hooks/pre-commit; run "python -m pre_commit install" manually if you develop in this checkout.'
        }
    } finally {
        Pop-Location
    }
}

function script:Find-Venv-Entry {
    param(
        [string]$VenvDir,
        [string]$Name
    )
    foreach ($candidate in @(
        (Join-Path $VenvDir "Scripts\$Name.exe"),
        (Join-Path $VenvDir "Scripts\$Name.cmd"),
        (Join-Path $VenvDir "Scripts\$Name"),
        (Join-Path $VenvDir "bin\$Name"))) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function script:Write-Wrapper {
    param(
        [string]$Name,
        [string]$Target
    )
    $wrapper = Join-Path $LocalBin "$Name.cmd"

    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would write: $wrapper"
        return
    }

    if (Test-Path -LiteralPath $wrapper) {
        Remove-Item -LiteralPath $wrapper -Force
    }

    $ownedInstallDir = Get-CanonicalPath $ClawCodexHome
    $body = '@echo off' + "`r`n" +
        'REM Auto-generated by clawcodex install.ps1 — do not edit by hand.' + "`r`n" +
        "REM CLAWCODEX_INSTALL_DIR=$ownedInstallDir" + "`r`n" +
        'REM Regenerate by re-running install.ps1.' + "`r`n" +
        'REM Point the runtime at the configured config dir; the wrapper itself is' + "`r`n" +
        'REM pinned to the install dir baked in at generation time, but the config' + "`r`n" +
        'REM dir can be re-pointed at runtime by the user via this env var.' + "`r`n" +
        'setlocal' + "`r`n" +
        "if `"%CLAWCODEX_CONFIG_DIR%`"==`"`" set `"CLAWCODEX_CONFIG_DIR=$ConfigDir`"" + "`r`n" +
        "`"$Target`" %*" + "`r`n" +
        'endlocal'
    Set-Content -LiteralPath $wrapper -Value $body -Encoding UTF8
    Log-Ok "$wrapper -> $Target  (CLAWCODEX_CONFIG_DIR=$ConfigDir)"
}

function script:Register-Commands {
    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would register: $LocalBin\clawcodex-dev.cmd, $LocalBin\clawcodex.cmd"
        return
    }
    if (-not (Test-Path -LiteralPath $LocalBin -PathType Container)) {
        New-Item -ItemType Directory -Path $LocalBin -Force | Out-Null
    }

    $entry = $null
    if ($UseVenv) {
        $entry = Find-Venv-Entry -VenvDir (Join-Path $ProductHome '.venv') -Name $EntryPoint
        if (-not $entry) {
            Die-With-Help "Entry point '$EntryPoint' not found inside $(Join-Path $ProductHome '.venv') — dependency install may have failed." `
                "Retry:    $SponsorScript update" `
                "Diagnose: $SponsorScript doctor"
        }
    } else {
        foreach ($candidate in @(
            (Join-Path $LocalBin "$EntryPoint.exe"),
            (Join-Path $LocalBin "$EntryPoint.cmd"),
            (Join-Path $LocalBin "$EntryPoint"))) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $entry = (Resolve-Path -LiteralPath $candidate).Path
                break
            }
        }
        if (-not $entry) {
            $found = Get-Command $EntryPoint -ErrorAction SilentlyContinue
            if ($found) { $entry = $found.Path }
        }
        if (-not $entry) {
            Die-With-Help "Entry point '$EntryPoint' not found on PATH after system install — check 'Get-Command $EntryPoint'." `
                "Retry:  $SponsorScript"
        }
    }

    Write-Wrapper -Name 'clawcodex-dev' -Target $entry
    Write-Wrapper -Name 'clawcodex'    -Target $entry
}

function script:Update-User-Path {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ([string]::IsNullOrEmpty($current)) { $current = '' }

    # Windows user PATH is semicolon-separated.  Normalize trailing separators
    # before comparing so repeated installs do not add equivalent duplicates.
    $normalizedBin = $LocalBin.TrimEnd([char[]]'\/')
    $alreadyInPath = @(
        $current -split ';' |
            Where-Object { $_ -and $_.Trim().TrimEnd([char[]]'\/') -ieq $normalizedBin }
    ).Count -gt 0
    if ($alreadyInPath) {
        Log-Ok "$LocalBin is already in User PATH"
        return
    }

    if ($DryRun) {
        ScriptP1
        Write-Host "[DRY-RUN] would append $LocalBin to User PATH"
        return
    }

    $newPath = if ($current) { "$LocalBin;$current" } else { $LocalBin }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Set-InstallUserPathOwnership -Added $true

    # Reflect in the current process so the rest of this session sees it.
    $env:Path = "$LocalBin;$env:Path"

    Log-Ok "Appended $LocalBin to User PATH  (open a new shell to make it visible everywhere)"
}

function script:Run-PostInstall-Setup {
    if (-not $RunSetup) {
        Log-Warn 'Setup wizard skipped (-NoSetup). Run clawcodex-dev manually to configure.'
        return
    }

    Log-Info 'Post-install setup wizard is available — launching clawcodex-dev setup ...'
    # We intentionally do NOT spawn a blocking interactive REPL here.  The
    # install script must remain non-interactive so it can run unattended
    # in CI / Docker / by orchestrators.  The wizard itself is a subcommand
    # the user runs themselves; we just announce it.
    $cmd = Get-Command clawcodex-dev -ErrorAction SilentlyContinue
    if ($cmd) {
        Log-Ok 'Run one of:'
        Write-Host "    ${C_Bold}clawcodex-dev${C_Reset}          # start the interactive REPL (triggers first-run setup if config is empty)"
        Write-Host "    ${C_Bold}clawcodex-dev -Help${C_Reset}   # see all options"
    } else {
        Log-Warn 'clawcodex-dev not on PATH yet — close and reopen PowerShell first.'
    }
}

function script:Update-Install {
    Log-Info "Updating clawcodex at $ClawCodexHome (ref: $RepoRef) ..."
    if (-not (Test-Path (Join-Path $ClawCodexHome '.git'))) {
        Die-With-Help "No existing install at $ClawCodexHome." `
            "Run: $SponsorScript install   (fresh install)" `
            "Or:  $SponsorScript doctor    (diagnose environment)"
    }
    Check-Git
    Install-Uv
    Ensure-Python
    Ensure-Python-Pyo3Compat
    Clone-OrUpdate-Repo
    Assert-RuntimePayload
    Setup-UpstreamSrc -RebuildIfStale
    Ensure-Local-EnvFile
    Create-Venv
    Install-Deps
    Install-Git-Hooks
    Register-Commands
    Update-User-Path
    Log-Ok 'Update complete.'
    Log-Info "Run '$SponsorScript verify' to confirm health."
}

function script:Install-Main {
    Write-Host "${C_Bold}clawcodex installer v$InstallerVersion${C_Reset}"
    Write-Host "  ${C_Bold}OS:${C_Reset}          $OS"
    Write-Host "  ${C_Bold}Install dir:${C_Reset} $ClawCodexHome"
    Write-Host "  ${C_Bold}Product dir:${C_Reset} $ProductHome"
    Write-Host "  ${C_Bold}Config dir:${C_Reset}  $ConfigDir"
    Write-Host "  ${C_Bold}Git ref:${C_Reset}     $RepoRef"
    if ($UseVenv) { Write-Host "  ${C_Bold}Venv:${C_Reset}        create at $ProductHome\.venv" }
    else          { Write-Host "  ${C_Bold}Venv:${C_Reset}        ${C_Yellow}skipped (-NoVenv, system Python)${C_Reset}" }
    if ($RunSetup) { Write-Host '  Setup wizard: announce only (non-blocking)' }
    else            { Write-Host "  ${C_Bold}Setup wizard:${C_Reset} ${C_Yellow}skipped (-NoSetup)${C_Reset}" }
    if ($DryRun)    { Write-Host "  ${C_Bold}Mode:${C_Reset}        ${C_Yellow}DRY-RUN (no changes will be made)${C_Reset}" }
    if ($LogFile)   { Write-Host "  ${C_Bold}Log file:${C_Reset}    $LogFile" }

    Log-Step '1/10  Checking prerequisites'
    Check-Git

    Log-Step '2/10  Installing uv (Astral, no admin)'
    # Re-expose uv on PATH in case it was installed earlier in this session.
    $env:Path = "$LocalBin;$((Join-Path $env:USERPROFILE '.cargo\bin'));$env:Path"
    Install-Uv

    Log-Step "3/10  Provisioning Python $PythonMinVersion+"
    Ensure-Python
    Ensure-Python-Pyo3Compat

    Log-Step '4/10  Cloning / updating repository'
    Clone-OrUpdate-Repo

    Log-Step '5/10  Validating runtime payload and setting up upstream source (src/)'
    Assert-RuntimePayload
    Setup-UpstreamSrc

    Log-Step '6/10  Initializing local release .env'
    Ensure-Local-EnvFile

    if ($UseVenv) { Log-Step '7/10  Creating virtual environment' } else { Log-Step '7/10  Preparing (no venv — using system Python)' }
    Create-Venv

    Log-Step '8/10  Installing dependencies (uv sync --extra all, lock-pinned)'
    Install-Deps

    Log-Step '9/10  Installing local Git hooks'
    Install-Git-Hooks

    Log-Step '10/10  Registering global commands & patching PATH'
    Register-Commands
    Update-User-Path

    Write-Host ''
    if ($DryRun) { Log-Ok 'Dry run complete; no changes were made.' }
    else         { Log-Ok 'Installation complete!' }
    Write-Host ''
    Write-Host "  ${C_Bold}Try it:${C_Reset}"
    Write-Host "    clawcodex-dev -Help    # primary command"
    Write-Host "    clawcodex    -Help     # alias of clawcodex-dev"
    Write-Host ''
    Write-Host "  ${C_Bold}Checkout at:${C_Reset}   $ClawCodexHome"
    Write-Host "  ${C_Bold}Installed at:${C_Reset}  $ProductHome"
    Write-Host "  ${C_Bold}Config at:${C_Reset}    $ConfigDir"
    Write-Host "  ${C_Bold}Commands at:${C_Reset}   $LocalBin\clawcodex-dev.cmd, $LocalBin\clawcodex.cmd"
    Write-Host ''

    Run-PostInstall-Setup

    Log-Warn "Open a new PowerShell window, or run:  `$env:Path = '$LocalBin;`$env:Path'"
}

function script:Hydrate-Main {
    Write-Host "${C_Bold}clawcodex source hydrator v$InstallerVersion${C_Reset}"
    Write-Host "  ${C_Bold}Checkout dir:${C_Reset} $ClawCodexHome"
    Write-Host "  ${C_Bold}Product dir:${C_Reset}  $ProductHome"
    Write-Host "  ${C_Bold}AgentSDK ref:${C_Reset} $RepoRef"
    Write-Host "  ${C_Bold}Upstream ref:${C_Reset} $UpstreamRef"
    Write-Host "  ${C_Bold}Patch set:${C_Reset}    $PatchSetId ($ExpectedPatchCount patches)"

    Check-Git
    Clone-OrUpdate-Repo
    Setup-UpstreamSrc
    if ($DryRun) { Log-Ok 'Dry run complete; source was not changed.' }
    else         { Log-Ok "Source hydration complete at $ProductHome\src" }
}

function script:Print-Usage-Hint {
    Write-StderrLine -Message "Try '$SponsorScript -Help' for usage."
}

function script:Invoke-OptionalInstallerCommand {
    param([Parameter(Mandatory)][string]$Name)

    $command = Get-Command -Name $Name -CommandType Function -ErrorAction SilentlyContinue
    if (-not $command) {
        Die-With-Help "Installer command '$Name' is not available yet." `
            'Merge the diagnostics/uninstall follow-up PR, then retry.'
    }
    & $command
}

function script:Initialize-Config {
    # Resolve overrides -> effective install/config paths.  Must run BEFORE
    # the install pipeline, otherwise -InstallDir / -Ref are silently ignored.
    $script:ClawCodexHome = if ($InstallDir) { $InstallDir } else { $DefaultInstallDir }
    $script:ConfigDir     = if ($ConfigPath) { $ConfigPath } else { $DefaultConfigDir }
    # Validate before Split-Path/Join-Path so malformed or drive-relative input
    # fails cleanly without PowerShell emitting non-terminating path errors.
    Assert-SafeInstallTarget
    $script:ClawCodexParentDir = Split-Path -Path $ClawCodexHome -Parent
    $script:ProductHome        = Join-Path $ClawCodexHome $ProductSubdir
    if ($Ref) { $script:RepoRef = $Ref }

    $script:OS = Detect-OS

    # Make uv visible early in case it's already installed but not on PATH.
    $env:Path = "$LocalBin;$((Join-Path $env:USERPROFILE '.cargo\bin'));$env:Path"

}

function script:Invoke-With-LogFile {
    param([scriptblock]$Body)

    # Validate the destination before starting transcript/tee.  If the
    # destination itself is unusable, its warning can only be shown on the
    # console; all subsequent initialization and command output is captured.
    if ($LogFile) {
        $logDir = Split-Path -Path $LogFile -Parent
        if ($logDir -and -not (Test-Path $logDir)) {
            try { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
            catch {
                Log-Warn "Cannot create log dir $logDir; -LogFile ignored"
                $script:LogFile = $null
            }
        }
        if ($LogFile) {
            try {
                # Touch the file so a permission error shows up here (under
                # the script's log dir creation), not deep inside the
                # pipeline where it would be harder to diagnose.
                # The parent directory was created and validated above.
                if (-not (Test-Path $LogFile)) {
                    Set-Content -LiteralPath $LogFile -Value '' -Encoding UTF8
                }
            } catch {
                Log-Warn "Cannot open log file $LogFile; -LogFile ignored"
                $script:LogFile = $null
            }
        }
    }
    if (-not $LogFile) {
        & $Body
        return
    }
    # Windows PowerShell 5.1 writes Write-Host directly to the host, so it
    # cannot be captured by *>&1.  Use a transcript there; PowerShell 6+ maps
    # Write-Host to the information stream and can use Tee-Object directly.
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        $transcriptStarted = $false
        try {
            Start-Transcript -Path $LogFile -Append -Force -ErrorAction Stop | Out-Null
            $transcriptStarted = $true
            & $Body
        } finally {
            if ($transcriptStarted) {
                Stop-Transcript | Out-Null
            }
        }
        return
    }

    # Tee-Object writes each input object to BOTH the file (append) AND its
    # output stream.  *>&1 merges all PowerShell streams into output.
    try {
        & $Body *>&1 | Tee-Object -FilePath $LogFile -Append
    } catch {
        # If the body throws, re-throw so the outer try/catch can set the rc.
        throw
    }
}

function script:Show-Help {
    $diagnosticsHelp = if (Get-Command Get-InstallStatus -CommandType Function -ErrorAction SilentlyContinue) {
        'Status, doctor, verify, and uninstall commands are available.'
    } else {
        'Status, doctor, verify, and uninstall arrive in the independently mergeable follow-up.'
    }
    @(
        'clawcodex PowerShell installer'
        ''
        'Usage:'
        '  .\install.ps1 [install] [options]'
        '  .\install.ps1 hydrate [options]'
        '  .\install.ps1 status|doctor|verify [options]'
        '  .\install.ps1 update [options]'
        '  .\install.ps1 uninstall [options]'
        '  .\install.ps1 help'
        ''
        'Options:'
        '  -Ref <ref> -InstallDir <path> -ConfigDir <path>'
        '  -NoVenv -NoSetup -DryRun -ForceSrc -Yes -LogFile <path>'
        ''
        'This part provides dependency installation, command registration,'
        'source hydration, and update.'
        $diagnosticsHelp
    ) | ForEach-Object { Write-Host $_ }
}

function script:Show-HelpZh {
    $diagnosticsHelp = if (Get-Command Get-InstallStatus -CommandType Function -ErrorAction SilentlyContinue) {
        '状态、诊断、校验和卸载命令已可用。'
    } else {
        '状态、诊断、校验和卸载由可独立合入的后续 PR 提供。'
    }
    @(
        'clawcodex PowerShell 安装脚本'
        ''
        '用法：'
        '  .\install.ps1 [install] [选项]'
        '  .\install.ps1 hydrate [选项]'
        '  .\install.ps1 status|doctor|verify [选项]'
        '  .\install.ps1 update [选项]'
        '  .\install.ps1 uninstall [选项]'
        '  .\install.ps1 help'
        ''
        '选项：'
        '  -Ref <分支或提交> -InstallDir <路径> -ConfigDir <路径>'
        '  -NoVenv -NoSetup -DryRun -ForceSrc -Yes -LogFile <路径>'
        ''
        '本部分提供依赖安装、命令注册、源码重建和更新流程。'
        $diagnosticsHelp
    ) | ForEach-Object { Write-Host $_ }
}

if ($Help)    { Show-Help; exit 0 }
if ($HelpZh)  { Show-HelpZh; exit 0 }
if ($Version) {
    Write-Host "install.ps1 v$InstallerVersion (installs clawcodex v$ClawCodexVersion)"
    exit 0
}

if ($Uninstall) { $Subcommand = 'uninstall' }
if (-not $Subcommand) { $Subcommand = 'install' }
$script:rc = 0
try {
    Invoke-With-LogFile -Body {
        Initialize-Config
        switch ($Subcommand) {
            'install' { Install-Main }
            'hydrate' { Hydrate-Main }
            'status'  { Invoke-OptionalInstallerCommand 'Get-InstallStatus' }
            'doctor'  { Invoke-OptionalInstallerCommand 'Invoke-Doctor' }
            'verify'  { Invoke-OptionalInstallerCommand 'Invoke-Verify' }
            'update'  { Update-Install }
            'uninstall' { Invoke-OptionalInstallerCommand 'Uninstall-Install' }
            'help'    { Show-Help }
            default {
                Log-Err "Unknown subcommand: $Subcommand"
                $script:rc = 1
            }
        }
    }
} catch {
    if (-not $_.Exception.Data['ClawCodexInstallerHandled']) {
        Log-Err "Installer crash: $_"
    }
    $script:rc = 1
} finally {
    Write-ExitSummary $script:rc
    Flush-StderrLogBuffer
}

exit $script:rc
