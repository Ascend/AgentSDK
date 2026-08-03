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
#  install.ps1 - clawcodex PowerShell installer (part1)
# ----------------------------------------------------------------------------
#  This staged part provides a complete source hydrate foundation:
#    1. resolve and validate an AgentSDK checkout;
#    2. fetch the pinned clawcodex upstream commit;
#    3. validate and apply the declared patch series;
#    4. atomically publish src/ and its integrity marker.
#
#  Usage:
#      .\install.ps1 hydrate
#      .\install.ps1 hydrate -DryRun
#      .\install.ps1 hydrate -ForceSrc
#      .\install.ps1 help
#
#  Full install, update, diagnostics, and uninstall land in follow-up PRs.
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
    exit 1
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
        Log-Err 'Git is not installed.'
        Get-OsInstallHint $OS | ForEach-Object { Write-StderrLine -Message $_ }
        $script:rc = 1
        exit 1
    }
    $version = & git --version
    Log-Ok $version
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
        schema_version = 1
        created_by     = 'clawcodex-ascend/install.ps1'
        install_dir    = (Get-CanonicalPath $ClawCodexHome)
        product_subdir = $ProductSubdir
        repo_url       = $RepoUrl
        created_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $markerFile -Encoding UTF8 -ErrorAction Stop
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

    # Set up log-file tee if requested.  Must happen AFTER $LogFile is bound
    # but BEFORE any other output.  After this redirection, [Console]::IsOutputRedirected
    # becomes true, so the [install.ps1] prefix is added on every line.
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
            # The actual teeing happens in Invoke-With-LogFile, which wraps
            # the main body.  Here we just validate the path is writable.
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
}

function script:Invoke-With-LogFile {
    param([scriptblock]$Body)
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
    @(
        'clawcodex PowerShell installer - part1 source hydrate foundation'
        ''
        'Usage:'
        '  .\install.ps1 hydrate [-Ref <ref>] [-InstallDir <path>]'
        '                          [-ConfigDir <path>] [-DryRun] [-ForceSrc]'
        '  .\install.ps1 help'
        ''
        'This staged part validates the AgentSDK checkout, reconstructs src/'
        'from the pinned upstream commit plus the declared patch series, and'
        'writes the source marker only after an atomic successful replacement.'
        ''
        'The install/update/diagnostic commands are added by follow-up PRs.'
    ) | ForEach-Object { Write-Host $_ }
}

function script:Show-HelpZh {
    @(
        'clawcodex PowerShell 安装脚本 - part1 源码重建基础'
        ''
        '用法：'
        '  .\install.ps1 hydrate [-Ref <分支或提交>] [-InstallDir <路径>]'
        '                          [-ConfigDir <路径>] [-DryRun] [-ForceSrc]'
        '  .\install.ps1 help'
        ''
        '当前阶段会校验 AgentSDK 工作区，使用固定的上游提交和有序补丁'
        '重建 src/，并且只在原子替换成功后写入源码完整性标记。'
        ''
        '安装、更新、诊断和卸载命令将在后续 PR 中提供。'
    ) | ForEach-Object { Write-Host $_ }
}

# Part1 entry point: source hydration only.
if ($Help)    { Show-Help; exit 0 }
if ($HelpZh)  { Show-HelpZh; exit 0 }
if ($Version) {
    Write-Host "install.ps1 v$InstallerVersion (part1 source hydrate)"
    exit 0
}

if ($Uninstall) { $Subcommand = 'uninstall' }
if (-not $Subcommand) { $Subcommand = 'hydrate' }
Initialize-Config

$script:rc = 0
try {
    Invoke-With-LogFile -Body {
        switch ($Subcommand) {
            'hydrate' { Hydrate-Main }
            'help'    { Show-Help }
            default {
                Log-Err "Subcommand '$Subcommand' is not available in installer part1."
                Log-Err 'Use hydrate/help now; install and diagnostics land in follow-up PRs.'
                $script:rc = 1
            }
        }
    }
} catch {
    Log-Err "Installer crash: $_"
    $script:rc = 1
} finally {
    Write-ExitSummary $script:rc
    Flush-StderrLogBuffer
}

exit $script:rc
