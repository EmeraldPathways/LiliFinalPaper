param(
    [string]$OutputZip = (Join-Path $PSScriptRoot "dist\lili-examiner-package.zip"),
    [string]$PackageRoot = "LILI_FINAL_PAPER_DEMO_examiner",
    [switch]$DryRun,
    [switch]$IncludeReferenceAssets
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$repoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$fixedTimestamp = [DateTimeOffset]::Parse("2026-07-21T00:00:00Z")
$requiredRelativePaths = @(
    "backend/app/data/raw/.gitkeep",
    "backend/app/data/raw/articles.csv",
    "backend/app/data/raw/customers.csv",
    "backend/app/data/raw/transactions_train.csv",
    "backend/app/data/processed/.gitkeep",
    "references/Config Manual.docx",
    "start-backend.cmd",
    "start-frontend.cmd",
    "backend/requirements.txt",
    "frontend/package.json",
    "frontend/package-lock.json"
)

function Get-RelativeZipPath {
    param([string]$FullName)

    $rootUri = New-Object System.Uri(($repoRoot.TrimEnd("\") + "\"))
    $fileUri = New-Object System.Uri($FullName)
    $relativeUri = $rootUri.MakeRelativeUri($fileUri)
    $relative = [System.Uri]::UnescapeDataString($relativeUri.ToString())
    return $relative.Replace("\", "/")
}

function Test-IncludedFile {
    param([string]$RelativePath)

    $path = $RelativePath.Replace("\", "/")
    $lower = $path.ToLowerInvariant()

    if ($lower -eq "backend/app/data/processed/.gitkeep") {
        return $true
    }

    $excludedPrefixes = @(
        ".git/",
        ".agents/",
        ".codex/",
        ".vscode/",
        "dist/",
        "coverage/",
        ".next/",
        "frontend/.next/",
        "frontend/node_modules/",
        "backend/.venv/",
        ".pytest_cache/",
        "backend/.pytest_cache/",
        ".logs/",
        "backend/.logs/",
        "frontend/.logs/",
        "backend/app/data/processed/",
        "references/reports/"
    )

    if (-not $IncludeReferenceAssets) {
        $excludedPrefixes += "references/web app screenshots/"
    }

    foreach ($prefix in $excludedPrefixes) {
        if ($lower.StartsWith($prefix)) {
            return $false
        }
    }

    $excludedExact = @(
        ".env",
        ".env.local",
        "backend/.env",
        "frontend/.env.local",
        "frontend/tsconfig.tsbuildinfo",
        "processed_5_experiment_files.md",
        "processed_rest_files.md",
        "backend-8009.out.log",
        "backend-8009.err.log",
        "frontend-3009.out.log",
        "frontend-3009.err.log",
        "references/lili - config manual 1707.md"
    )

    if ($excludedExact -contains $lower) {
        return $false
    }

    if ($lower.Contains("/__pycache__/")) {
        return $false
    }

    if ($lower.EndsWith(".pyc")) {
        return $false
    }

    if ($lower.EndsWith(".log")) {
        return $false
    }

    if ($lower.EndsWith(".zip")) {
        return $false
    }

    return $true
}

function Get-IncludedFiles {
    $allFiles = Get-ChildItem -LiteralPath $repoRoot -Recurse -File -Force
    $included = foreach ($file in $allFiles) {
        $relativePath = Get-RelativeZipPath -FullName $file.FullName
        if (Test-IncludedFile -RelativePath $relativePath) {
            [PSCustomObject]@{
                FullName     = $file.FullName
                RelativePath = $relativePath
                Length       = $file.Length
            }
        }
    }

    return $included | Sort-Object RelativePath
}

function Assert-PackageShape {
    param([object[]]$FileRecords)

    $relativePaths = $FileRecords | ForEach-Object { $_.RelativePath.Replace("\", "/") }
    $relativeSet = @{}
    foreach ($path in $relativePaths) {
        $relativeSet[$path.ToLowerInvariant()] = $true
    }

    foreach ($requiredPath in $requiredRelativePaths) {
        if (-not $relativeSet.ContainsKey($requiredPath.ToLowerInvariant())) {
            throw "Required examiner package file is missing from the zip set: $requiredPath"
        }
    }

    $unexpectedProcessed = $relativePaths | Where-Object {
        $lower = $_.ToLowerInvariant()
        $lower.StartsWith("backend/app/data/processed/") -and $lower -ne "backend/app/data/processed/.gitkeep"
    }

    if ($unexpectedProcessed) {
        throw "Unexpected generated processed artifact included in zip set: $($unexpectedProcessed[0])"
    }

    $unexpectedExplainability = $relativePaths | Where-Object {
        $_.ToLowerInvariant().StartsWith("backend/app/data/processed/explainability/")
    }

    if ($unexpectedExplainability) {
        throw "Unexpected explainability artifact included in zip set: $($unexpectedExplainability[0])"
    }
}

function Build-ManifestText {
    param(
        [object[]]$FileRecords,
        [bool]$IncludeReferenceAssetsFlag
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("Examiner package manifest")
    $lines.Add("Package root: $PackageRoot")
    $lines.Add("Deterministic entry timestamp: $($fixedTimestamp.ToString("u"))")
    $lines.Add("Includes raw Kaggle CSVs: yes")
    $lines.Add("Includes processed/.gitkeep only: yes")
    $lines.Add("Includes generated processed outputs: no")
    $lines.Add("Intended examiner run order: First1000 -> Seed99 formal plus bootstrap CI -> Seed99 explainability -> Hybrid ablation")
    $lines.Add("Includes installable dependencies (.venv, node_modules, .next): no")
    $lines.Add("Includes reference screenshots/PDFs: " + ($(if ($IncludeReferenceAssetsFlag) { "yes" } else { "no" })))
    $lines.Add("")
    $lines.Add("sha256 size_bytes relative_path")

    foreach ($record in $FileRecords) {
        $lines.Add("$($record.Sha256) $($record.Length) $($record.RelativePath)")
    }

    return ($lines -join "`n") + "`n"
}

function Copy-FileToZip {
    param(
        [System.IO.Compression.ZipArchive]$Archive,
        [string]$SourcePath,
        [string]$EntryPath
    )

    $entry = $Archive.CreateEntry($EntryPath, [System.IO.Compression.CompressionLevel]::Optimal)
    $entry.LastWriteTime = $fixedTimestamp

    $input = [System.IO.File]::OpenRead($SourcePath)
    try {
        $output = $entry.Open()
        try {
            $input.CopyTo($output)
        }
        finally {
            $output.Dispose()
        }
    }
    finally {
        $input.Dispose()
    }
}

$files = Get-IncludedFiles

if (-not $files -or $files.Count -eq 0) {
    throw "No files matched the examiner package rules."
}

$fileRecords = foreach ($file in $files) {
    $sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    [PSCustomObject]@{
        FullName     = $file.FullName
        RelativePath = $file.RelativePath
        Length       = $file.Length
        Sha256       = $sha256
    }
}

Assert-PackageShape -FileRecords $fileRecords

$totalBytes = ($fileRecords | Measure-Object -Property Length -Sum).Sum
$manifestText = Build-ManifestText -FileRecords $fileRecords -IncludeReferenceAssetsFlag $IncludeReferenceAssets.IsPresent

if ($DryRun) {
    Write-Host "Dry run only. No zip written."
    Write-Host "Included files: $($fileRecords.Count)"
    Write-Host ("Included bytes: {0:N0}" -f $totalBytes)
    Write-Host "Output zip: $OutputZip"
    Write-Host ""
    Write-Host "First 40 included paths:"
    $fileRecords | Select-Object -First 40 -ExpandProperty RelativePath | ForEach-Object { Write-Host " - $_" }
    return
}

$outputDirectory = Split-Path -Parent $OutputZip
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}

if (Test-Path -LiteralPath $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}

$fileStream = [System.IO.File]::Open($OutputZip, [System.IO.FileMode]::CreateNew)
try {
    $archive = New-Object System.IO.Compression.ZipArchive($fileStream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($record in $fileRecords) {
            $entryPath = "$PackageRoot/$($record.RelativePath)"
            Copy-FileToZip -Archive $archive -SourcePath $record.FullName -EntryPath $entryPath
        }

        $manifestEntry = $archive.CreateEntry("$PackageRoot/EXAMINER_ZIP_MANIFEST.txt", [System.IO.Compression.CompressionLevel]::Optimal)
        $manifestEntry.LastWriteTime = $fixedTimestamp
        $manifestWriter = New-Object System.IO.StreamWriter($manifestEntry.Open(), [System.Text.Encoding]::UTF8)
        try {
            $manifestWriter.NewLine = "`n"
            $manifestWriter.Write($manifestText)
        }
        finally {
            $manifestWriter.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }
}
finally {
    $fileStream.Dispose()
}

$zipHash = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash.ToLowerInvariant()
$zipSize = (Get-Item -LiteralPath $OutputZip).Length

Write-Host "Examiner zip created."
Write-Host "Path: $OutputZip"
Write-Host ("Size bytes: {0:N0}" -f $zipSize)
Write-Host "Files included: $($fileRecords.Count)"
Write-Host "ZIP sha256: $zipHash"
