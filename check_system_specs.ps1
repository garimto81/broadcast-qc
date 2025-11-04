# Windows 시스템 사양 확인 스크립트
# PowerShell 관리자 권한으로 실행

Write-Host "=== 시스템 사양 확인 ===" -ForegroundColor Green
Write-Host ""

# OS 정보
Write-Host "[ OS 정보 ]" -ForegroundColor Yellow
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture, TotalVisibleMemorySize | Format-List

# CPU 정보
Write-Host "[ CPU 정보 ]" -ForegroundColor Yellow
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List

# RAM 정보
Write-Host "[ RAM 정보 ]" -ForegroundColor Yellow
$ram = Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
$ramGB = [math]::Round($ram.Sum / 1GB, 2)
Write-Host "총 RAM: $ramGB GB"
Write-Host "RAM 슬롯 정보:"
Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer, Speed, Capacity | Format-Table

# GPU 정보
Write-Host "[ GPU 정보 ]" -ForegroundColor Yellow
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List

# 저장장치 정보
Write-Host "[ 저장장치 정보 ]" -ForegroundColor Yellow
Get-CimInstance Win32_DiskDrive | Select-Object Model, Size, MediaType | Format-Table

Write-Host "[ 드라이브 여유 공간 ]" -ForegroundColor Yellow
Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Used -ne $null} |
    Select-Object Name,
        @{Name="Used(GB)";Expression={[math]::Round($_.Used/1GB,2)}},
        @{Name="Free(GB)";Expression={[math]::Round($_.Free/1GB,2)}},
        @{Name="Total(GB)";Expression={[math]::Round(($_.Used+$_.Free)/1GB,2)}} |
    Format-Table

# Python 버전 확인
Write-Host "[ 설치된 소프트웨어 ]" -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion"
} catch {
    Write-Host "Python: 설치되지 않음" -ForegroundColor Red
}

# Node.js 버전 확인
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Node.js: $nodeVersion"
} catch {
    Write-Host "Node.js: 설치되지 않음" -ForegroundColor Red
}

# FFmpeg 확인
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
    Write-Host "FFmpeg: $ffmpegVersion"
} catch {
    Write-Host "FFmpeg: 설치되지 않음" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 시스템 사양 확인 완료 ===" -ForegroundColor Green
Write-Host ""

# MVP 실행 가능 여부 판단
$coreCount = (Get-CimInstance Win32_Processor).NumberOfCores
$ramGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB, 2)
$freeSpace = (Get-PSDrive C).Free / 1GB

Write-Host "[ MVP 실행 가능성 분석 ]" -ForegroundColor Cyan

if ($coreCount -ge 4 -and $ramGB -ge 8 -and $freeSpace -ge 50) {
    Write-Host "✅ 최소 요구사항을 만족합니다!" -ForegroundColor Green

    if ($coreCount -ge 8 -and $ramGB -ge 16) {
        Write-Host "🚀 최적 성능으로 실행 가능합니다!" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ 일부 요구사항이 부족합니다:" -ForegroundColor Yellow

    if ($coreCount -lt 4) {
        Write-Host "  - CPU 코어: $coreCount개 (최소 4개 필요)" -ForegroundColor Red
    }
    if ($ramGB -lt 8) {
        Write-Host "  - RAM: $ramGB GB (최소 8GB 필요)" -ForegroundColor Red
    }
    if ($freeSpace -lt 50) {
        Write-Host "  - 여유 공간: $([math]::Round($freeSpace,2)) GB (최소 50GB 필요)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "이 정보를 복사하여 제공해주세요." -ForegroundColor Yellow