# 중복 문서 정리 스크립트

Write-Host "📁 문서 정리 시작..." -ForegroundColor Cyan

# 현재 위치 설정
Set-Location "c:\claude\Broadcast QC"

# 백업 폴더 생성
$backupPath = "docs\archive_backup"
if (!(Test-Path $backupPath)) {
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    Write-Host "✅ 백업 폴더 생성: $backupPath" -ForegroundColor Green
}

# 이동시킬 파일 목록
$filesToMove = @(
    "docs\prd.md",
    "docs\prd_v2.0.md",
    "docs\prd_mvp_minimal_cost.md",
    "docs\tech_architecture.md"
)

Write-Host "`n📦 파일 이동 중..." -ForegroundColor Yellow

foreach ($file in $filesToMove) {
    if (Test-Path $file) {
        $fileName = Split-Path $file -Leaf
        $destination = Join-Path $backupPath $fileName

        # 파일 이동
        Move-Item -Path $file -Destination $destination -Force
        Write-Host "  ✓ 이동됨: $fileName → archive_backup\" -ForegroundColor Green
    } else {
        Write-Host "  - 파일 없음: $file" -ForegroundColor Gray
    }
}

Write-Host "`n📊 최종 문서 구조:" -ForegroundColor Cyan
Write-Host "docs\" -ForegroundColor White
$mainDocs = Get-ChildItem -Path "docs" -File -Filter "*.md" | Where-Object { $_.Name -ne "README.md" }
foreach ($doc in $mainDocs) {
    Write-Host "  ├── $($doc.Name)" -ForegroundColor Green
}
Write-Host "  ├── README.md" -ForegroundColor Green
Write-Host "  └── archive_backup\" -ForegroundColor Yellow

$backupDocs = Get-ChildItem -Path $backupPath -File -Filter "*.md"
foreach ($doc in $backupDocs) {
    Write-Host "      └── $($doc.Name)" -ForegroundColor Gray
}

# 통계
$activeCount = (Get-ChildItem -Path "docs" -File -Filter "*.md").Count
$backupCount = (Get-ChildItem -Path $backupPath -File -Filter "*.md" -ErrorAction SilentlyContinue).Count

Write-Host "`n📈 정리 결과:" -ForegroundColor Cyan
Write-Host "  • 활성 문서: $activeCount 개" -ForegroundColor Green
Write-Host "  • 백업 문서: $backupCount 개" -ForegroundColor Yellow
Write-Host "  • 총 문서: $($activeCount + $backupCount) 개" -ForegroundColor White

Write-Host "`n✅ 문서 정리 완료!" -ForegroundColor Green
Write-Host "핵심 문서만 유지되고 중복 문서는 백업되었습니다." -ForegroundColor White