# 문서 정리 스크립트

# 백업 폴더 생성
$backupPath = "docs\archive_backup"
if (!(Test-Path $backupPath)) {
    New-Item -ItemType Directory -Path $backupPath -Force
    Write-Host "백업 폴더 생성: $backupPath" -ForegroundColor Green
}

# 통합된 문서들을 백업 폴더로 이동
$filesToArchive = @(
    "docs\prd.md",
    "docs\prd_v2.0.md",
    "docs\prd_mvp_minimal_cost.md",
    "docs\tech_architecture.md"
)

foreach ($file in $filesToArchive) {
    if (Test-Path $file) {
        $fileName = Split-Path $file -Leaf
        Move-Item -Path $file -Destination "$backupPath\$fileName" -Force
        Write-Host "이동됨: $file → $backupPath\$fileName" -ForegroundColor Yellow
    }
}

# 현재 docs 폴더 구조 확인
Write-Host "`n현재 문서 구조:" -ForegroundColor Cyan
Get-ChildItem -Path "docs" -File | Select-Object Name, Length, LastWriteTime | Format-Table

Write-Host "`n백업된 문서:" -ForegroundColor Cyan
Get-ChildItem -Path $backupPath -File | Select-Object Name, Length, LastWriteTime | Format-Table

Write-Host "`n✅ 문서 정리 완료!" -ForegroundColor Green
Write-Host "- 핵심 문서 3개 유지 (PRD_MASTER.md, local_dev_setup_guide.md, optimized_config_for_your_system.md)" -ForegroundColor Green
Write-Host "- 중복 문서 4개 백업 ($backupPath)" -ForegroundColor Yellow