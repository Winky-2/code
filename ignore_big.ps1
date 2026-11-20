<#
ignore_big.ps1 - 防止大檔（模型權重等）卡住 git push

用法（在 repo 內任一位置執行）：
  .\ignore_big.ps1                 # 預設門檻 50MB：大檔寫入 .gitignore + 取消追蹤 + 檢查待推送 commit
  .\ignore_big.ps1 -LimitMB 100     # 自訂門檻
  .\ignore_big.ps1 -CheckOnly      # 只檢查，不改任何東西
若出現「無法載入，因為這個系統上已停用指令碼執行」：
  powershell -ExecutionPolicy Bypass -File .\ignore_big.ps1

建議習慣：每次 commit 前先跑一次。

------------------------------------------------------------------
背景：GitHub 限制
  - 單檔 > 100MB：整次 push 被拒（GH001: Large files detected）
  - 單次 push 總量 > 2GB：常見症狀 RPC failed; HTTP 500 / remote end hung up
  - 已 commit 的大檔會留在歷史裡，就算之後刪掉，push 仍會傳上去

救援流程（大檔已經 commit、但還沒推上去時）：
  git reset --soft origin/main     # 拆回未推送的 commit，硬碟檔案不刪
  git reset                        # 全部取消暫存
  .\ignore_big.ps1                 # 大檔寫入 .gitignore
  git add .
  git status                       # 確認沒有大檔
  git commit -m "說明"
  .\ignore_big.ps1 -CheckOnly      # 應顯示「沒有超過門檻的檔案」
  git push origin main --progress

注意：.gitignore 不能依大小過濾，本腳本是把「當下」超過門檻的檔案路徑逐一寫入。
      路徑含 [ ] # ! 等字元時 .gitignore 可能比對不到，跑完請用 git status 確認。
#>

param(
    [int]$LimitMB = 100,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$root = git rev-parse --show-toplevel 2>$null
if (-not $root) { Write-Host "這裡不是 git repo" -ForegroundColor Red; exit 1 }
Set-Location $root
$limit = $LimitMB * 1MB

# ---------- 1. 工作目錄中的大檔 ----------
$big = Get-ChildItem -Recurse -File -Force |
    Where-Object { $_.FullName -notmatch '\\\.git\\' -and $_.Length -gt $limit }
$rel = @($big | ForEach-Object { (Resolve-Path -Relative $_.FullName).Substring(2) -replace '\\', '/' })

Write-Host "`n[1] 工作目錄中 > $LimitMB MB 的檔案：$($rel.Count) 個" -ForegroundColor Cyan
if ($big) {
    $big | Select-Object @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }, FullName | Format-Table -AutoSize
}

if (-not $CheckOnly -and $rel.Count -gt 0) {
    $existing = if (Test-Path .gitignore) { @(Get-Content .gitignore) } else { @() }
    $new = @($rel | Where-Object { $existing -notcontains $_ })
    if ($new.Count -gt 0) {
        Add-Content .gitignore -Encoding UTF8 ("`n# auto: files > ${LimitMB}MB ($(Get-Date -Format 'yyyy-MM-dd'))`n" + ($new -join "`n"))
        Write-Host "已寫入 .gitignore：$($new.Count) 筆" -ForegroundColor Green
    } else {
        Write-Host ".gitignore 已包含全部大檔" -ForegroundColor Green
    }
    foreach ($f in $rel) { git rm --cached --quiet --ignore-unmatch -- "$f" | Out-Null }
    Write-Host "已取消追蹤（硬碟檔案保留）" -ForegroundColor Green
}

# ---------- 2. 待推送 commit 中的大檔 ----------
$upstream = git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null
if (-not $upstream) { $upstream = 'origin/main' }

Write-Host "`n[2] 待推送 commit（$upstream..HEAD）中 > $LimitMB MB 的檔案：" -ForegroundColor Cyan
$pending = git rev-list --objects "$upstream..HEAD" |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    Where-Object { $_ -like 'blob*' -and [long]($_.Split(' ')[2]) -gt $limit }

if ($pending) {
    $pending | ForEach-Object {
        $p = $_.Split(' ', 4)
        '{0,8:N1} MB  {1}' -f ([long]$p[2] / 1MB), $p[3]
    }
    Write-Host "`n大檔已在 commit 歷史中，直接 push 會失敗。請照檔頭「救援流程」處理。" -ForegroundColor Red
} else {
    Write-Host "沒有超過門檻的檔案，可以 push。" -ForegroundColor Green
}
