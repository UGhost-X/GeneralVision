$outLog = Join-Path $PWD 'llm_chat_server.out.log'
$errLog = Join-Path $PWD 'llm_chat_server.err.log'
$pidFile = Join-Path $PWD 'llm_chat_server.pid'
$py = Join-Path $PWD '.venv\Scripts\python.exe'
$port = 8000

# 若端口已被占用，说明服务已在运行
$listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "服务已在运行：http://127.0.0.1:$port (PID $($listening.OwningProcess))" -ForegroundColor Yellow
    exit 0
}

foreach ($f in @($outLog, $errLog)) {
    try { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force } } catch { }
}

$p = Start-Process -FilePath $py -ArgumentList @('llm_chat_server.py', '--port', "$port") -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
Set-Content -LiteralPath $pidFile -Value $p.Id

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if ($ready) {
    Write-Host "✔ 服务已启动：http://127.0.0.1:$port" -ForegroundColor Green
    Write-Host "   PID: $($p.Id) | 日志: llm_chat_server.out.log"
} else {
    Write-Host "启动失败，请查看日志：" -ForegroundColor Red
    if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 30 }
}
