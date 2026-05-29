# Octopus Python

這是原 `octopus/` Go 專案的 Python/FastAPI 移植版，預設放在 `./octopus-python`。

目前移植範圍：

- FastAPI 服務與 `/api/v1/*` 管理 API
- SQLAlchemy 資料模型與 SQLite/MySQL/PostgreSQL 連線
- 預設管理者 `admin/admin`
- Admin JWT 驗證與 `sk-octopus-*` API Key 驗證
- Channel、Group、Model/Price、Stats、Relay Log、Setting、匯入/匯出
- `/v1/models` 與 OpenAI/Anthropic/Gemini/Doubao/OpenAI Embeddings 的基礎 relay 代理
- 靜態資產目錄 `static/out`（若放入前端 build 產物會自動服務）

## 快速開始

```bash
cd octopus-python
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .
python -m octopus_python start
```

開發與測試依賴：

```bash
pip install -e .[dev]
python -m pytest
python -m ruff check .
```

Windows 若 `python` 指令指向 Microsoft Store alias，可改用：

```powershell
py -3 -m pytest
py -3 -m ruff check .
py -3 -m octopus_python version
```

啟動後預設監聽 `0.0.0.0:8080`，設定檔會自動建立於 `data/config.json`。

預設帳號：

- Username: `admin`
- Password: `admin`

## 設定

設定格式與 Go 版相容：

```json
{
  "server": {"host": "0.0.0.0", "port": 8080},
  "database": {"type": "sqlite", "path": "data/data.db"},
  "log": {"level": "info"}
}
```

環境變數支援 `OCTOPUS_SERVER_PORT`、`OCTOPUS_DATABASE_TYPE` 等 `OCTOPUS_` 前綴覆寫。

## 靜態前端

Python 版已將 Go 專案的 Web UI 建置後產物移植到 `static/out`，啟動時會自動掛載：

- 開發目錄：`octopus-python/static/out`
- Docker/安裝後工作目錄：`./static/out`

若你重新建置 `octopus/web`，可再次把輸出產物覆蓋到 `octopus-python/static/out`。

## 已移植的相容性重點

- 管理 API 回應格式維持 Go 版 `{code,message,data}`。
- 預設管理帳號仍為 `admin/admin`。
- JWT secret 與 Go 版一致使用 `username + password_hash`，API Key 前綴為 `sk-octopus-`。
- `/api/v1/log/list` 回傳前端相容的 `RelayLog[]`，不是分頁物件。
- `/api/v1/user/status` 回傳 `"ok"`，改密碼/改帳號成功字串與 Go handler 對齊。
- 統計日期採 `YYYYMMDD`，小時統計採 0..23，並補齊今日缺漏小時。
- Channel 更新保留明確傳入的 `0`/`false`，避免 `auto_group=0` 被誤寫成空字串。
- 密碼雜湊使用 `bcrypt` 套件直接實作，避免 Python 3.14 + passlib/bcrypt 5 的相容性問題。
- 串流 relay 已支援 SSE passthrough、首 token timeout 後切換下一個 channel、stream usage 解析與 relay log 補寫。
- FastAPI lifespan 會啟動背景維護 task：從 `models.dev` 自動抓取模型價格、channel 模型同步、relay log retention 清理。
- 價格同步遵循「手動價格優先」：非 0 的使用者自訂價格不會被覆蓋，未定價或新 channel model 會自動補入 `models.dev` 價格。

## 注意

Python 版不依賴 Go 專案的 `axonhub`，relay 採「協議相容的 HTTP 轉發 + 常用格式轉換」實作；複雜串流 usage 補齊與完整 provider 特例行為可能與 Go 版仍有細節差異。

目前已支援 OpenAI Chat/Responses/Embeddings/Images、Anthropic Messages、Gemini Contents 的常用轉換與 endpoint 選擇。串流已改為 passthrough 並補上首 token timeout 與 usage/log 聚合；少數 provider 專屬串流事件格式仍採 best-effort 解析。
