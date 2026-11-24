# 藥瓶定位、對焦、OCR 與 SQLite 查詢

先解壓整個 medicine_detection 資料夾，再從該資料夾開啟 Windows PowerShell。

- [報告版 PowerPoint](medicine_pipeline/presentations/四瓶藥品快速對焦與辨識_報告版.pptx)
- [實測報告](medicine_pipeline/report.md)與[操作手冊](medicine_pipeline/README.md)
- [4.40 秒實拍紀錄](medicine_pipeline/runs/scan_20261118_121929_904595/focus_scan.json)、[OCR 文字](medicine_pipeline/runs/scan_20261118_121929_904595/ocr/run_20261118_122714_233580/summary.json)與[SQLite 查詢結果](medicine_pipeline/runs/scan_20261118_121929_904595/ocr/run_20261118_122714_233580/lookup_20261118_122731_811274.json)

## 實測資料

| 每次改焦後等待 | 取樣前丟棄 | 四瓶結果 |
| --- | --- | --- |
| 0.15 秒 | 1 張 | 4.40 秒完成。見上方報告與 OCR 結果。 |
| 0 秒 | 0 張 | 2.18 秒時因清晰度分數不穩定而中止。[紀錄](medicine_pipeline/runs/scan_20261118_124955_652885/focus_scan.json) |
| 0 秒 | 1 張 | 2.91 秒完成存圖，每瓶只有 1～2 個焦點通過穩定檢查。[紀錄](medicine_pipeline/runs/scan_20261118_125028_110290/focus_scan.json) |
| 0 秒 | 3 張 | 3.31 秒完成存圖，部分瓶的清晰度分數下降。[紀錄](medicine_pipeline/runs/scan_20261118_125111_364728/focus_scan.json) |

0 秒指手動改焦後的穩定等待。每輪都保留開頭約 1 秒的相機自動對焦。零等待的實作驗證涵蓋存圖與清晰度，完整 OCR 與 SQLite 結果請看 4.40 秒版。

## Windows 環境

使用 64 位元 Python 3.10。這份 ZIP 已附模型權重、訓練資料、SQLite 快照，以及 USB 相機的 DirectShow 對焦元件。先安裝 Python 套件：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch==2.13.0' 'torchvision==0.28.0' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install 'ultralytics==8.4.105' 'paddleocr==3.7.0' 'paddlepaddle==3.2.0' 'paddlex==3.7.2' 'opencv-python==4.12.0.88' 'numpy==2.2.6' 'PyYAML==6.0.2' 'matplotlib==3.10.9'
```

可直接對附帶的照片重跑 OCR 與查詢：

```powershell
$scan = '.\medicine_pipeline\runs\scan_20261118_121929_904595'
.\.venv\Scripts\python.exe -m medicine_pipeline.main ocr $scan
```

現場拍攝時，接上支援自動及手動對焦的 USB 相機，讓四瓶正面標籤入鏡：

```powershell
$model = '.\medicine_pipeline\runs\train_20261118_091451_575457\fit\weights\last.pt'
.\.venv\Scripts\python.exe -m medicine_pipeline.main live --model $model --expected 4 --focus-search fast-autofocus
```

\`medicine_pipeline/runs/\` 保留訓練與各輪實拍，\`dataset/\` 保留 40 張訓練與 10 張驗證影像及標註。隨附的 \`ppocrv5_server_test/main.py\` 提供 PP-OCRv6 初始化所共用的 Windows 載入函式。此包以 USB 相機與 PP-OCRv6 流程為主，其他專案的相機 SDK、虛擬環境、模型快取與簡報暫存請依各專案自行準備。原始實測 JSON 保留當時的路徑和數值，重跑請使用上面的相對路徑命令。
