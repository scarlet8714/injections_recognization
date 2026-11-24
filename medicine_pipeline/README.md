# 藥瓶定位、逐瓶對焦與文字辨識操作手冊

這份手冊從開啟 PowerShell 開始，說明如何用現有資料訓練 YOLO11n、用 USB 相機逐瓶對焦、用 PP-OCRv6 small 讀取瓶身文字，再用 SQLite 查詢本機藥品資料。

## 1. 在這台電腦上跑一次完整流程

1. 從 Windows 開始選單開啟 **PowerShell**。
2. 貼上以下命令並按 Enter。PowerShell 接下來會從專案根目錄找程式和模型。

   ```powershell
   Set-Location 'C:\Users\scarl\Desktop\medicine_detection'
   ```

3. 確認 USB 相機已接上。把瓶子固定在畫面內，讓正面標籤對著相機。留一點間距，讓瓶蓋、瓶底和文字都能入鏡。掃描期間保持瓶子、相機和燈光的位置固定，並關閉同時操作相機的其他程式。
4. 輸入以下兩行。`--expected 4` 表示鏡頭前有 **4 瓶**，更換擺設後請相應修改數字。

   ```powershell
   $model = '.\medicine_pipeline\runs\train_20261118_091451_575457\fit\weights\last.pt'
   .\.venv\Scripts\python.exe -m medicine_pipeline.main live --model $model --expected 4
   ```

5. 程式會先讀取影格，找出瓶身的 ROI（感興趣區域，也就是要裁切的瓶框）。數量與 `--expected` 一致時，才會開始控制鏡頭。每完成一瓶，終端會顯示對焦值和清晰度分數。四瓶實測共用取樣後，掃描與重拍約 52.5 秒，然後執行文字辨識。這台 USB 相機加上 `--focus-settle 0.20` 實測約 40.9 秒。
6. `OCR completed` 這行末尾是 `summary.json` 的路徑。其旁邊的 `bottle_01`、`bottle_02` 等資料夾，按瓶身由左到右編號。想最快看文字，請開啟每瓶的 `recognized.txt`。
7. OCR 後會自動查詢 SQLite。`SQLite lookup` 這行末尾是 `lookup_時間戳.json` 的路徑。開啟它即可看到每瓶找到的資料庫候選、規格和查詢依據。

已安裝的 `.venv`、相機原生元件和權重可以直接使用。要重訓模型或在另一台電腦重建環境，再看後面的步驟。

## 2. 檢查資料並訓練 YOLO11n

### 資料放在哪裡

| 資料夾 | 影像 | 瓶框標註 |
| --- | ---: | ---: |
| `dataset/train/` | 40 張 | 113 個（每張 1、3 或 4 瓶） |
| `dataset/valid/` | 10 張 | 10 個（每張 1 瓶） |

每個分割包含 `images` 和 `labels`。同一張照片與標註的檔名主體相同，副檔名分別為 `.jpg`、`.txt`。一行 YOLO 標註是 `類別 中心X 中心Y 寬 高`，後四項使用相對於整張圖的 0 至 1 比例。現有資料的唯一類別是 `Class 0`，代表瓶身定位。50 張影像為 1920 × 1080，兩個分割之間查無檔案內容完全相同的圖片。

部分影像拍攝條件相近，驗證集每張又都只有一瓶。因此驗證分數主要反映這批單瓶影像表現。要檢查換場地、換燈光、多瓶相鄰和沒有瓶子的情況，可另拍獨立場景，整個拍攝場次一起分到驗證或測試集，避免連拍影格分散到不同集合。

### 為什麼用遷移學習

YOLO11n 是較輕量的物件偵測模型，能在這台 CPU 電腦訓練和執行。訓練從官方 COCO 預訓練權重 `models/yolo11n.pt` 開始，凍結前 10 個模組，讓後面的特徵層和偵測頭學習這批藥瓶。訓練使用 640 像素輸入、每批 4 張、固定種子、早停，以及小幅旋轉、平移、縮放、光色和低比例 Mosaic 增強。上下翻轉和左右翻轉都關閉，以保留藥瓶正常擺向與標籤方向。參數寫在 `train.py`。 [Ultralytics YOLO11 官方文件](https://docs.ultralytics.com/models/yolo11)

### 重新訓練及選權重

從專案根目錄執行下列命令即可開始訓練。

```powershell
.\.venv\Scripts\python.exe -m medicine_pipeline.train --device cpu
```

每次建立新資料夾 `medicine_pipeline/runs/train_時間戳/`。其中 `fit/results.csv` 是逐輪紀錄，`fit/weights/` 儲存檢查點，`training_summary.json` 裡的 `checkpoint` 是程式挑選的執行權重。下列命令可只重新評估這次已完成的訓練。

```powershell
.\.venv\Scripts\python.exe -m medicine_pipeline.train --device cpu --evaluate-run .\medicine_pipeline\runs\train_20261118_091451_575457
```

此次 CPU 訓練共完成 13 輪後早停。`best.pt` 由 Ultralytics 依驗證排名選出，`last.pt` 是最後一輪。下面的漏框與誤框統計使用實際執行的信心門檻 `0.40`、重複框抑制 IoU `0.45`，預測與標註 IoU 至少 `0.50` 算配對成功。程式以這個門檻下的 F1 選權重，平手時以 mAP50–95 選擇。

| 檢查點 | mAP50 | mAP50–95 | 正確配對 | 誤框 | 漏框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt` | 99.5% | 87.2% | 0 | 0 | 10 |
| `last.pt`（實拍採用） | 99.5% | 74.4% | 10 | 0 | 0 |

`mAP` 衡量不同信心與重疊條件下的偵測排名。這批資料的兩個權重在排名上都能找到瓶子，只有 `last.pt` 在採用的門檻下把 10 個驗證瓶框都找出來。

## 3. 相機預覽、對焦和清晰度

### 確認相機與框線

```powershell
.\.venv\Scripts\python.exe -m focus_test.main --focus-info
```

本次使用 Windows DirectShow 的 `Usb HDCamera`，索引 `0`，實際解析度 1920 × 1080。鏡頭回報對焦值範圍 0–1023、步進 1。這是裝置刻度，無法直接換算成毫米。若有多台 USB 相機，使用 `--camera-index 1` 等數字確認是哪一台。 [Microsoft 相機對焦屬性](https://learn.microsoft.com/en-us/windows-hardware/drivers/stream/ksproperty-cameracontrol-focus)

若想先目視確認每瓶各有一個框，用下列命令開預覽。

```powershell
$model = '.\medicine_pipeline\runs\train_20261118_091451_575457\fit\weights\last.pt'
.\.venv\Scripts\python.exe -m focus_test.main --model $model --expected 4 --imgsz 640 --confidence 0.40 --iou 0.45 --focus-settle 0.35 --result-dir .\medicine_pipeline\runs
```

1. 點一下相機視窗，按 **D**，檢查綠框是否各自包住一瓶，且沒有框錯相鄰瓶。
2. 黃框是清晰度計分區，預設取瓶框內水平 15%–85%、垂直 45%–90%。讓想判斷清楚程度的字位在黃框內，可用 `--focus-region 0.15 0.35 0.85 0.92` 調整。
3. 按 **F** 開始掃描。掃描中 **Esc** 可取消。如果只預覽，按 **Q** 結束後再跑第 1 節的一鍵流程。
4. 若已按 **F** 完成掃描，可以直接執行第 4 節重跑 OCR 的命令，目標換成終端印出的 `scan_時間戳` 資料夾。

偵測數量不符時，先把瓶子分開並讓整個瓶身入鏡，再按 **D**。預覽可用 **M** 手動拉框，每框按 Enter，全部完成按 Esc，然後按 **F**。若正確的相鄰瓶框因重疊而被抑制，可提高 `--iou` 後再核對框線。

### 如何找到每瓶的最佳對焦

1. 記錄相機原本的自動或手動模式，固定 YOLO 瓶框，使每個焦點都量到同一瓶的同一片區域。
2. 從鏡頭回報的完整範圍做 17 點粗掃，每瓶根據自己的高分區域繼續細掃，直到可設定的最小步進。每個對焦值共用同一組取樣影格，各瓶分開計分。設定鏡頭後讀回確認，等 0.35 秒，再丟掉 3 張畫面。
3. 對接下來的 3 張影格計分，使用中位數；分數變動過大時會重試。預設用 **Laplacian variance** 排名，利用文字邊緣快速明暗變化評估對焦。也計算 **Tenengrad**（Sobel 梯度能量），供同瓶前後比較。加 `--focus-metric tenengrad` 可改用 Tenengrad 選焦。
4. 回到該瓶的最佳設定再拍攝，保存分數居中的實際影格，以及同一影格裁出的完整瓶框和計分區。每瓶各有自己的影格。全部完成或中止時，程式會嘗試還原掃描前的鏡頭設定，結果寫入 `focus_scan.json`。

`focus_curves.png` 的橫軸是鏡頭設定值，縱軸是選焦分數。通常同一瓶、同一畫面區域在更清楚時會得到較高分。瓶身文字密度、反光和曝光會影響數值，所以請比較同瓶各焦點與原始影格的分數。以下情況標成 `review`，預設先略過 OCR。

- `weak_focus_peak` 表示最佳分數沒有超過穩定粗掃中位數的 1.10 倍，或缺少可靠的粗掃基準。
- `focus_range_edge` 表示最佳值在鏡頭可設定範圍的端點。
- `verification_below_peak` 表示重新拍攝後的分數低於掃描峰值的 80%。

出現 `review` 時，可開啟 `rois/NN.png` 和 `focus/NN.png` 檢查文字，調整光源、位置或黃框後重拍。鏡頭停穩較慢可加入 `--focus-settle 0.50`。這些規則用來找出需要人工確認的照片，清晰度分數沒有跨場景通用的合格值。

### 選擇較窄的對焦範圍

以下兩個模式會保留各瓶獨立計分和驗證重拍。實測使用粗掃 11 點和 0.20 秒穩定等待。

```powershell
$model = '.\medicine_pipeline\runs\train_20261118_091451_575457\fit\weights\last.pt'
.\.venv\Scripts\python.exe -m medicine_pipeline.main live --model $model --expected 4 --focus-search autofocus --scan-samples 11 --focus-settle 0.20
.\.venv\Scripts\python.exe -m medicine_pipeline.main live --model $model --expected 4 --focus-search range --focus-range 600 710 --scan-samples 11 --focus-settle 0.20
```

- `autofocus` 啟用相機自動對焦，等待約一秒後讀取焦點，以該值為起點，從前後各 55 裝置刻度開始搜尋。若某瓶的最高分落在可擴大的區間邊界，程式會沿那一側擴大搜尋。
- `range` 只在指定範圍內搜尋，第一個設定為最接近中點的可用焦點。上例從 `655` 開始。當最佳值落在上下限時，該瓶會標記待核對。
- 省略 `--focus-search` 使用全範圍搜尋。這次兩種較窄範圍的實測結果可參考 [實測報告](report.md)。

### 用自動對焦起頭的快速模式

若只要對目前這四個固定位置的藥瓶對焦與存圖，使用先前保存的 ROI：

```powershell
$rois = '.\medicine_pipeline\runs\scan_20261118_115809_002098\roi_config.json'
.\.venv\Scripts\python.exe -m focus_test.main --scan --expected 4 --rois $rois --result-dir .\medicine_pipeline\runs --focus-search fast-autofocus
```

這條指令跑完就能在新掃描資料夾的 `rois/` 找到四張裁切圖。沿用這組 ROI 的三輪實測分別為 **4.41 秒、4.38 秒、4.40 秒**，計時含相機自動對焦、逐瓶選圖、存圖及還原相機設定。瓶子移位後，先重新定位 ROI。 [三輪紀錄](report.md#自動對焦起頭的五秒內模式)

需要接著執行辨識和查詢時，使用完整流程：

```powershell
.\.venv\Scripts\python.exe -m medicine_pipeline.main live --model $model --expected 4 --focus-search fast-autofocus
```

相機自動對焦約一秒後，程式從讀回值開始，預設在前後各 72 刻度中評估 9 個位置。所有瓶共用每個位置的影格，各瓶保留最高穩定分數對應的原尺寸照片。快速模式直接存掃描時取得的照片，省下逐瓶移鏡重拍。選中的焦點若在搜尋邊界，仍會標記待核對。

這台相機的四瓶實拍完成對焦與存圖為 **3.90–4.41 秒**，含最初自動對焦與相機設定還原。相機冷啟動、YOLO、曲線圖、OCR 和 SQLite 各自另計。 [四輪實測比較](report.md#自動對焦起頭的五秒內模式)

## 4. 用 PP-OCRv6 small 辨識並檢查文字

一鍵流程會自動讀取通過焦點品質檢查的完整瓶框，每輪只載入一次 `PP-OCRv6_small_det`（找文字區塊）和 `PP-OCRv6_small_rec`（讀出文字）。這兩個 small 模型支援繁體中文和英文標籤。 [PaddleOCR PP-OCRv6 官方文件](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

下列命令可讀取磁碟上的照片，重跑 OCR 並查詢 SQLite。只想辨識文字時，在命令末尾加 `--skip-lookup`。

```powershell
.\.venv\Scripts\python.exe -m medicine_pipeline.main ocr .\medicine_pipeline\runs\scan_20261118_091952_224326
```

若已人工檢查標成 `review` 的照片，可以在重跑命令最後加 `--include-review`，輸出仍保留原品質標示。每次 OCR 都建立新資料夾，方便比較。

| 檔案 | 打開後會看到什麼 |
| --- | --- |
| `detection_frame.png` | 開始定位時相機拍到的全畫面 |
| `roi_config.json` | 每瓶在全畫面的座標 |
| `rois/01.png` | 第 1 瓶對焦後的完整裁切圖，OCR 讀這張 |
| `focus/01.png` | 第 1 瓶實際做清晰度計分的小區域 |
| `frames/01.png` | 第 1 瓶最佳對焦值重拍的完整影格 |
| `focus_scan.json` | 瓶框、焦點、各次測量與相機設定還原結果 |
| `focus_curve.csv`、`focus_curves.png` | 逐瓶的對焦位置與清晰度曲線 |
| `ocr/run_…/summary.json` | 每瓶的清晰度、OCR 狀態和辨識文字 |
| `ocr/run_…/bottle_01/annotated.png` | 第 1 瓶被 OCR 找到的文字位置及編號 |
| `ocr/run_…/bottle_01/recognized.txt` | 每行的編號、信心值、原始文字 |
| `ocr/run_…/bottle_01/paddle_result.json` | PaddleOCR 原始輸出，含偵測與辨識資料 |
| `ocr/run_…/lookup_時間戳.json` | 每瓶的藥證字號、查詢狀態和 SQLite 候選資料 |

文字信心值是模型給該行的分數。若瓶面彎曲、文字貼近反光區或一部分標籤繞到背面，仍可能出現錯字、切行或漏字。確認某行時，將 `recognized.txt` 的編號對照 `annotated.png`，再看 `rois/NN.png` 原圖。

## 5. 查詢 SQLite 並看懂結果

資料庫是 `medicine_pipeline/sql/medicine.sqlite3`。`medicines` 表有 1,013 筆，包含中文名、英文名、規格、廠商和藥證字號。程式透過 Python 內建的 `sqlite3` 唯讀查詢。現有條碼欄位皆為空陣列，這個流程使用 OCR 的藥證字號與名稱。

`live` 和 `ocr` 會自動執行查詢。如果已經有 `summary.json`，可只重跑 SQL。以下使用這次實拍的既有 OCR 結果，執行時不會重新開相機或載入辨識模型。

```powershell
$summary = '.\medicine_pipeline\runs\scan_20261118_091952_224326\ocr\run_20261118_092134_889939\summary.json'
.\.venv\Scripts\python.exe -m medicine_pipeline.main lookup $summary
```

用另一份相同表結構的 SQLite 檔時，可在命令末尾加 `--database 'C:\資料夾\其他資料庫.sqlite3'`。每次查詢都會在 OCR 報告旁建立一份新的 `lookup_時間戳.json`。

程式從每瓶的 `rois[].lines[].text` 找出有 `藥製字` 或 `藥輸字` 前綴的六位數字號，統一全形字元與空白後，查 `license_no` 欄位。開頭的零會保留，批號等獨立數字不當成藥證。字號有唯一資料列，且 OCR 信心值至少 `0.80`、焦點與同瓶其他字號證據無待核情形時，標為 `matched`。同一字號有多筆時會全部列出，方便對照瓶身規格。

藥證字號缺少或查不到時，程式用信心值至少 `0.80` 的 OCR 行嘗試比對資料庫名稱開頭。中文至少三個漢字，英文使用完整單詞。名稱比對結果列為候選。`license_evidence` 保留 OCR 原文、行號和查過的字號；名稱候選的 `name_evidence` 可用來回查是哪一行文字命中。

| 狀態 | 意思 |
| --- | --- |
| `matched` | 字號查到一筆，且通過上述檢查 |
| `review` | 有候選，但須對照瓶身，例如同一字號有多種規格，或只有名稱命中 |
| `not_found` | 這份資料庫沒有與 OCR 證據相符的資料列 |
| `skipped` | 該瓶的 OCR 狀態未產生可查詢的文字 |

## 6. 2026-11-18 這次實拍的結果

YOLO11n 在 640 像素輸入時框出當下的 4 瓶，信心值為 0.917–0.985。逐瓶對焦後，4 瓶的 Laplacian 分數都比開始定位時高，也都通過焦點品質檢查。相機恢復原本的自動對焦模式。 [完整焦點紀錄](runs/scan_20261118_091952_224326/focus_scan.json) · [對焦曲線](runs/scan_20261118_091952_224326/focus_curves.png)

| 瓶號（左至右） | 最佳對焦值 | 原始 Laplacian | 重拍 Laplacian | 重拍 Tenengrad | OCR 行數 | OCR 原始文字節錄 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | 650 | 1,377.18 | 1,668.97 | 21,111.43 | 15 | `衛署藥製字第030620號` |
| 02 | 660 | 3,248.77 | 3,526.62 | 29,457.37 | 16 | `艾莫斯克` |
| 03 | 702 | 1,672.43 | 3,768.63 | 47,141.46 | 13 | `佐淨菌`、`衛署藥製字第049651號` |
| 04 | 631 | 1,193.23 | 2,287.30 | 40,359.10 | 9 | `得術泰`、`Dynastat` |

OCR 共輸出 53 行。有些靠近瓶身弧面邊緣的行仍缺字，例如第 2 瓶的 `AmoClav IV Powder for Injed`。這些結果原樣保留，便於後續校對與查詢。 [OCR 完整結果](runs/scan_20261118_091952_224326/ocr/run_20261118_092134_889939/summary.json)

第一次查詢時，第 1、3 瓶的藥證字號查無資料。依提供的許可證截圖補入兩筆後，以同一份 OCR 結果重查。

| 瓶號 | 查詢依據 | SQLite 結果 |
| --- | --- | --- |
| 01 | `衛署藥製字第030620號` | 命中賜福樂信注射劑，750 mg |
| 02 | `衛署藥製字第057877號` | 艾莫斯克靜脈乾粉注射劑有 600 mg 和 1.2 g 兩筆，需核對規格 |
| 03 | `衛署藥製字第049651號` | 命中佐淨菌靜脈輸液，5 mg/mL |
| 04 | `得術泰`、`Dynastat` | 找到一個名稱候選，資料庫列為得術泰注射劑 40 毫克 |

[補入後的 SQLite 查詢結果](runs/scan_20261118_091952_224326/ocr/run_20261118_092134_889939/lookup_20261118_102637_845329.json)

同組門檻另測了一張既有的相鄰 5 瓶畫面，640 像素輸入只找出 4 個瓶框。960 和 1280 的輸入也未在該畫面得到 5 個正確完整框，且在本次 4 瓶參考影格增加了重複框。因此本次設定保留 640，對相鄰或遮擋的場景要先檢查框線。

## 7. 在另一台 Windows 電腦準備環境

1. 安裝 64 位元 Python 3.10。這台電腦實測使用 Python 3.10.11、Torch 2.13.0 CPU、Ultralytics 8.4.105、PaddleOCR 3.7.0、PaddlePaddle 3.2.0、OpenCV 4.12.0。
2. 在專案根目錄執行下列命令建立環境和安裝套件。已有可使用的 `.venv` 時可略過。

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install 'torch==2.13.0' 'torchvision==0.28.0' --index-url https://download.pytorch.org/whl/cpu
   .\.venv\Scripts\python.exe -m pip install 'ultralytics==8.4.105' 'paddleocr==3.7.0' 'paddlepaddle==3.2.0' 'paddlex==3.7.2' 'opencv-python==4.12.0.88' 'numpy==2.2.6'
   ```

3. 下載官方權重並解壓到 `medicine_pipeline/models/`。此步需要網路。 [Ultralytics 權重發佈頁](https://github.com/ultralytics/assets/releases/tag/v8.3.0) · [PaddleOCR 模型清單](https://www.paddleocr.ai/main/en/version3.x/model_list.html)

   ```powershell
   $destination = '.\medicine_pipeline\models'
   New-Item -ItemType Directory -Force -Path $destination | Out-Null
   curl.exe -fL 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' -o "$destination\yolo11n.pt"
   if ($LASTEXITCODE -ne 0) { throw 'YOLO11n 權重下載失敗' }
   $source = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0'
   foreach ($name in 'PP-OCRv6_small_det', 'PP-OCRv6_small_rec') {
       $archive = "$destination\${name}_infer.tar"
       curl.exe -fL "$source/${name}_infer.tar" -o $archive
       if ($LASTEXITCODE -ne 0) { throw "下載失敗：$name" }
       tar.exe -xf $archive -C $destination
       if ($LASTEXITCODE -ne 0) { throw "解壓失敗：$name" }
   }
   ```

4. 有 `focus_test/native/build/Release/focus_control.exe` 就可使用相機。若檔案缺少，在裝有 CMake、Visual Studio C++ 桌面開發工具和 Windows SDK 的 **Developer PowerShell** 執行下列命令來編譯。

   ```powershell
   cmake -S focus_test/native -B focus_test/native/build -A x64
   cmake --build focus_test/native/build --config Release
   ```

5. 將 `medicine.sqlite3` 放入 `medicine_pipeline/sql/`。依第 2 節訓練並取得 `checkpoint`，再依第 1 節跑一輪。若 Windows 回報 `0x80070005` 或開不了相機，先關閉其他相機程式並檢查 **設定 → 隱私權與安全性 → 相機** 的桌面應用程式存取。重新執行 `--focus-info` 確認。
