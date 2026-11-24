# Focus test

獨立測試：預設使用既有 YOLO 模型定位藥瓶，固定 ROI 後，從原始解析度計算
每瓶的 Laplacian variance 與 Tenengrad。保留背景差分與手動框選作比較。
依賴現有 `.venv` 的 `opencv-python`、`numpy`、`ultralytics`、`torch`。
模型為 `segmentation_roi/model/best.pt`，預設 `imgsz=1280`、confidence=0.5，
只在按 D 時推論，模型快取後重用；清晰度計分不需要反覆推論。
預設透過 OpenCV DirectShow
直接讀取一般 USB 攝影機，不啟動 C++ camera server。

## 相機測試

從專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py
```

預設為 USB camera index 0。如果畫面不是要使用的相機，可切換 index：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --camera-index 1
.\.venv\Scripts\python.exe .\focus_test\main.py --camera-index 0 --width 1920 --height 1080
```

預設要求 **1920×1080**；也可用 `--width`、`--height` 修改。裝置可能協商成其他解析度，
啟動時會印出實際值。預覽最長邊縮為 1280，保存影像與計分仍使用相機實際回傳的原始解析度。
開啟失敗時，先關閉其他占用該 USB 相機的程式。若要使用專案原本的 DFK / IC 3.5
後端，另以 `--backend ic35` 啟動；該模式才會使用 C++ camera server。

1. 放好五瓶，確保完整瓶身在畫面內；固定相機、光源、解析度、曝光、Gain、白平衡。
2. 按 **D** 使用 YOLO 偵測並固定 ROI，不必移走藥瓶或拍空背景。檢查每瓶是否各有一個綠框。
3. 黃框是實際計分區域，預設取瓶框內水平 15–85%、垂直 45–90%；這只是幾何內縮，
   **不是自動辨識標籤**。確認黃框涵蓋要評估的文字，沒有混入鄰瓶。
4. ROI 固定後畫面持續更新分數，可調對焦比較同一瓶的變化。藥瓶移位後需重新選 ROI。
5. **Space** 保存原圖、各瓶圖、計分區圖、標註圖、JSON 與可重用的 ROI 設定。
6. **C** 清除 ROI；**Q / Esc / 關閉主視窗**結束。

## 逐瓶對焦與選圖

按 **D**（或 **M**）確認五個框後，按 **F**，程式會依每瓶的清晰度分數選焦。

1. 用 DirectShow `IAMCameraControl::GetRange` 讀取該相機的最小值、最大值及步進。
2. 關閉自動對焦，以合法刻度粗掃全範圍，再逐層細掃各瓶的高分區域，直到最小步進。每個對焦值取一組影格供各瓶 ROI 計分，其他瓶需要同一個刻度時可重用已算出的分數。
3. 每次改值後，持續讀取影格等待鏡頭穩定，再丟棄 3 張，取 3 張的清晰度中位數排名。
   若樣本分數差異超過 20%，會再等候並重取，最多重試 2 次；仍不穩定的點不參與選焦。
4. 回到該瓶的最佳對焦值，再拍 3 張，保存分數居中的影格及裁切圖，接著處理下一瓶。
5. 全部完成、失敗或取消時，還原掃描前的手動值，或原本的自動模式；自動模式的數值會繼續變動。
   掃描中按 **Q / Esc** 可取消。

只有恰好五個已確認 ROI 才會開始（可用 `--expected` 修改數量）。若模型漏抓，先用 M 框選。
瓶子與相機全程需固定；曝光、Gain、白平衡、光源也應穩定，以免分數被亮度或雜訊影響。
預設以 Laplacian variance 排名，可改 `--focus-metric tenengrad`。
這是對「取樣到的焦點」做粗到細搜尋，不保證全範圍數學上的最高點。
若有移動、鏡頭穩定時間不足或驗證分數明顯下降，應檢查影像後重新掃描。

本機 `Usb HDCamera`（index 0）實測回報：**0～1023、步進 1、預設 192**；
這是 1024 個可設定位置，不代表 1024 個距離，也不能直接換算成毫米。
不同驅動可能使用自訂刻度；程式會查詢裝置，不把這個範圍套用到其他相機。
參考 [Microsoft 對焦屬性說明](https://learn.microsoft.com/en-us/windows-hardware/drivers/stream/ksproperty-cameracontrol-focus)。

只讀取對焦資訊，或直接執行一輪不開預覽視窗的掃描：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --focus-info
.\.venv\Scripts\python.exe .\focus_test\main.py --scan
.\.venv\Scripts\python.exe .\focus_test\main.py --scan --rois <roi_config.json>
```

掃描設定範例（鏡頭移動較慢時提高 settle）：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --focus-settle 0.5 --focus-discard 5 --focus-samples 3
```

輸出為 `result/scan_時間戳/`：`rois/01.png`～`05.png` 是各瓶獨立對焦後的圖，
`frames/` 保留各自完整影格，`focus/` 是計分區域；`focus_scan.json` 記錄原始模式、
每次設定／讀回值、取樣分數、選中的焦點及還原結果，`focus_curve.csv` 可用於分析。
成功後另產生 `focus_curves.png`，顯示五瓶各自的焦點曲線。
`detection_frame.png` 與 `roi_config.json` 保存掃描開始時的定位依據。
`Space` 仍只保存當前影格，**F 才會控制鏡頭做逐瓶掃描**。

### 編譯 USB 對焦控制元件

原生元件只依賴 Windows SDK／MSVC，不依賴 IC 3.5；目前工作區已編譯。
新環境需在可使用 CMake 的 Visual Studio Developer PowerShell 執行：

```powershell
cmake -S focus_test/native -B focus_test/native/build -A x64
cmake --build focus_test/native/build --config Release
```

產生 `focus_test/native/build/Release/focus_control.exe`；此元件讀寫相機控制，
影格仍由 OpenCV DirectShow 取得。程式會驗證手動設定後的讀回值，避免把不生效的設定當成成功。

模型仍可能漏偵測或框不準，尤其遇到遮擋或不同外觀的藥瓶，需檢查實際框線。
**M** 可改用手動框選：拖曳一瓶後按 Enter，重複五次，最後按 Esc 結束。
也可直接框選各瓶的標籤，搭配 `--focus-region 0 0 1 1` 計算整個手選區。
不會強制湊五個框。數量不符仍會保存，JSON 標示 `needs_roi_review`；
數量為五也不代表一定正確，必須先看框。

設定範例：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --imgsz 1280 --confidence 0.5
.\.venv\Scripts\python.exe .\focus_test\main.py --focus-region 0 0 1 1
```

## 五張對焦圖的 OCR 測試

使用專案現有的 `PP-OCRv5_server_det` 與 `PP-OCRv5_server_rec`，一次載入模型，
逐張測試掃描輸出的 `rois/`；不重新拍照或控制相機。

```powershell
.\.venv\Scripts\python.exe .\focus_test\ocr_test.py .\focus_test\result\scan_20261114_133838_718539
```

輸出在該掃描資料夾的 `ocr/run_時間戳/`，每瓶包含原始 OCR JSON、標註圖及
`recognized.txt`，總表為 `summary.json`。預設不加影像前處理，保留低信心文字作分析。
信心值只代表模型對已偵測文字的分數，不等於正確率，也不反映漏偵測的文字。

## 背景差分比較模式

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --roi-method background
```

此模式才使用 `absdiff → Threshold → Morphology → Connected Components`，不載入模型。
先移走藥瓶按 **B** 保存空背景，再放回藥瓶按 **D**。需要相機、曝光、光線穩定，且瓶子分開。
11/14 兩次實拍的原始結果出現全畫面大框；其中一張遮罩約 71% 為前景，
空背景與放入藥瓶後的背景亮度差超過原設門檻，推測與自動曝光有關。
離線亮度補償仍無法可靠分開瓶子，因此此模式不再作為預設。
若區塊的框超過全圖 85%，程式會拒絕套用，保留原有 ROI，並提示改用模型或手選。

`--threshold` 為 0–255 的差分門檻；`--min-area-ratio` 是區塊面積佔全圖的最小比例；
`--morph-size` 是縮小偵測圖上的 closing kernel，須為正奇數。
偵測預設最長邊 1280，座標映回原圖，裁切與評分不縮放；可用 `--detect-width 4128`
測試更細的間隙。按 D 後顯示的 mask 是該次偵測畫面，不是即時更新的遮罩。
改變相機解析度會清除背景與 ROI。改變曝光、燈光等條件後，也應重新拍空背景。
對焦變動會改變背景細節，因此建議先完成定位，再固定框評估對焦。

## 離線測試

預設使用模型，只需藥瓶照片：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --image <藥瓶照片.png>
```

背景差分模式需相同拍攝設定的空背景：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --roi-method background --image <藥瓶照片.png> --background <空背景.png>
```

或載入先前保存的 `roi_config.json`，在相同解析度與擺放位置的照片上比較：

```powershell
.\.venv\Scripts\python.exe .\focus_test\main.py --image <照片.png> --rois <roi_config.json>
```

設定檔格式，座標為原圖像素，右、下界不包含在 ROI 中：

```json
{"image_size": [4128, 3096], "boxes": [[100, 200, 500, 1000]]}
```

## 輸出及解讀

每次 Space 或離線執行都建立 `focus_test/result/run_時間戳/`：

- `capture.png`、`annotated.png`：完整原圖與框線分數。
- `rois/01.png`…：依由左到右排序的完整 ROI。
- `focus/01.png`…：實際計分的區域。
- `focus_results.json`：座標、原始分數、偵測參數及 ROI 來源；YOLO 模式包含分類與信心值。
- `roi_config.json`：可重用的固定框。
- `background.png`、`mask.png`：使用背景差分時保存的背景與偵測遮罩。

分數不做銳化、縮放或去噪；Laplacian 使用灰階、`CV_64F`、`ksize=1`，
Tenengrad 使用 3×3 Sobel 的平均梯度能量；兩者排除一像素濾波邊界。
PNG 避免新增 JPEG 壓縮影響。

使用 `--backend ic35` 啟動 camera server 時，原始輸出會保存到 `result/camera_server_時間戳.log`。
啟動失敗會顯示退出碼、SDK 原始錯誤與日誌路徑。若出現
`No IC Imaging Control 3.5 camera found`，代表 SDK 未列舉到相機，需檢查
USB／電源、驅動與目前執行環境的裝置存取權限。

目前不設定通用的清楚／模糊門檻；按 F 才會依每瓶的焦點曲線選擇影格。
應比較同一固定 ROI、相同成像設定下的影格；不同標籤文字量、反光、雜訊都會影響數值，
不能直接用五瓶的分數互相比較清晰度。ROI 編號只代表左右順序，沒有物件追蹤。

## 驗證

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s focus_test -p "test_*.py" -v
```

測試涵蓋分離區塊、相連合併、背景尺寸不符、全畫面差分拒絕、原圖座標映射、
模糊影像分數下降、相機生命週期及結果輸出。
另以具有五個已知焦點的模擬相機驗證五次獨立搜尋、合法步進、圖片與焦點對應、
取消／裝置失敗時還原，以及拒絕未生效的手動設定。
另外以 `run_20261114_131818_882593`、`run_20261114_131836_727346` 的實拍原圖，
離線驗證既有模型在 `imgsz=1280`、confidence=0.5 下的定位結果。
驗證輸出保存於 `focus_test/result/diagnosis/model_fix/`，有限樣本不能代表所有擺法。
