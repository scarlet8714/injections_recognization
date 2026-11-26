# 手部偵測實驗

使用即時相機或錄影觀察手部位置。放置位置只需要提供原始影格上的像素座標。

## 安裝及回放

從專案根目錄執行。以下使用 64 位元 Python 3.13，套件安裝在獨立的虛擬環境。

```powershell
py -3.13 -m venv .venv-hand
.\.venv-hand\Scripts\python.exe -m pip install -r .\hand_test\requirements.txt
$model = '.\.venv-hand\hand_landmarker.task'
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile $model
.\.venv-hand\Scripts\python.exe -m hand_test.replay .\放瓶錄影.mp4 --model $model --output .\手部回放.jsonl --point 540,720 --point 890,720
```

上方是參考座標範例，原點在原始影片左上角，向右及向下分別是 X、Y 正方向。可重複指定 `--point` 或省略。預覽以黃色十字標示放置參考點，以綠點標示偵測到的手部關鍵點。按 Q 或 Esc 可結束回放，已處理的結果會保留。重跑請指定新的輸出路徑。加上 `--no-preview` 可直接產出紀錄。

## 判讀紀錄

首行記錄來源、模型、套件版本及放置參考點。每張影格的 `timestamp_ms` 使用來源影片的顯示時間，以第一張有顯示時間的影格為零點。可直接拿來對照原影片，包含可變影格率的錄影。無法讀到時間時標為不可用。`processing_ms` 是本機偵測耗時，從轉換影格色彩到取得手部位置，與影片內的時間分開記錄。

| 狀態 | 意義 |
| --- | --- |
| `detected` | 偵測到一隻或多隻手。`hands` 是各隻手的關鍵點座標，每個點為按影像寬高正規化的 `[x, y]`。 |
| `no_hand` | 這張有效影格未偵測到手。 |
| `read_error` | 影格、影片軌或顯示時間無法使用。 |
| `processing_error` | 影格可讀，偵測處理發生錯誤。 |

末行彙整各狀態的張數。人工或由代理比對時，可先在原始影片標記伸手、停留、遮擋與收手的區間，再用同一時間軸查找漏檢及偵測中斷。建議保留原始影片和 JSONL，供相同片段重跑。

## 即時相機與同步錄影

以現有 USB 相機即時執行相同的手部偵測。看到預覽後可做放瓶、停留和收手等動作，完成後按 Q 或 Esc。

```powershell
.\.venv-hand\Scripts\python.exe -m hand_test.live --model $model --output .\實拍\即時紀錄.jsonl --record .\實拍\原始影片.mp4
```

`--record` 可省略。保留錄影時，影格時間以第一張實際取得的相機影格為零點，可用同一時間軸比對影片與紀錄。即時紀錄與錄影請各指定新的路徑。`--seconds` 可指定何時結束，`--camera-index` 可切換相機。末行的 `processed_fps` 是這次實驗取得並處理的影格頻率，會受到預覽和同步錄影影響。

## 色彩候選模式

使用 `--detector color` 在即時預覽顯示最大的疑似手部區塊。綠框可以核對候選位置，黃色十字仍用來標示放置參考點。

```powershell
.\.venv-hand\Scripts\python.exe -m hand_test.live --detector color --output .\實拍\色彩紀錄.jsonl --record .\實拍\驗證影片.mp4
.\.venv-hand\Scripts\python.exe -m hand_test.replay .\實拍\驗證影片.mp4 --detector color --output .\實拍\色彩回放.jsonl
```

本模式把影格縮至 480 × 270，使用 YCrCb 色彩遮罩，保留面積至少 648 像素的最大連通區塊。逐影格的 `region_area_px` 保留最大區塊面積。`detected` 表示已出現合乎門檻的候選，`hand_regions` 保存綠框的正規化 `[x1, y1, x2, y2]` 座標。

目前用這個模式驗證不同放置動作及背景下的候選區塊。預覽與紀錄供人工核對，放瓶參考點可以在畫面上輔助比對。

## 實測摘要

前三段實拍使用同一台 USB 相機與固定工作台取景。下列張數是從人工核對的清楚區間統計。

| 情境 | 色彩候選結果 |
| --- | --- |
| 第一瓶握瓶停留 | 112／112 張偵測到手。空桌與只剩藥瓶的 432 張均無誤判。 |
| 換瓶與手快速掠過 | 掠過 9／9 張、握瓶 93／93 張偵測到手。收手後只剩藥瓶的 235 張均無誤判。 |
| 即時放瓶後回手碰原瓶 | 偵測到兩段連續訊號，共 77 張及 53 張。即時與回放的 582 張判定逐張一致。 |

即時實測解析度為 1920 × 1080，處理頻率為 29.52 FPS，單張色彩判定耗時 P95 為 2.05 毫秒。兩次收手時，候選區塊降低到門檻以下的影格上緣仍有淡色手影。後續可依各格取放範圍驗證拍攝時機。不同照明、操作者與手套仍待實拍。

回歸測試可從專案根目錄執行。

```powershell
.\.venv-hand\Scripts\python.exe -m unittest hand_test.test_live hand_test.test_replay focus_test.test_focus focus_test.test_scan
```
