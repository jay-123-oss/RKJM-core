# RKMJ External 3-4B Model Quantization Test Suite

**Location:** `/home/jaydeep/Documents/rkjm-core/TEST/quantization_test/`

> **Note:** This entire folder is strictly isolated **OUTSIDE** the core framework directory (`rkmj-core/`). The script imports `rkmj` and `universal` externally.

---

## ⚡ 2 Ways to Run (Automatic Download or Manual Upload)

### Option 1: Manual Model Upload (Fastest - No Download Waiting!)
Agar aapne model pehle se download kiya hua hai ya kisi fast downloader (IDM, browser, etc.) se download kiya hai:

1. Model ke files ko is folder ke andar daal dijiye:
   `/home/jaydeep/Documents/rkjm-core/TEST/quantization_test/manual_model/`
   *(Zaroori files: `config.json`, `*.safetensors`, aur `tokenizer*`)*

2. Aur simply run kijiye:
   ```bash
   cd /home/jaydeep/Documents/rkjm-core
   ./myenv/bin/python TEST/quantization_test/quantize_and_test.py
   ```
   *(Script automatic `manual_model/` ko detect karke bina kisi internet download ke direct quantize kar degi!)*

**Ya fir agar aapka model kisi aur folder me rakha hai:**
```bash
./myenv/bin/python TEST/quantization_test/quantize_and_test.py \
  --local-dir /path/to/your/downloaded/model
```

---

### Option 2: Automatic Internet Download
Agar internet fast hai aur direct Hugging Face se download karwana ho:
```bash
./myenv/bin/python TEST/quantization_test/quantize_and_test.py \
  --model-id "Qwen/Qwen2.5-3B-Instruct"
```
