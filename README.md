# FSDD Digits — fast, tiny, and live

Riad Dajani

**Sumary of Results:***
[train] loss=0.0123 acc=0.996 | val_acc=0.993 f1=0.993
[eval] accuracy=0.9967, macroF1=0.9967
Early-exit p>0.88

**Goal:** “audio in → digit out” with **very low latency** and **high accuracy**, plus a few things that push the envelope for a tiny model: **early exit**, **live enrollment** (few-shot prototypes), and **interpretability** in both *training* and *live* runs.

---

## TL;DR

* **Features:** fixed-size **MFCC 64×64** (or 3-ch log-mel in earlier experiments), **dataset-level normalization**.
* **Model:** **3×Conv → MaxPool → 3×Dense (BN+ReLU)** with dropout in the MLP, **Adadelta** optimizer, CE loss.
* **Live:** streaming mic with **VAD**, **early-exit** when confidence is high, **proto head** for enrolled examples.
* **Interpretability:** prints a **confusion matrix** each epoch; during live, prints a brief **“why”** (most salient MFCC/mel bands).
* **Key results (what you should expect):** On the clean FSDD test split, this family of models typically reaches **\~97–99%** accuracy after 30–50 epochs. If you see much lower, check the troubleshooting notes (normalization and feature shape are the usual culprits).

> I intentionally **built on prior, well-tested baselines** (https://github.com/adhishthite/sound-mnist) for FSDD to keep things efficient and robust; then added:
>
> * **Early-exit** (for faster, confident decisions),
> * **Enrollment & prototypes** (adapt in seconds to a new speaker/mic),
> * **Live loop & interpretability** (confusion matrices + “why” heat-hints during live).

---

## Setup

### 1) Python & system deps

* Python **3.10+** recommended
* macOS (Homebrew):

  ```bash
  brew install portaudio  # needed for the 'sounddevice' mic input
  ```
* Linux:

  * Make sure ALSA/PulseAudio dev packages are available (for PortAudio).
* (Optional) FFmpeg if you plan to experiment with other audio formats.

### 2) Create a virtual env and install Python deps

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install numpy torch librosa sounddevice datasets scikit-learn joblib
```

> If you see build errors for `sounddevice`, ensure PortAudio is installed (see step 1), then `pip install sounddevice` again.

### 3) Hugging Face access (dataset)

You can either **login** or **set an env var**:

* Easiest:

  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```
* Or export a token (works well in CI/containers):

  ```bash
  export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
  ```

  (If you already use another variable name in your environment, you can also set `HF_TOKEN` to the same value.)

---

## Training

I default to **Adadelta**, **batch=64**, **epochs=50**, and **mild augmentation** (small time-shift + light noise):

```bash
python fsdd_demo.py train --epochs 50 --batch-size 64 --augment 1
```

What you’ll see:

* Per-epoch **train loss/accuracy**.
* **Confusion matrix** on the FSDD test split.
* Best checkpoint auto-saved (e.g., `cnn_model.pt`).

**Tip:** If accuracy stalls <90% on clean FSDD, something’s off with normalization or feature shapes. See **Troubleshooting** below.

---

## Evaluation

```bash
python fsdd_demo.py eval
```

Prints **test accuracy** and **macro-F1** using the saved checkpoint.

---

## Live demo & enrollment

### Default live settings (prototype mode)

Prototype mode blends the CNN and the enrolled (few-shot) prototype head for robust, low-latency predictions:

```bash
python fsdd_demo.py live --mode proto
```

### Faster early-exit (less stable)

Lower the threshold and require fewer consecutive checks:

```bash
python fsdd_demo.py live --mode base --ee-thresh 0.88 --ee-consec 2 --ee-every 2
```

### Enroll your own digits (few-shot)

You can enroll multiple times to average different takes (1–3 passes recommended):

```bash
python fsdd_demo.py enroll
```

Then test your prototypes live again:

```bash
python fsdd_demo.py live --mode proto
```

**Notes**

* On first run, macOS will ask for **microphone permission**. Grant it to your shell/terminal app.
* VAD knobs (`--vad-…`) are exposed if you need to tweak thresholds in a noisy room.

---

## Approach (short)

1. **Features:** I adopted a **fixed 64×64 MFCC image** per utterance (or 3-channel log-mel + Δ + ΔΔ in our other script). I use **absolute dB/reference** and compute **dataset-level normalization** (`mean/std` per band), ensuring stable statistics across files/sessions. (cred: https://github.com/adhishthite/sound-mnist)

2. **Model:** A compact CNN with **3 conv layers** and **1 max-pool** followed by a **3-layer MLP** (BatchNorm after every conv and dense, ReLU activations, dropout in the MLP). Optimized with **Adadelta**; loss is **categorical cross-entropy**.

3. **Early-exit:** During streaming inference, I compute features over growing buffers and **exit early** when confidence passes a threshold for N consecutive checks. This reduces perceived latency substantially when the model is sure early.

4. **Enrollment & prototypes:** I support **one-shot (few-shot) enrollment**. I extract the CNN embedding and store **per-digit prototypes** (cosine space). At inference I can blend CNN probabilities with prototype similarities (**hybrid**), improving robustness across mics/voices.

5. **Interpretability:**

   * Training prints a **confusion matrix** each epoch (see where it’s confused).
   * Live mode returns **“why”** by highlighting the MFCC bands with the strongest gradient saliency for the predicted class.

---

## Key results (what to expect)

* On the **clean FSDD test split**, this MFCC/CNN family is known to reach **\~97–99% accuracy** after \~30–50 epochs with light regularization.
* **Latency:** End-to-end decision time is dominated by VAD and buffer size. The model itself is tiny; **feature+forward** runs in **milliseconds** on CPU. Early-exit often triggers **well before utterance end**, improving responsiveness.

> If your numbers are far below this, 99% of the time the cause is **mismatched normalization stats** (old `*.npz` file from a different feature setup) or a **feature shape mismatch**. See below.

---

## Tips & troubleshooting

* **Wipe stale artifacts** when you change features or settings:

  ```bash
  rm -f mfcc_norm.npz mel_norm.npz cnn_model.pt cnn_prototypes.joblib
  ```
* **Normalization must match the feature shape.** The code computes and caches `*_norm.npz` once. If you switch MFCC↔mel or change band counts, **recompute**.
* **Sanity pass (no aug):** If you’re debugging learning issues, do one short run with `--augment 0` to confirm it can overfit the clean split.
* **Live VAD too strict?** Lower the gate and min length:

  ```bash
  python fsdd_demo.py live --mode proto --vad-thresh-mult 1.5 --vad-min-len 100
  ```
* **Mic levels:** If the live stream seems “too quiet,” the code applies a tiny RMS lift, but you can also move closer to the mic or increase input gain.

---

## File map (what matters)

* `fsdd_demo.py` — single entry point:

  * `train` / `eval`
  * `live` (mic streaming, early-exit)
  * `enroll` (few-shot prototypes)
* Artifacts:

  * `cnn_model.pt` — best checkpoint
  * `*_norm.npz` — dataset normalization
  * `cnn_prototypes.joblib` — enrolled prototypes

---

## Acknowledgments

* **Free Spoken Digit Dataset (FSDD)** — open dataset of spoken digits at 8 kHz.
* Prior MFCC/CNN baselines in the community informed our **64×64 MFCC** choice and training recipe; I layered on **early-exit**, **prototypes**, and **live interpretability** to make it practical and fun to use in the real world.
* https://github.com/adhishthite/sound-mnist for model architecture and pipeline setup
