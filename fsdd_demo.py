#!/usr/bin/env python3
"""
digits_cnn_mfcc.py — FSDD spoken-digit recognizer (MFCC 64x64, Adadelta)

Implements the architecture & hyperparams described in the "sound-mnist" repo:
- MFCC "image" of fixed size 64x64
- Model: 3 Conv (BN+ReLU) → 1 MaxPool → 3 Dense (BN+ReLU) with Dropout on hidden layers
- Optimizer: Adadelta (default lr), Loss: CrossEntropy, Batch=64, Epochs=50
Ref: https://github.com/adhishthite/sound-mnist (README)  [see live citations from the assistant message]

Includes:
- Global dataset normalization saved to mfcc_norm.npz
- One-shot prototypes (cosine) and hybrid head
- Live microphone with early-exit and VAD
"""

import argparse, os, sys, time, math, queue, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import joblib
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, Audio

try:
    import sounddevice as sd
except Exception:
    sd = None

# -------------------- Config --------------------

SR = 8000

# Build fixed 64x64 MFCC "images"
N_MFCC = 64
N_FFT = 512            # ~64 ms at 8 kHz
WIN_LENGTH = 256       # ~32 ms
HOP_LENGTH = 128       # ~16 ms → ~62–64 frames for ~1s audio
FMIN, FMAX = 20, 4000

MODEL_PATH = "cnn_model.pt"
PROTO_PATH = "cnn_prototypes.joblib"
NORM_PATH = "mfcc_norm.npz"
DEVICE = "cpu"  # keep portable & ultra-low-latency

# -------------------- Features --------------------

def preemphasis(x, coef=0.97):
    return np.append(x[0], x[1:] - coef * x[:-1]) if len(x) > 1 else x

def mfcc_64x64(y: np.ndarray, sr: int=SR) -> np.ndarray:
    """Return a (1, 64, 64) MFCC 'image': freq=64 coeffs, time=64 frames (pad/center-crop)."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="kaiser_fast")
        sr = SR
    # ensure a minimum length for stable framing
    if len(y) < WIN_LENGTH:
        y = np.pad(y, (0, WIN_LENGTH - len(y)), mode="edge")
    y = preemphasis(y)

    # MFCC: use 64 mel filters and 64 cepstral coeffs to get (64, T)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MFCC, fmin=FMIN, fmax=FMAX
    ).astype(np.float32)  # (64, T)

    # Time pad or center-crop to exactly 64 frames
    T = mfcc.shape[1]
    if T < 64:
        pad = 64 - T
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="edge")
    elif T > 64:
        # center-crop
        start = (T - 64) // 2
        mfcc = mfcc[:, start:start+64]

    # Now exactly (64,64). Add channel dim.
    X = mfcc[None, :, :]  # (1, 64, 64)
    return X

# -------------------- Dataset & Normalization --------------------

def load_fsdd_paths():
    ds = load_dataset("silky1708/Free-Spoken-Digit-Dataset")
    ds = ds.cast_column("audio", Audio(decode=False))
    def extract(split):
        return [(ex["audio"]["path"], int(ex["label"])) for ex in ds[split]]
    return extract("train"), extract("test")

def compute_global_mfcc_stats(items, max_files=None):
    """Compute per-coeff mean/std across time over the training set, shape (64,1)."""
    rng = list(range(len(items)))
    random.shuffle(rng)
    if max_files is not None:
        rng = rng[:min(max_files, len(rng))]
    bands = []
    for i in rng:
        path, _ = items[i]
        y, sr = librosa.load(path, sr=None, mono=True)
        X = mfcc_64x64(y, sr)[0]   # (64,64)
        bands.append(X)            # per-file (64,64)
    A = np.concatenate([b.reshape(N_MFCC, -1) for b in bands], axis=1)  # (64, sumT)
    mean = A.mean(axis=1, keepdims=True).astype(np.float32)
    std  = (A.std(axis=1, keepdims=True) + 1e-6).astype(np.float32)
    return mean, std

class FSDDMFCCDataset(Dataset):
    def __init__(self, items, mfcc_mean, mfcc_std, augment=0, train=True):
        self.items = items
        self.augment = augment
        self.train = train
        self.mean = mfcc_mean  # (64,1)
        self.std  = mfcc_std   # (64,1)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y, sr = librosa.load(path, sr=None, mono=True)

        # Light waveform aug (helps live robustness): ≤50 ms shift + SNR 20–30 dB
        if self.train and self.augment > 0:
            max_shift = int(0.050 * (sr if sr else SR))
            s = np.random.randint(-max_shift, max_shift+1)
            if s > 0: y = np.concatenate([np.zeros(s), y[:-s]])
            elif s < 0: y = np.concatenate([y[-s:], np.zeros(-s)])
            rms = np.sqrt(np.mean(y**2) + 1e-9)
            noise_rms = rms / (10**(np.random.uniform(20,30)/20.0))
            y = (y + np.random.randn(len(y))*noise_rms).astype(np.float32)

        X = mfcc_64x64(y, sr)      # (1,64,64)
        # Normalize per-coeff (broadcast across time)
        X[:, :, :] = (X[:, :, :] - self.mean) / self.std
        X_t = torch.from_numpy(X)  # (1,64,64)
        label_t = torch.tensor(label, dtype=torch.long)
        return X_t, label_t

def collate_stack(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)  # (B,1,64,64)
    y = torch.stack(ys, dim=0)
    lengths = torch.full((len(batch),), 64, dtype=torch.long)  # fixed 64 frames
    return X, y, lengths

# -------------------- Model: 3 Conv + 1 MaxPool + 3 Dense --------------------

class SoundMNISTNet(nn.Module):
    """
    3 Conv (BN + ReLU) → MaxPool → 3 Dense (BN + ReLU) with Dropout on hidden dense layers.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv stack
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        # Flatten: after conv3, spatial size is 32x32, channels=128 → 131072
        self.flat_dim = 128 * 32 * 32

        # MLP (3 dense total: two hidden + output), BN after each hidden
        self.fc1 = nn.Linear(self.flat_dim, 256, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.do2 = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,1,64,64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.fc1(x); x = F.relu(self.bn_fc1(x)); x = self.do1(x)
        x = self.fc2(x); x = F.relu(self.bn_fc2(x)); z = self.do2(x)  # embedding
        logits = self.fc3(z)
        return logits, z

# -------------------- Train / Eval --------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train(epochs=50, batch_size=64, lr=None, augment=1, max_norm_files=None):
    set_seed(42)
    train_items, test_items = load_fsdd_paths()

    # (Re)build or load global MFCC stats (64x1) for this exact pipeline
    recompute = True
    if os.path.exists(NORM_PATH):
        try:
            norm = np.load(NORM_PATH)
            if norm["mean"].shape[0] == N_MFCC:
                mfcc_mean, mfcc_std = norm["mean"], norm["std"]
                recompute = False
        except Exception:
            pass
    if recompute:
        print("[norm] computing (fresh)…")
        mfcc_mean, mfcc_std = compute_global_mfcc_stats(train_items, max_files=max_norm_files)
        np.savez(NORM_PATH, mean=mfcc_mean, std=mfcc_std)
        print(f"[norm] saved to {NORM_PATH}")
    else:
        print(f"[norm] Loaded {NORM_PATH}")

    train_ds = FSDDMFCCDataset(train_items, mfcc_mean, mfcc_std, augment=augment, train=True)
    test_ds  = FSDDMFCCDataset(test_items,  mfcc_mean, mfcc_std, augment=0, train=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_stack)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_stack)

    model = SoundMNISTNet().to(DEVICE)

    # Optimizer: Adadelta (default lr=1.0 unless overridden)
    opt = torch.optim.Adadelta(model.parameters()) if lr is None else torch.optim.Adadelta(model.parameters(), lr=lr)

    # Loss: CrossEntropy (categorical crossentropy)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        t0 = time.time()
        for X, y, lengths in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * y.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            n += y.size(0)

        train_loss = tot / n
        train_acc = correct / n

        acc, f1, cm = evaluate(model, test_dl, return_cm=True)
        dt = time.time() - t0
        print("Confusion (rows=true, cols=pred):\n", cm)
        print(f"[ep {ep:02d}] loss={train_loss:.4f} acc={train_acc:.3f} | val_acc={acc:.3f} f1={f1:.3f} | {dt:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "meta": {"sr": SR, "n_mfcc": N_MFCC}}, MODEL_PATH)
            print(f"  ↳ saved best to {MODEL_PATH} (val_acc={best_acc:.3f})")

def evaluate(model=None, dl=None, return_cm=False):
    if not os.path.exists(NORM_PATH):
        train_items, _ = load_fsdd_paths()
        mean, std = compute_global_mfcc_stats(train_items, max_files=800)
        np.savez(NORM_PATH, mean=mean, std=std)
    norm = np.load(NORM_PATH)
    mfcc_mean, mfcc_std = norm["mean"], norm["std"]

    if model is None or dl is None:
        _, test_items = load_fsdd_paths()
        test_ds = FSDDMFCCDataset(test_items, mfcc_mean, mfcc_std, augment=0, train=False)
        dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_stack)
        model = SoundMNISTNet().to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, lengths in dl:
            X = X.to(DEVICE)
            logits, _ = model(X)
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float((y_true == y_pred).mean())
    # macro F1
    f1 = 0.0
    for c in range(10):
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1 += 2*prec*rec/(prec+rec+1e-9)
    f1 = float(f1/10.0)

    if return_cm:
        cm = np.zeros((10,10), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return acc, f1, cm
    return acc, f1

# -------------------- Prototypes (few-shot head) --------------------

def load_model():
    model = SoundMNISTNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def load_norm():
    norm = np.load(NORM_PATH)
    return norm["mean"], norm["std"]

def embedding_from_wave(model, y, sr):
    X = mfcc_64x64(y, sr)   # (1,64,64)
    mean, std = load_norm()
    X[:, :, :] = (X[:, :, :] - mean) / std
    X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)  # (1,1,64,64)
    with torch.no_grad():
        _, z = model(X)
    z = F.normalize(z, dim=-1).squeeze(0).cpu().numpy()  # (128,)
    return z

def load_prototypes():
    if os.path.exists(PROTO_PATH):
        return joblib.load(PROTO_PATH)
    return {"protos": {}, "space": "emb", "created": time.time(), "notes": "1-shot CNN prototypes"}

def save_prototypes(obj): joblib.dump(obj, PROTO_PATH)

def proto_probs(z, protos: Dict[int, np.ndarray], temperature=10.0):
    if not protos: return None
    keys = sorted(protos.keys())
    sims = np.array([float(np.dot(z, protos[k])) for k in keys], dtype=np.float32)
    logits = sims * float(temperature)
    logits -= logits.max()
    p = np.exp(logits); p /= (p.sum() + 1e-12)
    return keys, p

# -------------------- Live mic with early-exit --------------------

@dataclass
class VADConfig:
    silence_ms: int = 500
    min_len_ms: int = 120
    max_len_ms: int = 2000
    thresh_mult: float = 1.5

@dataclass
class EarlyExit:
    enabled: bool = True
    check_every_frames: int = 5
    thresh: float = 0.92
    consec: int = 3

class LiveRecognizer:
    def __init__(self, mode="hybrid", ee=EarlyExit(), vad=VADConfig()):
        if sd is None:
            raise RuntimeError("sounddevice not available. `pip install sounddevice`.")
        self.model = load_model()
        self.protos = load_prototypes().get("protos", {})
        self.mode = mode
        self.ee = ee
        self.vad = vad
        self.q = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        x = indata.mean(axis=1) if indata.ndim>1 else indata[:,0]
        self.q.put(x.copy())

    def start(self, device=None):
        print("[live] Opening microphone stream... (Ctrl+C to exit)")
        self.stream = sd.InputStream(
            channels=1, samplerate=SR, dtype="float32",
            callback=self._callback, blocksize=int(0.02*SR), device=device
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close(); self.stream=None

    def _calibrate_noise(self, seconds=0.6):
        print(f"[calibration] Please be quiet for {int(seconds*1000)} ms...")
        chunks=[]; t_end=time.time()+seconds
        while time.time()<t_end:
            try: chunks.append(self.q.get(timeout=1.0))
            except queue.Empty: pass
        if not chunks: return 1e-3
        noise=np.concatenate(chunks); rms=np.sqrt(np.mean(noise**2))+1e-12
        print(f"[calibration] baseline RMS: {rms:.6f}")
        return rms

    def _collect_utterance(self, noise_rms):
        thresh=noise_rms*self.vad.thresh_mult
        frame_len=int(0.02*SR); hop=frame_len
        voiced=[]; voiced_samples=0; silence_run=0; started=False
        while True:
            try: chunk=self.q.get(timeout=1.0)
            except queue.Empty: continue
            i=0
            while i+frame_len<=len(chunk):
                frame=chunk[i:i+frame_len]; i+=hop
                rms=np.sqrt(np.mean(frame**2))+1e-12
                if rms>thresh:
                    started=True; voiced.append(frame); voiced_samples+=len(frame); silence_run=0
                else:
                    if started:
                        silence_run+=len(frame)
                        if (silence_run/SR)*1000>=self.vad.silence_ms:
                            utt=np.concatenate(voiced) if voiced else np.array([],dtype=np.float32)
                            return utt
            if started and (voiced_samples/SR)*1000>=self.vad.max_len_ms:
                utt=np.concatenate(voiced) if voiced else np.array([],dtype=np.float32)
                return utt

    def _predict(self, y, sr):
        # live-only RMS lift if very quiet
        r = float(np.sqrt(np.mean(y**2)) + 1e-12)
        if r < 0.01:
            y = y * (0.02 / r)
        X = mfcc_64x64(y, sr)              # (1,64,64)
        mean, std = load_norm()
        X[:, :, :] = (X[:, :, :] - mean) / std
        X_t = torch.from_numpy(X).unsqueeze(0).to(DEVICE)  # (1,1,64,64)
        with torch.no_grad():
            logits, z = self.model(X_t)
            p = F.softmax(logits, dim=-1)[0].cpu().numpy()
            z = F.normalize(z, dim=-1)[0].cpu().numpy()
        # prototypes
        keys, pp = None, None
        if self.protos:
            out = proto_probs(z, self.protos, temperature=10.0)
            if out: keys, pp = out
        if self.mode=="base" or pp is None:
            probs = p
        elif self.mode=="proto":
            probs = pp
        else:
            if pp.shape[0]!=10:
                tmp=np.zeros(10, np.float32); tmp[:pp.shape[0]]=pp; pp=tmp
            probs = 0.5*p + 0.5*pp
        pred=int(np.argmax(probs)); conf=float(np.max(probs))
        return pred, conf, probs, X

    def _saliency(self, X1, pred):
        X_t = torch.from_numpy(X1).unsqueeze(0).to(DEVICE)
        X_t.requires_grad_(True)
        logits, _ = self.model(X_t)
        logit = logits[0, pred]
        self.model.zero_grad()
        logit.backward()
        grad = X_t.grad.detach()[0].cpu().numpy()  # (1,64,64)
        band_score = np.mean(np.abs(grad), axis=(0, 2))   # (64,)
        top = np.argsort(-band_score)[:4]
        return [(int(b), float(band_score[b])) for b in top]

    def loop(self, device=None, ee_thresh=0.92, ee_consec=3, check_every_frames=5):
        self.start(device=device)
        noise_rms = self._calibrate_noise(0.6)
        print(f"[live] Mode={self.mode} • Early-exit p>{ee_thresh} for {ee_consec} checks • Speak a digit (0–9).")
        try:
            while True:
                utt = self._collect_utterance(noise_rms)
                if utt.size==0 or (len(utt)/SR)*1000 < self.vad.min_len_ms:
                    print("[live] (too short / empty)"); continue
                frames=int(0.02*SR); hop=frames; consec_ok=0
                for j in range(0, len(utt), hop):
                    cur = utt[:j+frames] if (j+frames)<=len(utt) else utt
                    if (j // hop) % max(1, check_every_frames) != 0:
                        continue
                    if len(cur) < int(0.08*SR):  # skip tiny buffers
                        continue
                    t0=time.time()
                    pred, conf, probs, X1 = self._predict(cur, SR)
                    if conf>=ee_thresh:
                        consec_ok+=1
                        if consec_ok>=ee_consec:
                            latency_ms=(time.time()-t0)*1000.0
                            why = self._saliency(X1, pred)
                            why_s = ", ".join([f"mfcc{b}({s:.1f})" for b,s in why])
                            print(f"[live][early] → {pred}  (dur ~{(len(cur)/SR)*1000:.0f} ms, compute {latency_ms:.1f} ms, conf={conf:.2f})")
                            print(f"        why: {why_s}")
                            break
                    else:
                        consec_ok=0
                else:
                    t0=time.time()
                    pred, conf, probs, X1 = self._predict(utt, SR)
                    latency_ms=(time.time()-t0)*1000.0
                    why = self._saliency(X1, pred)
                    why_s = ", ".join([f"mfcc{b}({s:.1f})" for b,s in why])
                    print(f"[live] → {pred}  (dur ~{(len(utt)/SR)*1000:.0f} ms, compute {latency_ms:.1f} ms, conf={conf:.2f})")
                    print(f"       why: {why_s}")
        except KeyboardInterrupt:
            print("\n[live] Bye!")
        finally:
            self.stop()

# -------------------- Enrollment --------------------

def record_once(prompt: str, seconds: float = 1.0, device=None):
    if sd is None: raise RuntimeError("sounddevice not available.")
    print(prompt); time.sleep(0.2)
    buf = sd.rec(int(seconds*SR), samplerate=SR, channels=1, dtype="float32", device=device)
    sd.wait()
    x = buf[:,0]
    x = librosa.effects.trim(x, top_db=30, frame_length=int(0.025*SR), hop_length=int(0.010*SR))[0]
    return x

def enroll(device=None):
    model = load_model()
    protos = {}
    print("[enroll] We'll record one sample for each digit 0..9. Speak clearly.")
    for d in range(10):
        x = record_once(f"[enroll] Say '{d}' after the beep...", seconds=1.2, device=device)
        z = embedding_from_wave(model, x, SR)
        protos[int(d)] = z
        print(f"[enroll] captured digit {d}")
    obj = {"protos": protos, "space": "emb", "created": time.time(), "notes": "1-shot CNN prototypes", "device": str(device)}
    joblib.dump(obj, PROTO_PATH)
    print(f"[enroll] Saved prototypes to {PROTO_PATH}")

def clear_prototypes():
    if os.path.exists(PROTO_PATH):
        os.remove(PROTO_PATH); print("[enroll] Cleared prototypes.")
    else:
        print("[enroll] No prototypes to clear.")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Spoken-digit recognizer (MFCC 64x64, Adadelta).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train and save the model.")
    ap_train.add_argument("--epochs", type=int, default=50)
    ap_train.add_argument("--batch-size", type=int, default=64)
    ap_train.add_argument("--lr", type=float, default=None, help="Adadelta lr; default None uses optimizer default.")
    ap_train.add_argument("--augment", type=int, default=1, help="Raw waveform augment (noise+shift).")
    ap_train.add_argument("--max-norm-files", type=int, default=None, help="Subset size for MFCC norm computation.")

    sub.add_parser("eval", help="Evaluate best checkpoint.")

    ap_enroll = sub.add_parser("enroll", help="One-shot enroll 0..9 from microphone.")
    ap_enroll.add_argument("--device", default=None)

    sub.add_parser("clear-protos", help="Delete stored prototypes.")

    ap_live = sub.add_parser("live", help="Live mic demo with early-exit.")
    ap_live.add_argument("--mode", default="hybrid", choices=["base","proto","hybrid"])
    ap_live.add_argument("--device", default=None)
    ap_live.add_argument("--ee-thresh", type=float, default=0.92)
    ap_live.add_argument("--ee-consec", type=int, default=3)
    ap_live.add_argument("--ee-every", type=int, default=5)
    ap_live.add_argument("--vad-thresh-mult", type=float, default=1.5)
    ap_live.add_argument("--vad-min-len", type=int, default=120)
    ap_live.add_argument("--vad-silence", type=int, default=500)
    ap_live.add_argument("--vad-max-len", type=int, default=2000)

    args = ap.parse_args()

    if args.cmd == "train":
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, augment=args.augment, max_norm_files=args.max_norm_files)
    elif args.cmd == "eval":
        acc, f1 = evaluate()
        print(f"[eval] accuracy={acc:.4f}, macroF1={f1:.4f}")
    elif args.cmd == "enroll":
        enroll(device=args.device)
    elif args.cmd == "clear-protos":
        clear_prototypes()
    elif args.cmd == "live":
        vad = VADConfig(
            silence_ms=args.vad_silence,
            min_len_ms=args.vad_min_len,
            max_len_ms=args.vad_max_len,
            thresh_mult=args.vad_thresh_mult,
        )
        LiveRecognizer(
            mode=args.mode,
            ee=EarlyExit(check_every_frames=args.ee_every, thresh=args.ee_thresh, consec=args.ee_consec),
            vad=vad
        ).loop(device=args.device, ee_thresh=args.ee_thresh, ee_consec=args.ee_consec, check_every_frames=args.ee_every)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
