## Heart Diseases Classification with Deep, Hybrid, and Ensemble Learning

Early and accurate detection of cardiovascular conditions can save lives. This repository contains an end-to-end study and reproducible pipeline for classifying heart sounds using Machine Learning (ML), Deep Learning (DL), Hybrid, and Ensemble approaches. The work compares baseline models against hybrid CNN–RNN architectures and model-agnostic ensembles.

### Key Highlights
- **Problem**: Heart sound (phonocardiogram) classification into normal and pathological categories.
- **Data**: Kaggle Heartbeat Sounds Set_A (176 samples; class imbalance addressed). See Dataset section.
- **Approach**: ML baselines (LR, SVM, RF, KNN, GBM, XGBoost, LightGBM, DT, NB), DL baselines (CNN, LSTM, Bi-LSTM, GRU), Hybrids (CNN-LSTM, CNN-BiLSTM, CNN-GRU), and Ensembles (soft/hard voting across DL models).
- **Result (reported)**:
  - Best single family: Recurrent (Bi-LSTM/GRU) for sequential features.
  - Best hybrid: **CNN-LSTM** (≈92% accuracy at 200 epochs).
  - Best overall: **Soft-Voting Ensemble** (CNN + LSTM + Bi-LSTM + GRU) at ≈94% accuracy.

### Dataset
- Heartbeat Sounds (Set_A) from Kaggle: [Kinguistics – Heartbeat Sounds](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)
- Composition (Set_A): NaN 52, Artifact 40, Murmur 34, Normal 31, Extrahls/Extrasystole 19.
- Preprocessing used in the study:
  - Resample audio to 22,050 Hz.
  - Compute 128-band Mel-spectrograms with 128 time frames (Librosa).
  - ML models: vectorize 2D spectrogram to 1D features; standardize (zero mean, unit variance).
  - DL models: data augmentation (±10% time-stretch, ±1 semitone pitch shift, Gaussian noise SNR 20–30 dB).

### Methods
- **Classical ML**: Logistic Regression, SVM, Random Forest, KNN, Gradient Boosting, XGBoost, LightGBM, Decision Tree, Naive Bayes.
- **Deep Learning**: CNN (spatial features), LSTM/Bi-LSTM/GRU (temporal features).
- **Hybrid**: CNN-LSTM, CNN-BiLSTM, CNN-GRU — CNN extracts high-level feature maps; RNN models sequence dependencies.
- **Ensemble**: Model-agnostic soft- and hard-voting over CNN, LSTM, Bi-LSTM, GRU.

### Training & Evaluation
- Optimizer: Adam (lr=1e-3), batch size: 16, epochs: up to 250, early stopping (patience=15).
- Split: stratified hold-out (80/20).
- Metrics: accuracy, precision, recall, F1; confusion matrices and per-class metrics.

### Results (Summary)
- Baselines:
  - CNN peaked ≈82% (200 epochs), overfit at 250 (≈61%).
  - LSTM unstable (≈55–75%).
  - Bi-LSTM strong and stable (≈88–91%).
  - GRU competitive (up to ≈90%).
- Hybrids:
  - CNN-LSTM best hybrid (≈92% acc, P/R/F1 ≈92%).
  - CNN-BiLSTM fluctuated; sensitive to hyperparameters.
  - CNN-GRU moderate.
- Ensembles:
  - Soft Voting: ≈94% across metrics.
  - Hard Voting: ≈90% across metrics.

Notes: Small dataset and single hold-out split may inflate metrics. Use cross-validation and external validation for deployment-grade claims.

### Quickstart (Reference Scaffold)
Below is a lightweight scaffold to reproduce the data pipeline and train example models. Adapt to your environment.

```bash
# 1) Create environment
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy scikit-learn librosa matplotlib pandas jupyter torch torchvision torchaudio xgboost lightgbm

# 2) Prepare data
# Download Set_A from Kaggle into data/set_a

# 3) Run notebooks (EDA/training)
jupyter notebook notebooks/

# 4) Or run scripts (if using src/)
python src/train_cnn_lstm.py --data_dir data/set_a --epochs 200 --batch_size 16
python src/train_ensemble.py --models cnn lstm bilstm gru --voting soft
```

### Minimal Code Snippet (Feature Extraction)
```python
import librosa
import numpy as np

def load_mel_spectrogram(path, sr=22050, n_mels=128, frames=128):
    y, _ = librosa.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    # center/pad or crop to fixed time frames
    if S_db.shape[1] < frames:
        pad = frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad)))
    else:
        S_db = S_db[:, :frames]
    return S_db.astype(np.float32)
```

### How to Cite
If you use this repository or the accompanying paper, please cite:

> Heart Diseases Classification Through Deep Learning, Hybrid Learning and Ensemble Learning Techniques, 2024. See `Conf_paper.pdf` for full details.

Relevant references (from paper):
- Ali et al., 2024. AIP Conf. Proc. 3232(1):020022.
- Zabil et al., 2024. Electrical Engineering Technical Journal 1(1):4514–4523.
- Chen et al., 2021. Remote Sensing 13:4712.
- Student & Raju, 2022. IJRASET 10:961–964.
- Sarker, 2021. SN Computer Science 2:420.

### Limitations & Future Work
- Small dataset; limited external validation. Future work: standardized benchmarks (e.g., PhysioNet/CinC 2016), rigorous CV, and external hold-outs.
- Improve hybrid stability with better tuning/regularization; investigate weighted voting and stacking ensembles.
- Add explainability (e.g., SHAP/LIME) and explore real-time/edge deployment.

### Contributors
- Rakibul Hassan — Ensemble Learning (lead), integration of CNN/LSTM/Bi-LSTM/GRU voting.
- Md. Aminul Haq Emon — Deep Learning (CNN, LSTM, BiLSTM, GRU)
- Istiaque Uddin Hyder — Hybrid modeling (CNN-LSTM, CNN-GRU, CNN-BiLSTM).
- Md Faysal Hossen & Nowaz Bin Yonus — Classical ML baselines and data preprocessing.

If we missed anyone, please open a PR to update this section.

### Acknowledgments
- PhysioNet/CinC 2016 challenge organizers and the open-source communities behind Librosa, PyTorch, scikit-learn, XGBoost, and LightGBM.
- Kaggle and dataset contributors for making heart sound data available to the community.



# Heart-Diseases-Classification-Through-Deep-Learning-Hybrid-Learning-and-Ensemble-Learning-Techniques
