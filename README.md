
# SSVEPNet EEG Classification Project Guide

## 1. Project Setup
1. **Clone Repository**  
   ```bash
   git clone https://github.com/your-username/SSVEPNet.git
   cd SSVEPNet
   
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Data Collection

* **Load MATLAB Files**

  ```python
  from scipy.io import loadmat

  data = {}
  for subj in range(1, 11):
      mat = loadmat(f"data/Sub_{subj}.mat")
      data[f"Sub_{subj}"] = {
          "eeg": mat["EEG_signals"],    # shape: (channels, samples, trials)
          "labels": mat["labels"].ravel()
      }
  ```
* **Inspect & Organize**

  * Confirm shape and sampling rate.
  * Store each subject’s EEG and labels in a unified DataFrame.

---

## 3. Preprocessing

1. **Filtering**

   * Band-pass (e.g. 5–50 Hz) to isolate SSVEP frequencies.
   * Notch at 50 Hz to remove power-line noise.

   ```python
   from mne.filter import filter_data

   filtered = filter_data(raw_data, sfreq=250, l_freq=5, h_freq=50)
   ```
2. **Power Spectral Density**

   ```python
   from scipy.signal import welch

   freqs, psd = welch(filtered, fs=250, nperseg=256)
   ```
3. **Consolidate**

   * Build a pandas DataFrame with columns:
     `Subject, Trial, Channel, Frequency, Power, Label`

![image](https://github.com/user-attachments/assets/f9901497-d29a-46f3-982e-5acfaaa5265f)

---

## 4. Feature Transformation & Normalization

1. **One-Hot Encoding (Channel)**

   ```python
   df = pd.get_dummies(df, columns=["Channel"])
   ```
2. **Standard Scaling (Power)**

   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   df[power_cols] = scaler.fit_transform(df[power_cols])
   ```

---

## 5. Classification & Evaluation

1. **Canonical Correlation Analysis (CCA)**

   ```python
   from mne.decoding import CCA

   cca = CCA(n_components=1)
   X_cca = cca.fit_transform(X, y)
   ```
2. **K-Nearest Neighbors (KNN)**

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.pipeline import make_pipeline

   model = make_pipeline(CCA(n_components=1), KNeighborsClassifier(n_neighbors=3))
   ```
3. **5-Fold Cross-Validation**

   ```python
   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
   print("Mean accuracy:", scores.mean())
   ```

---

## 6. Results & Interpretation

* Report per‐fold accuracies and overall mean.
* Analyze confusion matrix (optional).

<img width="557" alt="image" src="https://github.com/user-attachments/assets/60e2d4c2-cfca-4234-b489-0c68b8a2f71c" />

---

## 7. Next Steps

* Experiment with different filter bands or CCA components.
* Compare KNN with other classifiers (SVM, Random Forest).
* Implement real-time SSVEP detection pipeline.

```
```
