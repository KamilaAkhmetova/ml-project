# 🔭 MAGIC Gamma Telescope — Particle Classification

> Automatically classify high-energy particles captured by the MAGIC telescope as **gamma rays (signal)** or **hadrons (noise)** using machine learning.

---

## 📌 Problem Statement

The MAGIC telescope (La Palma, Canary Islands) records tens of thousands of atmospheric shower images every night. The overwhelming majority are **hadrons** — cosmic background noise. Gamma rays, by contrast, carry information about specific astrophysical sources: supernovae, black holes, and pulsars.

Without automatic classification, the telescope is essentially blind — it records everything but can identify nothing. Our ML pipeline solves this.

---

## 🗂 Project Structure

```
ml-project/
├── data/
│   ├── magic04.data          # Raw dataset (UCI)
│   └── processed/            # Train/test splits (generated)
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb     # Model training & evaluation
├── src/
│   └── preprocessing.py      # Reusable preprocessing pipeline
├── models/                   # Saved model files (.pkl)
└── README.md
```

---

## 📊 Dataset

**MAGIC Gamma Telescope Dataset** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

- **19,020 samples** — 12,332 gamma (65%), 6,688 hadron (35%)
- **10 features** — geometric descriptors of Cherenkov light ellipses (Hillas parameters)
- **Binary target** — `g` = gamma signal, `h` = hadron background

| Feature | Description |
|---------|-------------|
| fLength | Major axis of ellipse [mm] |
| fWidth | Minor axis of ellipse [mm] |
| fSize | log10(sum of pixel contents) |
| fConc | Ratio: 2 highest pixels / fSize |
| fConc1 | Ratio: highest pixel / fSize |
| fAsym | Distance from highest pixel to center [mm] |
| fM3Long | 3rd root of 3rd moment along major axis [mm] |
| fM3Trans | 3rd root of 3rd moment along minor axis [mm] |
| fAlpha | Angle of major axis with vector to origin [deg] |
| fDist | Distance from origin to ellipse center [mm] |

---

## 🤖 Models & Results

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|----------|----|-----------|--------|---------|
| **XGBoost** 🏆 | **0.8862** | **0.9151** | 0.8858 | **0.9465** | **0.9387** |
| Random Forest | 0.8859 | 0.9148 | 0.8869 | 0.9444 | 0.9374 |
| MLP Neural Net | 0.8801 | 0.9118 | 0.8714 | 0.9562 | 0.9328 |
| Logistic Regression | 0.7836 | 0.8439 | 0.7929 | 0.9019 | 0.8313 |

**Best model: XGBoost** with ROC-AUC = 0.9387

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/KamilaAkhmetova/ml-project.git
cd ml-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing.ipynb
notebooks/03_Modeling.ipynb
```

---

## 🔑 Key Findings

- **fAlpha** is the most discriminative feature — gamma rays from a specific source always have small fAlpha values, while hadrons are distributed randomly across 0–90°
- **Tree-based models** (XGBoost, Random Forest) significantly outperform linear baseline
- All features are statistically significant (Mann-Whitney U test, p < 0.05)
- fConc and fConc1 are highly correlated — potential for dimensionality reduction


