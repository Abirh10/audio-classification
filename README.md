# Quran Recitations Audio Classification using PyTorch

An interactive deep learning repository dedicated to classifying audio recordings of Quranic recitations by different reciters. This project utilizes **PyTorch** for neural network architecture and **Librosa** for audio feature processing and extraction[cite: 1].

---

## 🚀 Project Overview

Audio classification bridges signal processing with deep learning. This repository contains an end-to-end pipeline designed to:
*   **Download & Streamline:** Automate dataset ingestion directly from Kaggle[cite: 1].
*   **Process Audio:** Map raw audio file paths and extract spectral features using state-of-the-art audio toolkits (`librosa`)[cite: 1].
*   **Analyze Data:** Provide statistical insights and visual distributions of the training targets[cite: 1].
*   **Deep Learning Ready:** Fully configured to leverage hardware acceleration (**CUDA/GPU**) for training neural networks seamlessly[cite: 1].

---

## 🛠️ Tech Stack & Dependencies

The project is built using Python 3 and relies on the following core ecosystem libraries[cite: 1]:

| Library | Purpose |
| :--- | :--- |
| **PyTorch (`torch`)** | Model building, optimization, and tensor computations[cite: 1]. |
| **Librosa** | Audio loading, sampling, and acoustic feature engineering[cite: 1]. |
| **Pandas & NumPy** | Structured data manipulation, CSV parsing, and matrix operations[cite: 1]. |
| **Scikit-Learn** | Target label encoding (`LabelEncoder`)[cite: 1]. |
| **Scikit-Image** | Audio wave/spectrogram interpolation and resizing[cite: 1]. |
| **Matplotlib** | Data visualization and training metrics plotting[cite: 1]. |

---

## 📊 Dataset Insights

The model trains on the **Quran Recitations for Audio Classification** dataset from Kaggle[cite: 1]. 

### Dataset Snapshot
*   **Total Audio Samples:** 6,687 recordings[cite: 1]
*   **Target Classes:** 12 unique, world-renowned reciters[cite: 1]
*   **Features:** Raw paths mapped to categorical labels (`FilePath`, `Class`)[cite: 1]

### Reciter Class Distribution
The dataset is well-distributed among the following 12 reciters[cite: 1]:

*   Saud Al-Shuraim (696 samples)[cite: 1]
*   Saad Al-Ghamdi (688 samples)[cite: 1]
*   Abdul Rahman Al-Sudais (648 samples)[cite: 1]
*   Yasser Al-Dosari (576 samples)[cite: 1]
*   Ali Al-Huthaify (576 samples)[cite: 1]
*   Bandar Baleelah (576 samples)[cite: 1]
*   Maher Al-Muaiqly (576 samples)[cite: 1]
*   Abdullah Al-Buaijan (504 samples)[cite: 1]
*   Mohammad Ayoub (480 samples)[cite: 1]
*   Nasser Al-Qatami (456 samples)[cite: 1]
*   Abdul Bari Al-Thubaity (456 samples)[cite: 1]
*   Mohammed Al-Luhaidan (455 samples)[cite: 1]

---

## ⚙️ Workflow Breakdown

### 1. Environmental Setup & Acceleration
The notebook automatically handles device placement. If an NVIDIA GPU is detected, the workflow accelerates execution via CUDA; otherwise, it gracefully defaults to CPU execution[cite: 1].

```python
device = "cuda" if torch.torch.cuda.is_available() else "cpu"
```

### 2. Automated Dataset Retrieval
Using `opendatasets`, the pipeline pulls down the 2.12 GB dataset smoothly with your Kaggle API token[cite: 1].

```bash
pip install opendatasets --quiet
```

### 3. Path Engineering & Sanitization
Because local directory paths can vary from metadata configurations, the code cleanses and formats the directory mappings dynamically[cite: 1]:

```python
data_df['FilePath'] = '/content/quran-recitations-for-audio-classification/Dataset/' + data_df['FilePath'].str[1:]
```

---

## 🏃‍♂️ How to Run This Repository

1. **Clone the Repository:**
```bash
   git clone [https://github.com/your-username/quran-recitation-classification.git](https://github.com/your-username/quran-recitation-classification.git)
   cd quran-recitation-classification
   ```

2. **Install Required Packages:**
```bash
   pip install torch librosa pandas numpy scikit-learn matplotlib scikit-image opendatasets
   ```

3. **Configure Kaggle Credentials:**
   Ensure you have your `kaggle.json` API key ready. When prompted by the notebook runtime, provide your Kaggle username and token to initialize the automated download[cite: 1].

4. **Execute the Notebook:**
   Open the `.ipynb` file in Google Colab, Jupyter Lab, or VS Code, and execute the cells sequentially to visualize the data and begin pipeline processing[cite: 1].

---

## 📈 Next Steps
* [ ] Implement Mel-Frequency Cepstral Coefficients (MFCCs) feature extraction using `librosa`[cite: 1].
* [ ] Construct a Custom PyTorch `Dataset` and `DataLoader` for batching[cite: 1].
* [ ] Design and train a Convolutional Neural Network (CNN) optimized for audio spectrogram patterns.

---

**Author:** Abir Hirani  
**Project Context:** Software Development / Machine Learning Audio Analysis Pipeline.
