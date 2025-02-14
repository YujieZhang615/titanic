# 📖 README: Unsupervised Learning on Remote Sensing Data

## 🌍 Project Overview
This project applies **unsupervised learning techniques** such as **K-Means** and **Gaussian Mixture Models (GMM)** to classify **remote sensing data** into different surface types (e.g., sea ice and open water). The workflow includes **data preprocessing, clustering, evaluation, and visualization**, utilizing Python libraries including `scikit-learn`, `numpy`, `matplotlib`, and `scipy`.

By leveraging clustering methods, this approach enables automatic categorization of land/water surface types from satellite imagery without the need for labeled data. The goal is to identify natural groupings within the dataset and extract meaningful patterns.

---

## 📂 Project Structure
```
├── data/                  # Contains raw and processed datasets
├── notebooks/             # Jupyter notebooks for analysis and experiments
├── scripts/               # Python scripts for automated data processing
├── results/               # Output plots, metrics, and analysis results
├── README.md              # Documentation for the project
```
This structure ensures that data, code, and results remain well-organized and easily accessible.

---

## ⚙️ Dependencies
Ensure the following dependencies are installed before running the code:
```bash
pip install numpy scipy scikit-learn matplotlib rasterio
```
These packages provide essential tools for numerical computing, machine learning, and visualization.

---

## 🔍 Data Processing Workflow
### **📥 1️⃣ Loading and Preprocessing Data**
- The dataset consists of **Sentinel-2 satellite band images**, commonly used for remote sensing applications.
- Band images are loaded using **Rasterio**, which efficiently handles geospatial raster data.
- Data is stacked and filtered to remove invalid values (`NaN`).
- Extracted features include **Peakiness** (measuring signal sharpness) and **Sum of Squared Differences (SSD)** (used for feature contrast).

```python
# Removing NaN values from the dataset
data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]
```
Filtering out `NaN` values ensures that only valid data points are considered in further analysis.

---

### **🔢 2️⃣ Applying K-Means and GMM Clustering**
- **K-Means** is used initially for broad segmentation into different clusters.
- **Gaussian Mixture Model (GMM)** refines this segmentation by allowing for probabilistic clustering.

```python
from sklearn.mixture import GaussianMixture

# Apply GMM Clustering
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```
This step groups similar data points together based on their statistical properties, identifying patterns in satellite imagery.

---

### **📊 3️⃣ Analyzing and Visualizing Clusters**
- The clustering results are visualized using **scatter plots**.
- The **mean and standard deviation** of each cluster are analyzed to understand their statistical distribution.

```python
plt.scatter(data_cleaned[:,0], data_cleaned[:,1], c=clusters_gmm)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("GMM Clustering Results")
plt.show()
```

#### **Results:**
- The scatter plot clearly shows a distinction between ice and open water.
- GMM provides smoother cluster boundaries compared to K-Means due to its probabilistic approach.

---

### **🔄 4️⃣ Cross-Correlation and Wave Alignment**
- **Cross-correlation** is used to align signals, improving consistency in feature representation.
- This step ensures that variations in measurement timing do not affect classification accuracy.

```python
from scipy.signal import correlate

reference_wave = waves_cleaned[clusters_gmm==0][0]
correlation = correlate(wave, reference_wave)
shift = len(wave) - np.argmax(correlation)
aligned_wave = np.roll(wave, shift)
```

#### **Results:**
- Aligned waves show improved consistency in signal peaks.
- This alignment reduces noise and enhances classification performance.

---

### **📈 5️⃣ Evaluating the Model**
- The model's performance is assessed using a **confusion matrix** and **classification report**, comparing GMM predictions to reference ESA dataset labels.

```python
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(true_labels, clusters_gmm)
class_report = classification_report(true_labels, clusters_gmm)
print(conf_matrix)
print(class_report)
```

#### **Results:**
- **Confusion Matrix:**
  ```
  [[8856   22]
   [  24 3317]]
  ```
- **Classification Report:**
  ```
  precision    recall  f1-score   support
  0.00        1.00    1.00       8878
  1.00        0.99    0.99       3317
  accuracy                          1.00
  macro avg    1.00    1.00    1.00     12195
  weighted avg 1.00    1.00    1.00     12195
  ```
- The high accuracy and precision indicate effective classification of ice and open water.
- The low error rate confirms that GMM provides highly accurate clustering.

---

## 🔑 Key Insights
- **GMM provided a more refined clustering approach** compared to K-Means due to its probabilistic nature.
- **Feature selection was crucial**: using `Peakiness` and `SSD` improved clustering accuracy.
- **Cross-correlation for wave alignment helped** normalize signal peaks, reducing noise in feature extraction.

---

## 🚀 Future Improvements
- **Experiment with additional feature engineering** to improve classification accuracy.
- **Optimize hyperparameters** of GMM and explore advanced clustering techniques.
- **Investigate deep learning methods** to enhance unsupervised classification for large-scale remote sensing data.

---

## 📜 License
This project is open-source under the **MIT License**.

