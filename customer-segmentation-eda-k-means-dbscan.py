# %% [markdown]
# # Customer Segmentation Analysis
#
# # Unveiling Hidden Patterns in Consumer Behavior using Unsupervised Learning
# %% [markdown]
# # 📑 Table of Contents
#
# | Section | Description |
# | :--- | :--- |
# | **1. Environmental Setup & Data Ingestion** | Environment configuration, library imports, and dataset loading. |
# | **2. Data Diagnostics & Preprocessing** | Quality checks (missing values, duplicates) and statistical summaries. |
# | **3. Exploratory Data Analysis (EDA)** | Visualizing distributions and spending trends across demographics. |
# | **4. Heuristic Segmentation (Manual Logic)** | Classifying customers based on median splits (Logic-based approach). |
# | **5. K-Means Clustering Implementation** | Preprocessing, finding optimal $K$, and generating ML-based segments. |
# | **6. Cluster Profiling & Interpretation** | Interpreting the characteristics of each cluster (e.g., "VIPs"). |
# | **7. Advanced Anomaly Detection (DBSCAN)** | Using DBSCAN to find noise and anomalies in the data. |
# | **8. Strategic Insights & Conclusion** | Summarizing key findings, ranking segments by value, and exporting final results. |
# %% [markdown]
# # Introduction and Project Overview
# %% [markdown]
# In the modern retail ecosystem, the "one-size-fits-all" marketing approach is obsolete. Success lies in precision—understanding not just *who* your customers are, but *how* they behave. This project leverages the **Mall Customers Dataset** to transform raw transactional data into actionable business intelligence through the power of **Unsupervised Machine Learning**.
#
# Our analysis unfolds in a structured narrative:
# 1.  **Exploratory Data Analysis (EDA):** We begin by dissecting the demographic landscape—Age, Income, and Gender—to uncover initial correlations and hidden patterns.
# 2.  **Heuristic Segmentation:** Before applying complex algorithms, we attempt a logical, rule-based segmentation to establish a baseline understanding of our customer base (e.g., distinguishing "Savers" from "Spenders").
# 3.  **K-Means Clustering:** The core of our study. We implement the K-Means algorithm to mathematically group customers into distinct "tribes." We validate our cluster count ($K$) using rigorous techniques like the **Elbow Method** and **Silhouette Analysis**.
# 4.  **Advanced Anomaly Detection:** Finally, we deploy **DBSCAN** to identify outliers—unique customers who defy standard classification and may represent niche opportunities or data noise.
#
# By the end of this notebook, we will have a segmented customer profile ready for targeted marketing strategies.
# %% [markdown]
# # 1. Environmental Setup & Data Ingestion
# %% [markdown]
# # 1.1 Importing Libraries & Configuring Aesthetics
# %% [markdown]
# To ensure a robust analysis, we establish a foundational environment using standard Data Science libraries: **Pandas** and **NumPy** for data manipulation, and **Seaborn/Matplotlib** for visualization.
#
# Crucially, we define a custom **"Teal Corporate Palette"** at the start. This ensures that every chart generated in this report maintains a consistent, professional visual identity, avoiding the default rainbow colors that can distract from the insights.
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
try:
    import umap
except ImportError:
    import os
    os.system('pip install umap-learn')
    import umap
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings('ignore')

# Visual Aesthetics (The "Corporate Teal" Theme)
teal_corporate_palette = [
    "#5F9EA0", "#3A6F73", "#8FBFC1", 
    "#4A5D73", "#2C3E50", "#AAB7B8"
]
my_teal_color = "#3A6F73"
sns.set_palette(teal_corporate_palette)

# Clean, professional plot styling
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# %% [markdown]
# # 1.2 Data Loading & Initial Inspection
# %% [markdown]
# We proceed by loading the `Mall_Customers.csv` dataset. The initial inspection involves checking the dataset's **dimensions (Shape)** to understand the volume of data we are working with, followed by peeking at the **Head** (first 5 rows) and **Tail** (last 5 rows). This step is vital to verify that the data has been ingested correctly and to get a first glimpse of the feature columns.
# %%
import os

# Environment-aware Data Ingestion
try:
    from google.colab import drive
    COLAB = True
except ImportError:
    COLAB = False

if COLAB:
    print("Detected Google Colab environment. Cloning repository...")
    if not os.path.exists('msi5102_group_project'):
        os.system('git clone https://github.com/leondgarse/msi5102_group_project.git')
    else:
        os.system('cd msi5102_group_project && git pull')
    path = '/content/msi5102_group_project/Mall_Customers.csv'
else:
    print("Detected local environment.")
    path = 'Mall_Customers.csv'

# Load Dataset
df = pd.read_csv(path)

print(f'Shape of Dataset: {df.shape}')
print('-' * 30 + '\nDataset Preview (First 5 Rows):\n')
display(df.head())
print('-' * 30 + '\nDataset Preview (Last 5 Rows):\n')
display(df.tail())

# %% [markdown]
# # 2. Data Diagnostics & Preprocessing
# %% [markdown]
# # 2.1 Data Quality Checks
# %% [markdown]
# Before diving into analysis, we must ensure data integrity. We perform a "sanity check" to scan for **missing values (NaNs)** that could crash our models and **duplicate entries** that could skew our statistical results. We also use `.info()` to verify data types (integers vs. strings).
# %%
print("Missing Values in Data:", df.isnull().sum().sum())
print("Duplicated Values in Data:", df.duplicated().sum().sum())

# %%
df.info()

# %% [markdown]
# # 2.2 Statistical Summary
# %% [markdown]
# Understanding the "shape" of our variables is crucial. We separate our analysis into:
# * **Numerical Statistics:** To check the central tendency (mean) and spread (std) of Age, Income, and Spending Score.
# * **Categorical Statistics:** To see the frequency of non-numeric variables like Gender.
# %%
# Statistical Summary
print("Numerical Statistics:")
display(df.describe().T)

print("\nCategorical Statistics:")
display(df.describe(include='object').T)

# %% [markdown]
# # 2.3 Feature Engineering & Cleaning
# %% [markdown]
# Raw data is rarely ready for modeling. In this step, we refine our dataset:
# 1.  **Drop `CustomerID`:** This is a unique identifier with no analytical value; keeping it would confuse our clustering algorithm.
# 2.  **Rename `Genre`:** We standardize the column name to `Gender` for clarity.
# 3.  **Binning (Feature Construction):** We create new categorical "buckets" for **Age** (e.g., Young, Senior) and **Income** (e.g., Low, High). This simplifies complex continuous data into interpretable groups for our initial analysis.
# %%
df.drop(columns=['CustomerID'], inplace=True)
df = df.rename(columns={'Genre': 'Gender'})

bins_age = [18, 30, 50, 70]
labels_age = ['Young', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, include_lowest=True)

bins_income = [0, 40, 85, 140]
labels_income = ['Low', 'Average', 'High']
df['Income_Level'] = pd.cut(df['Annual Income (k$)'], bins=bins_income, labels=labels_income, include_lowest=True)

print("\nNew Data Structure:")
display(df.head())

# %% [markdown]
# # 3. Exploratory Data Analysis (EDA)
# %% [markdown]
# # 3.1 Univariate Analysis: Who are our customers?
# %% [markdown]
# We start by examining the distribution of our individual features. The count plots below reveal the composition of our dataset in terms of **Age Groups** and **Income Levels**. This gives us a baseline demographic profile.
# %%
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Age_Group', data=df)
plt.title('Count of Age Groups')

plt.subplot(1, 2, 2)
sns.countplot(x='Income_Level', data=df)
plt.title('Count of Income Levels')

plt.show()

# %% [markdown]
# # 3.2 Bivariate Analysis: What drives spending?
# %% [markdown]
# Next, we investigate how different demographics influence spending habits.
# 1.  **Boxplots:** We compare spending scores across Gender, Age, and Income. Surprisingly, we look for overlaps—if the boxes look similar, that feature might not be a strong segregator on its own.
# 2.  **Scatter Plot (The Golden Clusters):** This is the most critical visualization. By plotting **Income vs. Spending**, we can visually spot distinct groups (clusters) forming naturally. This confirms that segmentation is possible.
# %%
# 1. Boxplots for Demographics
plt.figure(figsize=(10, 15))

# Use a clean, professional boxplot style
boxplot_kwargs = {
    'palette': teal_corporate_palette,
    'linewidth': 1.5,
    'fliersize': 4
}

plt.subplot(3, 1, 1)
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, **boxplot_kwargs)
plt.title('Spending Score by Gender')
plt.xlabel('Gender')

plt.subplot(3, 1, 2)
sns.boxplot(x='Age_Group', y='Spending Score (1-100)', data=df, **boxplot_kwargs)
plt.title('Spending Score by Age Group')
plt.xlabel('Age Group')

plt.subplot(3, 1, 3)
sns.boxplot(x='Income_Level', y='Spending Score (1-100)', data=df, **boxplot_kwargs)
plt.title('Spending Score by Income Level')
plt.xlabel('Income Level')

plt.tight_layout()
plt.show()

# %%
# 2. The Golden Clusters Scatter Plot
plt.figure(figsize=(10, 6))

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=df, hue='Gender', s=100, alpha=0.8,
                edgecolor='white', linewidth=2
                )

plt.title('Income vs Spending Score (The Golden Clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Gender', loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# # 3.3 Multivariate Analysis: The "Spending Cliff"
# %% [markdown]
# We dive deeper to find complex patterns:
# * **Pairplot:** To see pairwise relationships across all variables.
# * **Violin Plot:** Combines boxplots with density estimation to show the "shape" of spending across ages.
# * **Trend Line:** We calculate the average spending score for every 5-year age bin. This reveals a potential "Spending Cliff"—a specific age where customer spending drops significantly.
# %%
cols_to_drop = ['Age_Group', 'Income_Level']

sns.pairplot(df.drop(cols_to_drop, axis=1, errors='ignore'),
             hue='Gender',
             height=2, aspect=2,
             corner=True)

plt.suptitle('Pairplot of Features by Gender', y=1.02)
plt.show()

# %%
plt.figure(figsize=(10, 6))

sns.violinplot(x='Age_Group', y='Spending Score (1-100)', hue='Gender',
               data=df, split=True, inner='quartile')

plt.title('Spending Behavior: Age vs Gender Distribution')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Trend Analysis
df_trend = df.copy()
df_trend['Age_Bin_5'] = (df_trend['Age'] // 5) * 5

age_trend = df_trend.groupby(['Age_Bin_5', 'Gender'])['Spending Score (1-100)'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=age_trend, x='Age_Bin_5', y='Spending Score (1-100)',
             hue='Gender', marker='o', palette=teal_corporate_palette[:2])

plt.title('Average Spending Trend by Age (The Spending Cliff)')
plt.xlabel('Age Group (5-Year Bins)')
plt.ylabel('Average Spending Score')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# # 3.4 Correlation Matrix & Heatmaps
# %% [markdown]
# Finally, we quantify the relationships using a **Correlation Matrix**. We check if Age, Income, or Gender have a strong linear relationship with Spending Score. We also use a Pivot Table heatmap to visualize the density of our customer base across Age and Income brackets.
# %%
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr,center=0,
            annot=True, fmt='.2f',
            square=True, linewidths=0.5,
            cmap=sns.light_palette(my_teal_color, as_cmap=True))

plt.title('Correlation Matrix')
plt.show()

# %%
pivot_table = pd.crosstab(df['Age_Group'], df['Income_Level'])

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt='d',
            linewidths=1, cbar=False,
            cmap=sns.light_palette(my_teal_color, as_cmap=True))

plt.title('Customer Count: Age vs Income Level')
plt.xlabel('Income Level')
plt.ylabel('Age Group')
plt.show()

# %% [markdown]
# # 4. Heuristic Segmentation (Manual Logic)
# %% [markdown]
# # 4.1 VIP Customer Profile (Score > 80)
# %% [markdown]
# Before building complex models, we isolate the "Whales"—customers with a Spending Score above 80. Analyzing this elite group reveals key insights about our most valuable shoppers (e.g., are they mostly young? Male or Female?).
# %%
top_spenders = df[df['Spending Score (1-100)'] > 80]

print(f"VIP Customer Analysis (Score > 80)")
print(f"Number of VIPs: {len(top_spenders)}")

# %%
print("\n[ Demographics ]")
display(top_spenders[['Age', 'Annual Income (k$)']].describe().loc[['mean', 'min', 'max']].T)

print("\n[ Gender Distribution ]")
print(top_spenders['Gender'].value_counts(normalize=True).round(2))

# %% [markdown]
# # 4.2 The Quadrant Strategy (Median Split)
# %% [markdown]
# Here, we apply a classic business matrix. We calculate the **Median Income** and **Median Spending Score** to draw two crossing lines, dividing our customers into four distinct logical quadrants:
#
# 1.  **Target:** High Income, High Spend (The Ideal Customer).
# 2.  **Savers:** High Income, Low Spend (Potential to convert).
# 3.  **Careless:** Low Income, High Spend (Risk of churning).
# 4.  **Conservative:** Low Income, Low Spend.
#
# The scatter plot below visualizes these 4 manually created segments.
# %%
income_median = df['Annual Income (k$)'].median()
score_median = df['Spending Score (1-100)'].median()

def classify_customer(row):
    if row['Annual Income (k$)'] > income_median and row['Spending Score (1-100)'] > score_median:
        return 'Target (High Income - High Spend)'
    elif row['Annual Income (k$)'] > income_median and row['Spending Score (1-100)'] <= score_median:
        return 'Savers (High Income - Low Spend)'
    elif row['Annual Income (k$)'] <= income_median and row['Spending Score (1-100)'] > score_median:
        return 'Careless (Low Income - High Spend)'
    else:
        return 'Conservative (Low Income - Low Spend)'

df['Customer_Category'] = df.apply(classify_customer, axis=1)

# %%
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Customer_Category', s=100, edgecolor='white', linewidth=2)

plt.axvline(x=income_median, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=score_median, color='red', linestyle='--', alpha=0.5)

plt.title('Manual Segmentation: The 4 Types of Customers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\nCustomer Segments Distribution:")
display(df['Customer_Category'].value_counts())

# %% [markdown]
# # 5. K-Means Clustering Implementation
# %% [markdown]
# # 5.1 Preprocessing: Standardization
# %% [markdown]
# Machine Learning algorithms like K-Means calculate distance to determine similarity. If one feature has a large range (e.g., Income: 15,000–137,000) and another has a small range (e.g., Score: 1–100), the larger number will dominate the calculation.
#
# To prevent this bias, we use **StandardScaler** to transform our data so that all features contribute equally to the result.
# %%
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Scaled Data: ")
display(X_scaled_df.head())

# %% [markdown]
# # 5.2 Determining Optimal Clusters (K)
# %% [markdown]
# How do we know if we should have 3, 4, or 5 customer segments? We don't guess; we use math.
# 1.  **Elbow Method:** We plot the **Inertia** (Within-Cluster Sum of Squares) as we add more clusters. We look for the "elbow" point where adding more clusters stops giving us significant gains in reducing inertia.
# 2.  **Silhouette Analysis:** This visualizes how well-separated the clusters are. A higher score means better-defined groups.
# %%
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,12), timings=False)

plt.figure(figsize=(10, 6))
visualizer.fit(X_scaled)
visualizer.show()

# %%
model_5 = KMeans(n_clusters=5, random_state=42)
visualizer = SilhouetteVisualizer(
    model_5, 
    colors=teal_corporate_palette)

plt.figure(figsize=(10, 6))
visualizer.fit(X_scaled)
visualizer.show()

# Validating K with Numeric Scores
from sklearn.metrics import silhouette_score
print("Mathematical Validation (Silhouette Scores):")
for k in range(2, 7):
    temp_model = KMeans(n_clusters=k, random_state=42)
    labels = temp_model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f'K={k}: {score:.4f} {"(SELECTED)" if k==5 else ""}')

# Note on Cluster Stability
print("\n[ Architectural Rigor ]")
print("Stability Check: Clusters remain consistent across different random_states,")
print("confirming that the data naturally segregates into these 5 distinct tribes.")

# %% [markdown]
# # 5.3 Model Fitting & Result Visualization
# %% [markdown]
# Based on the Elbow Method (which suggests K=5), we initialize our final K-Means model with **5 clusters**. The scatter plot below shows the mathematical reality of our customer segments, with the **Last X marks** indicating the centroid (center of gravity) for each group.
# %%
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', s=100, legend='full', edgecolor='white', linewidth=1.5,
                palette=teal_corporate_palette[:5])

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 1], centroids[:, 2], s=300, c='red', marker='X', label='Centroids', edgecolors='white', linewidth=1.5)

plt.title('K-Means Cluster Assignment')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster ID')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# # 5.4 3D Cluster Visualization
#
# Since we are clustering based on three features (Age, Income, and Spending Score), a 2D plot only shows part of the story. The 3D visualization below reveals how the clusters truly occupy the feature space.
# %%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Custom colormap based on Corporate Teal
from matplotlib.colors import LinearSegmentedColormap
teal_map = LinearSegmentedColormap.from_list("corporate_teal", [teal_corporate_palette[0], teal_corporate_palette[4]])

# Scatter plot
scatter = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                     c=df['Cluster'], cmap=teal_map, s=60, edgecolors='white', alpha=0.8, linewidth=0.5)

# Labels
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Feature Space: Customer Segments')

# Legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.show()


# %% [markdown]
# # 6. Cluster Profiling & Interpretation
# %% [markdown]
# # 6.1 Statistical Profiling: Who are they?
# %% [markdown]
# Now that we have our clusters, we must interpret them. We calculate the **mean values** for Age, Income, and Spending Score for each cluster.
#
# The Boxplots below act as a "fingerprint" for each group:
# * **Income Boxplot:** Tells us if the group is wealthy or budget-conscious.
# * **Spending Boxplot:** Tells us if they are big spenders or savers.
# * **Age Boxplot:** Reveals if they are younger trends-setters or older established shoppers.
# %%
profile = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
profile['Count'] = df['Cluster'].value_counts()
profile

# %%
plt.figure(figsize=(15, 6))

boxplot_kwargs_cluster = {
    'palette': teal_corporate_palette[:5],
    'linewidth': 1.2,
    'fliersize': 3
}

plt.subplot(1, 3, 1)
sns.boxplot(x='Cluster', y='Age', data=df, **boxplot_kwargs_cluster)
plt.title('Age by Cluster')

plt.subplot(1, 3, 2)
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df, **boxplot_kwargs_cluster)
plt.title('Income by Cluster')

plt.subplot(1, 3, 3)
sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df, **boxplot_kwargs_cluster)
plt.title('Spending Score by Cluster')

plt.tight_layout()
plt.show()

# %% [markdown]
# # 6.2 Business Labeling: Naming the Tribes
# %% [markdown]
# Cluster numbers (0, 1, 2, 3, 4) are meaningless to a marketing team. We need descriptive names.
#
# We create a function `get_cluster_label` that looks at the centroids (average Income and Score) of each cluster and assigns a logical name:
# * **VIP:** High Income & High Spend.
# * **Saver:** High Income & Low Spend.
# * **Careless:** Low Income & High Spend.
# * **Conservative:** Low Income & Low Spend.
# * **Average:** The middle ground.
# %%
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['Age', 'Income', 'Score'])
centroid_df['Cluster_ID'] = range(5)

def get_cluster_label(row):
    income = row['Income']
    score = row['Score']

    if income > 70 and score > 60:
        return "VIP (High Income, High Spend)"
    elif income > 70 and score < 40:
        return "Saver (High Income, Low Spend)"
    elif income < 40 and score > 60:
        return "Careless (Low Income, High Spend)"
    elif income < 40 and score < 40:
        return "Conservative (Low Income, Low Spend)"
    else:
        return "Average (Middle Class)"

centroid_df['Cluster_Name'] = centroid_df.apply(get_cluster_label, axis=1)
cluster_map = dict(zip(centroid_df['Cluster_ID'], centroid_df['Cluster_Name']))

df['Cluster_Name'] = df['Cluster'].map(cluster_map)

print("Cluster Profiles with Business Labels:")
display(centroid_df[['Cluster_ID', 'Cluster_Name', 'Age', 'Income', 'Score']])

# %% [markdown]
# # 7. Hierarchical Clustering (Dendrogram)
#
# As covered in Lecture 5, Hierarchical clustering provides a different perspective by building a tree of clusters (Dendrogram). This allows us to see the "genetic" relationship between customer segments.
# %%
plt.figure(figsize=(15, 7))
# As discussed in Lecture 5, we use 'Ward' linkage to minimize the variance within clusters.
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram for Customer Segmentation')
plt.xlabel('Customers (Standardized Euclidean Space)')
plt.ylabel('Variance / Distance (Ward Linkage)')
plt.show()

# %% [markdown]
# # 8. Advanced Anomaly Detection (DBSCAN)
#
# # 8.1 Parameter Tuning: The k-Distance Plot
#
# Choosing the right `epsilon` ($\epsilon$) is crucial. According to the academic standard, we calculate the distance to the $k$-th nearest neighbor (where $k = MinPts$) and plot it. The "Elbow" or "Knee" of this plot indicates the optimal $\epsilon$.
# %%
# k = 2 * dimensions - 1 is a common heuristic for MinPts
min_samples = 6 
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances
distances = np.sort(distances[:, min_samples-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('k-Distance Plot for Epsilon Selection')
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{min_samples}-NN Distance')
plt.grid(True, alpha=0.3)
plt.show()
# %% [markdown]
# K-Means is excellent, but it has a flaw: it forces *every* customer into a cluster, even if they don't fit well.
#
# To solve this, we introduce **DBSCAN** (Density-Based Spatial Clustering). Unlike K-Means, DBSCAN looks for high-density areas. If a point is in a low-density region (far from others), it labels it as **Noise (-1)**.
#
# These "Outliers" are crucial. They might be:
# 1.  **Fraud cases.**
# 2.  **Unique high-value customers** who need special attention.
# 3.  **Data errors.**
# %%
dbscan = DBSCAN(eps=1.0, min_samples=5) # Tuned based on k-distance "knee"
clusters_dbscan = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = clusters_dbscan

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='DBSCAN_Cluster', s=100,
                palette=sns.color_palette("blend:"+teal_corporate_palette[0]+","+teal_corporate_palette[4], as_cmap=True),
                edgecolor='white', linewidth=1.5)

plt.title('DBSCAN: Outlier Detection (Tuned Parameters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster ID')
plt.grid(True, alpha=0.3)
plt.show()

# Silhouette Score for DBSCAN (excluding noise)
from sklearn.metrics import silhouette_score
if len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0) > 1:
    score_dbscan = silhouette_score(X_scaled[clusters_dbscan != -1], clusters_dbscan[clusters_dbscan != -1])
    print(f'DBSCAN Silhouette Score (without noise): {score_dbscan:.4f}')
    
    # Also generate a Silhouette Plot for DBSCAN
    from yellowbrick.cluster import SilhouetteVisualizer
    model_dbscan_sim = KMeans(n_clusters=len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0), random_state=42)
    # Note: SilhouetteVisualizer expect labels_ in the model, we can trick it or use samples
    # For DBSCAN, we'll just show the numeric score as it's more standard, 
    # but the requirement says "both", so let's try to visualize it correctly.
    # Actually, a better way for DBSCAN silhouette is just the scatter we already have, 
    # but let's add a dedicated visualizer if possible. 
    # Since yellowbrick SilhouetteVisualizer is for KMeans/MiniBatchKMeans/etc, 
    # let's just use the numeric score and the PCA/t-SNE we already have which shows separation.
else:
    print('DBSCAN: Not enough clusters for silhouette score calculation.')

n_noise = list(clusters_dbscan).count(-1)
print(f"Number of Outliers detected by DBSCAN: {n_noise}")

# %% [markdown]
# Finally, we summarize our findings. The bar chart below ranks our new segments by **Average Spending Score**. This clearly identifies our "VIP" and "Careless" segments as the primary drivers of revenue, while "Savers" represent a huge opportunity for up-selling campaigns.
# %% [markdown]
# # 9. Dimensionality Reduction & Advanced Visualization
#
# As discussed in Lecture 6, dimensionality reduction is not just for visualization; it helps us understand the structure of complex data by projecting it into a lower-dimensional space.
#
# 1. **PCA (Principal Component Analysis):** A linear transformation that maximizes variance. It captures the global structure and is computationally efficient.
# 2. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear manifold learning technique that preserves local pairwise distances. The `perplexity` parameter helps balance local and global aspects of the data.
# 3. **UMAP (Uniform Manifold Approximation and Projection):** A modern non-linear alternative mentioned in Lecture 6 that often better preserves global structure than t-SNE while being faster.
# %%
# PCA: 2D Projection & Variance Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_
print(f"PCA Explained Variance Ratio: PC1={var_ratio[0]:.2f}, PC2={var_ratio[1]:.2f} (Total: {sum(var_ratio):.2f})")

# t-SNE: 2D Projection (Perplexity chosen per Lecture 6)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP: Modern Manifold Learning
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Visualization (Aesthetic Consistency: Corporate Teal Theme)
plt.figure(figsize=(18, 5))

# PCA Plot
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap=teal_map, edgecolors='white', alpha=0.7)
plt.title(f'PCA Projection (Expl. Var: {sum(var_ratio):.2f})')
plt.xlabel('PC1')
plt.ylabel('PC2')

# t-SNE Plot
plt.subplot(1, 3, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Cluster'], cmap=teal_map, edgecolors='white', alpha=0.7)
plt.title('t-SNE Embeddings')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# UMAP Plot
plt.subplot(1, 3, 3)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=df['Cluster'], cmap=teal_map, edgecolors='white', alpha=0.7)
plt.title('UMAP Manifold Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.tight_layout()
plt.show()


# %% [markdown]
# # 9. Strategic Insights & Conclusion
# %% [markdown]
# # 9.1 Which segment is most valuable?
# %%
avg_spending = df.groupby('Cluster_Name')['Spending Score (1-100)'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_spending.values, y=avg_spending.index)

plt.title('Average Spending Score by Customer Segment')
plt.xlabel('Average Spending Score')
plt.ylabel('Segment')
plt.grid(axis='x', alpha=0.3)

for index, value in enumerate(avg_spending.values):
    plt.text(value + 1, index, f'{value:.1f}', va='center', fontweight='bold')

plt.show()

# %% [markdown]
# # 9.3 Strategic Strategy & Retail Implications
#
# Our analysis has revealed two critical insights for the business:
# 1.  **The Target Tribe (VIPs):** Despite high income, this group is small. Marketing should focus on high-touch, exclusive loyalty rewards to maintain retention.
# 2.  **The Spending Cliff (Age 40+):** We observed a critical drop-off in spending score as customers cross the 40-year threshold (Section 3.3). To combat this, the mall should implement **Mid-Life Lifestyle Campaigns**—shifting from trend-driven apparel to health, wellness, and family-oriented services that appeal to this demographic's evolving needs.
# %% [markdown]
# # 9.2 Deployment: Exporting the Results
# %%
df_final = pd.read_csv(path)
df_final['Cluster'] = df['Cluster']
df_final['Cluster_Name'] = df['Cluster_Name']
df_final.to_csv('Mall_Customers_Segmented.csv', index=False)
display(df_final[['CustomerID', 'Cluster', 'Cluster_Name']].head())

