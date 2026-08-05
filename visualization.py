import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. load EuroSAT numpy data
# X_data shape expected: (N, 12288) or (N, 64, 64, 3)
# y_data shape expected: (N,)
X_data = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\X_eurosat.npy')
X_data2 = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\X_eurosat.npy')
X_data3 = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\X_eurosat.npy')

y_data = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\y_eurosat.npy')
y_data2 = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\y_eurosat.npy')
y_data3 = np.load('C:\\Users\\ayele\\Documents\\ML_2025\\ML_project\\ML_Final_Project\\y_eurosat.npy')


if X_data.ndim > 2:
    X_data = X_data.reshape(X_data.shape[0], -1)

# subsampling for fast theoretical computations
np.random.seed(42)
idx = np.random.choice(len(X_data), size=5000, replace=False)
X_sub, y_sub = X_data[idx], y_data[idx]

# VISUALIZATION 1: PCA Variance vs VC-Dimension & Sauer Bound
pca_full = PCA(n_components=200).fit(X_sub)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

k_components = np.arange(1, 201)
vc_dims = k_components + 1  # VC dim for linear classifier in k dimensions

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Number of PCA Components (k)')
ax1.set_ylabel('Cumulative Explained Variance Ratio', color=color)
ax1.plot(k_components, cum_variance, color=color, linewidth=2.5, label='Cumulative Variance')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=100, color='red', linestyle='--', label='PCA k=100 (Project Choice)')

ax2 = ax1.twinx()  
color = 'tab:purple'
ax2.set_ylabel('Linear VC Dimension (k+1)', color=color)
ax2.plot(k_components, vc_dims, color=color, linestyle=':', linewidth=2, label='VC Dimension')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Figure 1: Trade-off between Information Retention and VC-Capacity')
fig.tight_layout()
plt.savefig('vis1_pca_vc_tradeoff.png', dpi=300)
plt.show()

# VISUALIZATION 2: Empirical Rademacher Complexity Calculation

def compute_empirical_rademacher(X, num_samples_list, num_boots=20):
    """
    Computes Empirical Rademacher Complexity for linear class bounded in L2 norm
    R_S(H) = E_sigma [ sup_{||w||<=1} (1/m) sum_{i=1}^m sigma_i <w, x_i> ]
           = (1/m) E_sigma [ || sum_{i=1}^m sigma_i x_i ||_2 ]
    """
    rademacher_results = []
    # normalize features to unit norm for standard Rademacher bound
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    for m in num_samples_list:
        rad_m = []
        for _ in range(num_boots):
            sub_idx = np.random.choice(len(X_norm), size=m, replace=False)
            X_m = X_norm[sub_idx]
            sigma = np.random.choice([-1, 1], size=m)
            # dual norm calculation
            vector_sum = np.dot(sigma, X_m)
            rad_val = (1.0 / m) * np.linalg.norm(vector_sum)
            rad_m.append(rad_val)
        rademacher_results.append(np.mean(rad_m))
    return rademacher_results

sample_sizes = [50, 100, 250, 500, 1000, 2000, 3500, 5000]
rad_complexities = compute_empirical_rademacher(X_sub, sample_sizes)

plt.figure(figsize=(9, 5))
plt.plot(sample_sizes, rad_complexities, 'o-', color='darkgreen', linewidth=2, label=r'Empirical $\hat{\mathcal{R}}_S(\mathcal{H})$')
# the theoretical O(1/sqrt(m)) rate overlay plot
theoretical_rate = rad_complexities[0] * np.sqrt(sample_sizes[0]) / np.sqrt(sample_sizes)
plt.plot(sample_sizes, theoretical_rate, '--', color='orange', label=r'Theoretical $O(1/\sqrt{m})$ Rate')

plt.title('Figure 2: Empirical Rademacher Complexity vs. Sample Size $m$')
plt.xlabel('Sample Size ($m$)')
plt.ylabel(r'Rademacher Complexity $\hat{\mathcal{R}}_S(\mathcal{H})$')
plt.legend()
plt.tight_layout()
plt.savefig('vis2_rademacher_complexity.png', dpi=300)
plt.show()

 
# VISUALIZATION 3: Geometry of Agnostic Setting (t-SNE Embeddings)
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

class_names = {
    0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 
    3: 'Highway', 4: 'Industrial', 5: 'Pasture', 
    6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'
}

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_sub[:2000])

plt.figure(figsize=(12, 8)) 
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_sub[:2000], cmap='tab10', alpha=0.7, s=15)

cbar = plt.colorbar(scatter)
cbar.set_ticks(np.arange(10)) 
cbar.set_ticklabels([class_names[i] for i in range(10)])
cbar.ax.tick_params(labelsize=12) 
cbar.set_label('Land Cover Class', fontsize=14) 

plt.title('Figure 3: t-SNE EuroSAT Manifold — High Agnostic Bayes Error for Linear Bounds', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.tight_layout()
plt.savefig('vis3_tsne_bayes_risk.png', dpi=300)
plt.show()


# fig 4 - SRM

# reduce dimensions for computational speed - pca
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_data2)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_data2, test_size=0.2, random_state=42)

# define a nested sequence of hypothesis classes (Tree Depths)
depths = [2, 4, 6, 8, 10, 15, 20, 25]
train_errors = []
test_errors = []

for d in depths:
    # Train Random Forest with constrained depth
    rf = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # calculating Empirical Risk L_S(h) and True Risk estimation L_D(h)
    err_train = zero_one_loss(y_train, rf.predict(X_train))
    err_test = zero_one_loss(y_test, rf.predict(X_test))
    
    train_errors.append(err_train)
    test_errors.append(err_test)

# plotting the SRM Curve
plt.figure(figsize=(10, 6))
plt.plot(depths, train_errors, 'b-o', linewidth=2, label=r'Empirical Risk $L_S(h)$ (Train Error)')
plt.plot(depths, test_errors, 'r-s', linewidth=2, label=r'Estimated True Risk $L_{\mathcal{D}}(h)$ (Test Error)')

# Highlight the optimal structural risk
optimal_depth = depths[np.argmin(test_errors)]
plt.axvline(x=optimal_depth, color='k', linestyle='--', label=f'SRM Minimum (Depth={optimal_depth})')

plt.title('Figure 4: Structural Risk Minimization in Random Forests')
plt.xlabel('Hypothesis Class Complexity (Maximum Tree Depth)')
plt.ylabel('Classification Error (0-1 Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('vis4_srm_curve.png', dpi=300)
plt.show()


from sklearn.metrics.pairwise import euclidean_distances

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 0. Load Data and Setup


class_names = {
    0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 
    3: 'Highway', 4: 'Industrial', 5: 'Pasture', 
    6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'
}

# flatten data for ML operations
X_flat = X_data3.reshape(X_data3.shape[0], -1)

# VISUALIZATION A: Spectral Intensity Distributions
# Calculate average pixel intensity for each image (grayscale equivalent)
mean_intensities = np.mean(X_flat, axis=1)

plt.figure(figsize=(10, 6))
# Plot overlapping classes to show Agnostic setting
problem_classes = [0, 2, 6] # AnnualCrop, Herbaceous, PermanentCrop

for c in problem_classes:
    sns.kdeplot(mean_intensities[y_data3 == c], fill=True, 
                label=class_names[c], alpha=0.5, linewidth=2)

plt.title('Figure A: Spectral Distribution Overlap (Agnostic PAC Setting)')
plt.xlabel('Average Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('vis_a_spectral_overlap.png', dpi=300)
plt.show()


# VISUALIZATION B: PCA Eigen-Images
pca = PCA(n_components=100, random_state=42)
pca.fit(X_flat)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
plt.suptitle('Figure B: Top 5 Principal Components (Eigen-Images)', y=1.05)

for i, ax in enumerate(axes):
    # normalize the component for visualization
    component = pca.components_[i]
    comp_norm = (component - component.min()) / (component.max() - component.min())
    # reshape back to 64x64x3
    eigen_image = comp_norm.reshape(64, 64, 3)
    
    ax.imshow(eigen_image)
    ax.set_title(f'PC {i+1}\nVar: {pca.explained_variance_ratio_[i]*100:.1f}%')
    ax.axis('off')

plt.tight_layout()
plt.savefig('vis_b_eigen_images.png', dpi=300)
plt.show()

# VISUALIZATION C: Class Centroid Distance Matrix
# Calculate the centroid (mean vector) for each class
centroids = []
for i in range(10):
    class_mean = np.mean(X_flat[y_data3 == i], axis=0)
    centroids.append(class_mean)
centroids = np.array(centroids)

# Calculate pairwise Euclidean distances between centroids
dist_matrix = euclidean_distances(centroids)

plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, annot=False, cmap='viridis_r', 
            xticklabels=[class_names[i] for i in range(10)],
            yticklabels=[class_names[i] for i in range(10)])

plt.title(r'Figure C: Class Centroid Distance Matrix in Feature Space $\mathbb{R}^D$')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vis_c_centroid_distances.png', dpi=300)
plt.show()