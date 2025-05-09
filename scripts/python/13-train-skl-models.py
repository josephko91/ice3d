import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# get python command line arguments
n_cores = sys.argv[1]
save_dir = sys.argv[2]
data_dir = sys.argv[3]
view = sys.argv[4]

# Set random seed
n_rand = 666

# Load data
data_file = 'ros-tabular-data.parquet'
data_path = os.path.join(data_dir, data_file)
df = pd.read_parquet(data_path)

# Subset data on view
df = df[df['view'] == 'default']

# Define features and targets
features = ['aspect_ratio', 'aspect_ratio_elip', 'extreme_pts',
            'contour_area', 'contour_perimeter', 'area_ratio', 'complexity',
            'circularity']
targets = ['rho_eff', 'sa_eff']

# Subset data for efficiency
df = df.sample(1_000_000, random_state=n_rand)
df.reset_index(inplace=True)

# Split data into features and targets
df_features = df[features]
df_targets = df[targets]
X = df_features
y = df_targets

# Train/test/validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n_rand)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=n_rand)

# Train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on validation data
y_val_pred = reg.predict(X_val)

# Create output dataframe for validation results
df_val_out = y_val.rename(columns={'rho_eff': 'rho_eff_truth', 'sa_eff': 'sa_eff_truth'})
df_val_out['n_arms'] = df['n_arms'].iloc[X_val.index]
df_val_out['rho_eff_pred'] = y_val_pred[:, 0]
df_val_out['sa_eff_pred'] = y_val_pred[:, 1]

# Calculate R2 scores
r2_rho = r2_score(df_val_out['rho_eff_truth'], df_val_out['rho_eff_pred'])
r2_sa = r2_score(df_val_out['sa_eff_truth'], df_val_out['sa_eff_pred'])
print(f'R2 for effective density = {r2_rho}')
print(f'R2 for effective surface area = {r2_sa}')

# Plot results for rho_eff
df_subset = df_val_out.sample(10_000, random_state=n_rand)
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=df_subset, x='rho_eff_pred', y='rho_eff_truth', hue='n_arms', alpha=0.5, legend='full', edgecolor='white')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.axline(xy1=(0, 0), slope=1, color='black')
ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[0], lims[1])
ax.set_aspect('equal')
ax.set_ylabel('predicted')
ax.set_xlabel('truth')
ax.set_title('Effective density [unitless]')
ax.text(0.2, 0.75, f'$R^2$ = {r2_rho:.2f}', transform=plt.gcf().transFigure, ha='left', size=16)
plt.show()

# Plot results for sa_eff
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=df_subset, x='sa_eff_pred', y='sa_eff_truth', hue='n_arms', alpha=0.5, legend='full')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.axline(xy1=(0, 0), slope=1, color='black')
ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[0], lims[1])
ax.set_aspect('equal')
ax.set_ylabel('predicted')
ax.set_xlabel('truth')
ax.set_title('Effective surface area [unitless]')
ax.text(0.6, 0.2, f'$R^2$ = {r2_sa:.2f}', transform=plt.gcf().transFigure, ha='left', size=16)
plt.show()

# Train Random Forest Regressor
rf = RandomForestRegressor(max_depth=15, random_state=n_rand, n_jobs=16)
rf.fit(X_train, y_train)

# Predict on validation data
y_val_pred_rf = rf.predict(X_val)

# Create output dataframe for Random Forest validation results
df_val_out_rf = y_val.rename(columns={'rho_eff': 'rho_eff_truth', 'sa_eff': 'sa_eff_truth'})
df_val_out_rf['n_arms'] = df['n_arms'].iloc[X_val.index]
df_val_out_rf['rho_eff_pred'] = y_val_pred_rf[:, 0]
df_val_out_rf['sa_eff_pred'] = y_val_pred_rf[:, 1]

# Calculate R2 score for Random Forest
r2_rf = r2_score(df_val_out_rf['rho_eff_truth'], df_val_out_rf['rho_eff_pred'])
print(f'R2 value for effective density (Random Forest): {r2_rf}')

# Plot Random Forest results for rho_eff
df_rf_subset = df_val_out_rf.sample(10_000, random_state=n_rand)
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=df_rf_subset, x='rho_eff_pred', y='rho_eff_truth', cmap='magma', fill=True, ax=ax, alpha=0.8)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.axline(xy1=(0, 0), slope=1, color='black')
ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[0], lims[1])
ax.set_aspect('equal')
ax.set_ylabel('Predicted')
ax.set_xlabel('Truth')
ax.set_title('Effective Density [unitless]')
ax.text(0.25, 0.75, f'$R^2$ = {r2_rf:.2f}', transform=plt.gcf().transFigure, ha='left', size=16)
plt.show()