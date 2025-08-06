# %% [markdown]
# # Turn‐Assisted Deep Cold Rolling NN Grid‐Search


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, r2_score

import keras
from keras import layers
from keras.callbacks import EarlyStopping

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# %% [markdown]
# ## 1. Load & Inspect Data

# %%
# Synthesize the data
np.random.seed(42)
df = pd.DataFrame({
    'ball_diameter':     np.random.uniform(10, 50, 200),
    'rolling_force':     np.random.uniform(100, 500, 200),
    'initial_roughness': np.random.uniform(0.1, 2.0, 200),
    'num_passes':        np.random.randint(1, 10, 200),
    'surface_hardness':  np.random.uniform(200, 400, 200),
    'final_roughness':   np.random.uniform(0.05, 1.0, 200)
})

print("Dataset shape:", df.shape)
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
print(df.describe().to_string())

# %%
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Scaling

# %%
print("Scaling method: StandardScaler (zero mean, unit variance).")
features = ['ball_diameter','rolling_force','initial_roughness','num_passes']
X = df[features].values
y_h = df['surface_hardness'].values
y_r = df['final_roughness'].values
y_j = df[['surface_hardness','final_roughness']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ## 4. Model Builder & Hyperparameter Grid

# %%
def build_model(n_hidden, n_neurons, activation, dropout_rate, optimizer, output_dim):
    m = keras.Sequential()
    m.add(layers.Input(shape=(X_scaled.shape[1],)))
    for _ in range(n_hidden):
        m.add(layers.Dense(n_neurons, activation=activation))
        m.add(layers.Dropout(dropout_rate))
    m.add(layers.Dense(output_dim))
    m.compile(loss='mse', optimizer=optimizer)
    return m

param_grid = {
    'n_hidden':     [1, 2, 3],
    'n_neurons':    [5, 10, 15],
    'activation':   ['sigmoid','relu','tanh'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'optimizer':    ['sgd','rmsprop','adam']
}

early_stop_cv    = EarlyStopping(monitor='loss',     patience=10, restore_best_weights=True)
early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# %% [markdown]
# ## 5. Grid Search with 10‐Fold CV

# %%
def run_grid_cv(X, y, output_dim):
    records = []
    for params in ParameterGrid(param_grid):
        mse_fold, r2_fold = [], []
        for ti, vi in kfold.split(X):
            Xtr, Xte = X[ti], X[vi]
            ytr, yte = y[ti], y[vi]
            ytr2 = ytr.reshape(-1, output_dim)
            yte2 = yte.reshape(-1, output_dim)

            m = build_model(**params, output_dim=output_dim)
            m.fit(Xtr, ytr2, epochs=100, batch_size=16,
                  callbacks=[early_stop_cv], verbose=0)

            ypred = m.predict(Xte)
            if output_dim == 1:
                ypred = ypred.flatten()
                yte2 = yte2.flatten()

            mse_fold.append(mean_squared_error(yte2, ypred))
            r2_fold.append(r2_score(yte2, ypred))

        records.append({**params,
                        'avg_mse': np.mean(mse_fold),
                        'std_mse': np.std(mse_fold),
                        'avg_r2':  np.mean(r2_fold),
                        'std_r2':  np.std(r2_fold)})
    return pd.DataFrame(records)

# %%
print("Surface hardness CV...")
df_h = run_grid_cv(X_scaled, y_h, output_dim=1)
print("Final roughness CV...")
df_r = run_grid_cv(X_scaled, y_r, output_dim=1)
print("Joint model CV...")
df_j = run_grid_cv(X_scaled, y_j, output_dim=2)

# %% [markdown]
# ## 6. Show Top‐10 Hyperparameters

# %%
for name, df_res in [('Hardness',df_h),('Roughness',df_r),('Joint',df_j)]:
    print(f"\nTop 10 for {name}:")
    display(df_res.sort_values('avg_mse').head(10))

# %% [markdown]
# ## 7. Hyperparameter Justification Plot

# %%
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_h, x='n_neurons', y='avg_mse',
                hue='n_hidden', palette='viridis', s=80, alpha=0.7)
plt.title("Hardness: avg_mse vs n_neurons (hue=n_hidden)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Refit Best Models & Plot Train/Val Loss

# %%
# Extract best parameter dicts
best_h = df_h.loc[df_h['avg_mse'].idxmin()].to_dict()
best_r = df_r.loc[df_r['avg_mse'].idxmin()].to_dict()
best_j = df_j.loc[df_j['avg_mse'].idxmin()].to_dict()
for d in (best_h, best_r, best_j):
    for k in ('avg_mse','std_mse','avg_r2','std_r2'):
        d.pop(k, None)

print("Best params (hardness):", best_h)
print("Best params (roughness):", best_r)
print("Best params (joint):", best_j)

# Train final with validation split
def train_final(X, y, params, output_dim):
    m = build_model(**params, output_dim=output_dim)
    h = m.fit(X, y.reshape(-1,output_dim),
              validation_split=0.1, epochs=200, batch_size=16,
              callbacks=[early_stop_final], verbose=0)
    return m, h

model_h, hist_h = train_final(X_scaled, y_h, best_h, 1)
model_r, hist_r = train_final(X_scaled, y_r, best_r, 1)
model_j, hist_j = train_final(X_scaled, y_j, best_j, 2)

# Plot losses
for title, hist in [('Hardness',hist_h),('Roughness',hist_r),('Joint',hist_j)]:
    plt.figure(figsize=(6,4))
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9. Final Performance on Full Data

# %%
yhat_h = model_h.predict(X_scaled).flatten()
yhat_r = model_r.predict(X_scaled).flatten()
yhat_j = model_j.predict(X_scaled)

print("Hardness   MSE/R²:",
      mean_squared_error(y_h,yhat_h), r2_score(y_h,yhat_h))
print("Roughness  MSE/R²:",
      mean_squared_error(y_r,yhat_r), r2_score(y_r,yhat_r))
print("Joint      MSE/R²:",
      mean_squared_error(y_j,yhat_j, multioutput='raw_values'),
      r2_score(y_j,yhat_j, multioutput='raw_values'))