import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

# Load the data
phenometrics = r'C:\Users\uv18\Dropbox\Article_phenocam\tables\treinamento_modelo_geral\training_dataset_final.xlsx'
df = pd.read_excel(phenometrics)

dict_explanotory_variables = {
    'D1': 'D1 HLS',
    'Di': 'Di HLS',
    'D2': 'D2 HLS',
    'D3': 'D3 HLS',
    'Dd': 'Dd HLS',
    'D4': 'D4 HLS'
}

dict_response_variables = {
    'Sowing': 'Planting Phenocam',
    'Emerging': 'Emergence phenocam'
}

# Convert date to DOY (Day of Year)
def convert_date_to_doy(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    doy = pd.to_datetime(f'{year}-{month}-{day}').timetuple().tm_yday
    return doy

# Calculate metrics for model evaluation
def _calculate_metrics(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)
    observed_non_nan = observed[~np.isnan(observed) & ~np.isnan(predicted)]
    predicted_non_nan = predicted[~np.isnan(observed) & ~np.isnan(predicted)]
    mse = np.nanmean((observed_non_nan - predicted_non_nan) ** 2)
    rmse = np.sqrt(mse)
    mae = np.nanmean(np.abs(observed_non_nan - predicted_non_nan))
    bias = np.nanmean(predicted_non_nan - observed_non_nan)
    r = np.corrcoef(observed_non_nan, predicted_non_nan)[0, 1]
    r2 = r ** 2
    return rmse, mae, r2, bias

# Apply the conversion function to the explanatory variables
df_select = df.iloc[:, 2:11]
for column in df_select.columns:
    df_select[column] = df_select[column].apply(convert_date_to_doy)

# Prepare plot
fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))  # Adjusted size for A4 paper
titles = ['a) Sowing MLR', 'b) Emerging MLR', 'c) Sowing ENR', 'd) Emerging ENR', 'e) Sowing SVM', 'f) Emerging SVM']
font = {'fontname': 'Times New Roman'}

# Define models
models = {
    "MLR": LinearRegression(),
    "ElasticNet": ElasticNet(max_iter=10000, tol=1e-4),
    "SVM": SVR()
}

# Hyperparameters for ElasticNet and SVM
params = {
    "ElasticNet": {
        "alpha": np.logspace(-4, 1, 6),
        "l1_ratio": np.linspace(0.1, 0.9, 9)
    },
    "SVM": {
        "C": np.logspace(-2, 2, 5),
        "epsilon": np.linspace(0.1, 1.0, 10)
    }
}

# Hyperparameter tuning function
def tune_model(model, param_grid, X, y):
    loo = LeaveOneOut()
    best_params = None
    best_score = float('inf')
    for param_combination in [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]:
        model.set_params(**param_combination)
        predictions, actuals = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            actuals.append(y_test.values[0])
        mse = mean_squared_error(actuals, predictions)
        if mse < best_score:
            best_score = mse
            best_params = param_combination
    model.set_params(**best_params)
    return model

# Loop through models and response variables
for model_name, model in models.items():
    for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
        X = df_select[dict_explanotory_variables.values()]
        y = df[response_value].apply(convert_date_to_doy)

        if model_name in params:
            model = tune_model(model, params[model_name], X, y)

        loo = LeaveOneOut()
        predictions, actuals = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            actuals.append(y_test.values[0])

        rmse, mae, r2, bias = _calculate_metrics(actuals, predictions)
        df['predictions'] = predictions
        df['actuals'] = actuals

        colors = {'Soybeans': 'green', 'Corn': 'yellow'}
        markers = {2023: 'o', 2022: 's', 2021: 'D'}

        ax = axes[list(models.keys()).index(model_name), i]
        for crop in df['Crop'].unique():
            for year in df['Ano'].unique():
                subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
                ax.scatter(subset['actuals'], subset['predictions'],
                           color=colors[crop], marker=markers[year],
                           edgecolor='black', s=50, alpha=0.5,
                           label=f'{crop} {year}')

        min_val = min(df['predictions'].min(), df['actuals'].min())
        max_val = max(df['predictions'].max(), df['actuals'].max())
        padding = 10
        min_limit = min_val - padding
        max_limit = max_val + padding

        ax.plot([min_limit, max_limit], [min_limit + 10, max_limit + 10], color='blue', linestyle='--', linewidth=1,
                label='+10 days')  # Line for +10 days

        ax.plot([min_limit, max_limit], [min_limit - 10, max_limit - 10], color='green', linestyle='--', linewidth=1,
                label='-10 days')  # Line for -10 days

        ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2,
                label='1:1 Line')
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)
        ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
        ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=12, **font)  # Adjust font size for A4
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

        textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.05, textstr, fontsize=8, bbox=props, transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right')  # Adjust font size for A4
        ax.text(-0.1, 1.05, titles[list(models.keys()).index(model_name) * 2 + i], transform=ax.transAxes, fontsize=12,
                # Adjust font size for A4
                weight='bold', **font, ha='left')

# Collect all legend handles and labels
handles, labels = [], []
for ax in axes.flatten():
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)


handles[0], handles[1], handles[2], handles[3], handles[4], handles[5] = handles[2], handles[0], handles[1], handles[5], handles[3], handles[4]
labels[0], labels[1], labels[2], labels[3], labels[4], labels[5] = labels[2], labels[0], labels[1], labels[5], labels[3], labels[4]

# Create a single legend outside the plot area
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12, frameon=False)  # Adjust the number of columns

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to fit the legend
plt.savefig('phenology_dates_all_models_A4_legenda.png', dpi=300)
plt.show()
