import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
phenometrics = r'C:\Users\uv18\Dropbox\Article_phenocam\tables\separar_por_cultura\training_dataset_final_soybeans.xlsx'
df = pd.read_excel(phenometrics)

dict_explanotory_variables = {'D1': 'D1 HLS',
                              'Di': 'Di HLS',
                              'D2': 'D2 HLS',
                              'D3': 'D3 HLS',
                              'Dd': 'Dd HLS',
                              'D4': 'D4 HLS'}

dict_response_variables = {'planting': 'Planting Phenocam',
                           'emergence': 'Emergence phenocam',
                           'harvest': 'harvest Phenocam'}

def convert_date_to_doy(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    doy = pd.to_datetime(f'{year}-{month}-{day}').timetuple().tm_yday
    return doy

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

df_select = df.iloc[:, 2:11]
for column in df_select.columns:
    df_select[column] = df_select[column].apply(convert_date_to_doy)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # Half A4 size in inches (portrait mode)
titles = ['a)', 'b)', 'c)']
font = {'fontname': 'Times New Roman'}

for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
    X = df_select[dict_explanotory_variables.values()]
    y = df[response_value].apply(convert_date_to_doy)  # Convert response variable values to DOY

    alpha_values = np.logspace(-4, 1, 6)
    l1_ratio_values = np.linspace(0.1, 0.9, 9)
    loo = LeaveOneOut()

    best_alpha = None
    best_l1_ratio = None
    best_score = float('inf')
    tolerance = 1e-4

    for alpha in alpha_values:
        for l1_ratio in l1_ratio_values:
            predictions = []
            actuals = []
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=tolerance)

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
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    print(f'{response_key} - Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}, Best MSE: {best_score}')

    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, tol=tolerance)
    predictions = []
    actuals = []

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
    markers = {2021: 'o', 2022: 's', 2023: 'D'}

    ax = axes[i]
    for crop in df['Crop'].unique():
        for year in df['Ano'].unique():
            subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
            ax.scatter(subset['actuals'], subset['predictions'],
                       color=colors[crop], marker=markers[year],
                       edgecolor='black', s=100, alpha=0.7,
                       label=f'{crop} {year}')

    min_val = min(df['predictions'].min(), df['actuals'].min())
    max_val = max(df['predictions'].max(), df['actuals'].max())
    padding = 10
    min_limit = min_val - padding
    max_limit = max_val + padding

    ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=20, **font)
    ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=20, **font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)

    textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')
    ax.text(-0.1, 1.05, titles[i], transform=ax.transAxes, fontsize=20, weight='bold', **font, ha='left')

    # Save the trained model to a file
    model_file_path = f'elastic_net_model_{response_key}_soybeans.pkl'
    joblib.dump(model, model_file_path)
    print(f'Model for {response_key} saved to {model_file_path}')

plt.tight_layout()
plt.savefig('phenology_dates_elastic_net_soybeans.png', dpi=300)
plt.show()


####################################################### Using Lasso regression ############################


import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

phenometrics = r'C:\Users\uv18\Dropbox\Article_phenocam\tables\treinamento_separar_por_cultura\training_dataset_final_corn.xlsx'
df = pd.read_excel(phenometrics)

dict_explanotory_variables = {'D1': 'D1 HLS',
                              'Di': 'Di HLS',
                              'D2': 'D2 HLS',
                              'D3': 'D3 HLS',
                              'Dd': 'Dd HLS',
                              'D4': 'D4 HLS'}

dict_response_variables = {'planting': 'Planting Phenocam',
                           'emergence': 'Emergence phenocam',
                           'harvest': 'harvest Phenocam'}

def convert_date_to_doy(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    doy = pd.to_datetime(f'{year}-{month}-{day}').timetuple().tm_yday
    return doy

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

df_select = df.iloc[:, 2:11]
for column in df_select.columns:
    df_select[column] = df_select[column].apply(convert_date_to_doy)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # Half A4 size in inches (portrait mode)
titles = ['a)', 'b)', 'c)']
font = {'fontname': 'Times New Roman'}

for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
    X = df_select[dict_explanotory_variables.values()]
    y = df[response_value].apply(convert_date_to_doy)  # Convert response variable values to DOY

    # Define the parameter grid for Lasso
    alpha_values = np.logspace(-4, 1, 6)
    loo = LeaveOneOut()

    best_alpha = None
    best_score = float('inf')

    for alpha in alpha_values:
        predictions = []
        actuals = []
        model = Lasso(alpha=alpha, max_iter=10000)

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
            best_alpha = alpha

    print(f'{response_key} - Best alpha: {best_alpha}, Best MSE: {best_score}')

    model = Lasso(alpha=best_alpha, max_iter=10000)
    predictions = []
    actuals = []

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
    markers = {2021: 'o', 2022: 's', 2023: 'D'}

    ax = axes[i]
    for crop in df['Crop'].unique():
        for year in df['Ano'].unique():
            subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
            ax.scatter(subset['actuals'], subset['predictions'],
                       color=colors[crop], marker=markers[year],
                       edgecolor='black', s=100, alpha=0.7,
                       label=f'{crop} {year}')

    min_val = min(df['predictions'].min(), df['actuals'].min())
    max_val = max(df['predictions'].max(), df['actuals'].max())
    padding = 10
    min_limit = min_val - padding
    max_limit = max_val + padding

    ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=14, **font)
    ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=14, **font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=12)

    textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')
    ax.text(-0.1, 1.05, titles[i], transform=ax.transAxes, fontsize=16, weight='bold', **font, ha='left')

plt.tight_layout()
plt.savefig('phenology_dates_lasso_corn.png', dpi=300)
plt.show()


################################################# using MLR ##############################################


import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

phenometrics = r'C:\Users\uv18\Dropbox\Article_phenocam\tables\treinamento_separar_por_cultura\training_dataset_final_soybeans.xlsx'
df = pd.read_excel(phenometrics)

dict_explanotory_variables = {'D1': 'D1 HLS',
                              'Di': 'Di HLS',
                              'D2': 'D2 HLS',
                              'D3': 'D3 HLS',
                              'Dd': 'Dd HLS',
                              'D4': 'D4 HLS'}

dict_response_variables = {'planting': 'Planting Phenocam',
                           'emergence': 'Emergence phenocam',
                           'harvest': 'harvest Phenocam'}

def convert_date_to_doy(date):
    year = int(str(date)[:4])
    month = int(str(date)[4:6])
    day = int(str(date)[6:8])
    doy = pd.to_datetime(f'{year}-{month}-{day}').timetuple().tm_yday
    return doy

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

df_select = df.iloc[:, 2:11]
for column in df_select.columns:
    df_select[column] = df_select[column].apply(convert_date_to_doy)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # Half A4 size in inches (portrait mode)
titles = ['a)', 'b)', 'c)']
font = {'fontname': 'Times New Roman'}

for i, (response_key, response_value) in enumerate(dict_response_variables.items()):
    X = df_select[dict_explanotory_variables.values()]
    y = df[response_value].apply(convert_date_to_doy)  # Convert response variable values to DOY

    loo = LeaveOneOut()

    predictions = []
    actuals = []

    model = LinearRegression()

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
    markers = {2021: 'o', 2022: 's', 2023: 'D'}

    ax = axes[i]
    for crop in df['Crop'].unique():
        for year in df['Ano'].unique():
            subset = df[(df['Crop'] == crop) & (df['Ano'] == year)]
            ax.scatter(subset['actuals'], subset['predictions'],
                       color=colors[crop], marker=markers[year],
                       edgecolor='black', s=100, alpha=0.7,
                       label=f'{crop} {year}')

    min_val = min(df['predictions'].min(), df['actuals'].min())
    max_val = max(df['predictions'].max(), df['actuals'].max())
    padding = 10
    min_limit = min_val - padding
    max_limit = max_val + padding

    ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    ax.set_xlabel(f'Observed {response_key} (DOY)', fontsize=14, **font)
    ax.set_ylabel(f'Predicted {response_key} (DOY)', fontsize=14, **font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=12)

    textstr = f'R² = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nBIAS = {bias:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')
    ax.text(-0.1, 1.05, titles[i], transform=ax.transAxes, fontsize=16, weight='bold', **font, ha='left')

plt.tight_layout()
plt.savefig('phenology_dates_MLR_soybeans.png', dpi=300)
plt.show()
