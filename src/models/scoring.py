import joblib
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import auc

from src.features.sampling import calculate_frequency
from src.utils.utils import list_models

def model_summary(model, X_train, y_train, X_test, y_test, sample_weight_train=None, sample_weight_test=None, verbose=False):
    """
    Summarizes the model performance on training and test datasets using Poisson deviance,
    improvement in deviance vs a dummy model, RMSE, and frequency metrics, considering sample weights.

    Parameters:
    - model: The trained model to evaluate.
    - X_train: Training features.
    - y_train: Training target.
    - X_test: Test features.
    - y_test: Test target.
    - sample_weight_train: Weights for training samples.
    - sample_weight_test: Weights for test samples.

    Returns:
    - summary: A dictionary containing the evaluation metrics.
    """
    # Predict on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    X_train_pred = X_train.copy().assign(y_pred=y_train_pred * sample_weight_train)
    X_test_pred = X_test.copy().assign(y_pred=y_test_pred * sample_weight_test)

    # Calculate Poisson deviance
    poisson_deviance_train = mean_poisson_deviance(y_train, y_train_pred, sample_weight=sample_weight_train)
    poisson_deviance_test = mean_poisson_deviance(y_test, y_test_pred, sample_weight=sample_weight_test)

    # Dummy model for comparison
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train, sample_weight=sample_weight_train)
    y_train_dummy_pred = dummy.predict(X_train)
    y_test_dummy_pred = dummy.predict(X_test)

    # Calculate Poisson deviance for dummy model
    poisson_deviance_train_dummy = mean_poisson_deviance(y_train, y_train_dummy_pred, sample_weight=sample_weight_train)
    poisson_deviance_test_dummy = mean_poisson_deviance(y_test, y_test_dummy_pred, sample_weight=sample_weight_test)

    # Calculate improvement in deviance
    improvement_train = 1 - (poisson_deviance_train / poisson_deviance_train_dummy)
    improvement_test = 1 - (poisson_deviance_test / poisson_deviance_test_dummy)

    # Calculate frequency metrics
    frequency_train_actual = calculate_frequency(X_train)
    frequency_train_model = calculate_frequency(X_train_pred, 'y_pred')
    frequency_test_actual = calculate_frequency(X_test)
    frequency_test_model = calculate_frequency(X_test_pred, 'y_pred')

    # Compile summary
    summary = {
        'Poisson Deviance Dummy Train': poisson_deviance_train_dummy * 100,
        'Poisson Deviance Dummy Test': poisson_deviance_test_dummy * 100,
        'Poisson Deviance Train': poisson_deviance_train * 100,
        'Poisson Deviance Test': poisson_deviance_test * 100,
        'Improvement in Deviance Train': improvement_train * 100,
        'Improvement in Deviance Test': improvement_test * 100,
        'Frequency Train Actual': frequency_train_actual * 100,
        'Frequency Train Model': frequency_train_model * 100,
        'Frequency Test Actual': frequency_test_actual * 100,
        'Frequency Test Model': frequency_test_model * 100
    }

    # Print the summary with formatted percentages
    if verbose:
        for key, value in summary.items():
            if 'Improvement' in key or 'Frequency' in key:
                print(f"{key}: {value:.3f}%")
        else:
            print(f"{key}: {value:.3f}")

    return summary, X_train_pred, X_test_pred


def print_summary(summary):
    for key, value in summary.items():
        if 'Improvement' in key or 'Frequency' in key:
            print(f"{key}: {value:.3f}%")
        else:
            print(f"{key}: {value:.3f}")


def lorenz_curve(y_true, y_pred, exposure):
    """
    Calculate Lorenz curve and Gini coefficient.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    exposure : array-like
        Exposure values
        
    Returns:
    --------
    tuple
        ((cumulative_exposure, cumulative_claim_amount), gini_coefficient)
    """
    # Convert inputs to numpy arrays
    y_true, y_pred, exposure = map(np.asarray, [y_true, y_pred, exposure])

    # Order samples by increasing predicted risk
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    
    # Calculate cumulative values
    cumulative_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulative_claim_amount /= cumulative_claim_amount[-1]
    
    cumulative_exposure = np.cumsum(ranked_exposure)
    cumulative_exposure /= cumulative_exposure[-1]
    
    # Calculate Gini coefficient
    gini_coefficient = 1 - 2 * auc(cumulative_exposure, cumulative_claim_amount)
    
    return ((cumulative_exposure, cumulative_claim_amount), gini_coefficient)


def evaluate_models(train, y_freq_train, test, y_freq_test, weights_train, weights_test):
    """
    Evaluate multiple models and return their performance metrics in a dictionary.
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    y_freq_train : Series
        Training target values
    test : DataFrame
        Test data
    y_freq_test : Series
        Test target values
    weights_train : Series
        Sample weights for training data
    weights_test : Series
        Sample weights for test data
        
    Returns:
    --------
    dict
        Dictionary with model evaluation metrics
    """
    model_paths = list_models()
    models = {model[0]: joblib.load(model[1]) for model in model_paths}
    
    dict_to_df = defaultdict(list)
    for model in models.items():
        summary = model_summary(model[1], train, y_freq_train, test, y_freq_test, weights_train, weights_test)
        dict_to_df["model"] += [model[0]]
        dict_to_df["pipeline"] += [model[1]]
        dict_to_df["ImprovementDevianceTrain"] += [summary[0]["Improvement in Deviance Train"] / 100]
        dict_to_df["ImprovementDevianceTest"] += [summary[0]["Improvement in Deviance Test"] / 100]
        ginis_train = lorenz_curve(summary[1]['ClaimNb'], summary[1]['y_pred'], summary[1]['Exposure'])
        ginis_test = lorenz_curve(summary[2]['ClaimNb'], summary[2]['y_pred'], summary[2]['Exposure'])
        dict_to_df["LorenzCurveTrain"] += [ginis_train[0]]
        dict_to_df["GiniCoefficientTrain"] += [ginis_train[1]]
        dict_to_df["LorenzCurveTest"] += [ginis_test[0]]
        dict_to_df["GiniCoefficientTest"] += [ginis_test[1]]
    
    return pd.DataFrame(dict_to_df)

def plot_lorenz_curves(model_stats):
    """Plot the Lorenz curves for each model in model_stats."""
    plt.figure(figsize=(10, 8))

    model_stats = model_stats.sort_values(by='GiniCoefficientTest', ascending=False).head(3)

    for i, (lorenz_data, model_name, improvement, gini) in enumerate(zip(model_stats.LorenzCurveTest, model_stats.model, model_stats.ImprovementDevianceTest, model_stats.GiniCoefficientTest)):
        # Extract x and y coordinates for each model
        x_coords, y_coords = lorenz_data
        plt.plot(x_coords, y_coords, label=f'{model_name} (DevianceImp: {improvement*100:.2f}%, Gini: {gini:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Equality/Random model')
    plt.xlabel('Cumulative Exposure')
    plt.ylabel('Cumulative Claims')
    plt.title('Lorenz Curve - Test Set')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importances(model):
    """Plot feature importances for the given model."""
    # Get feature names and importances
    feature_names = model.named_steps['preprocessor'][-1].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_

    # Create a bar plot for feature importances
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(1)