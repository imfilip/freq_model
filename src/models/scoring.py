import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error

from src.features.sampling import calculate_frequency

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