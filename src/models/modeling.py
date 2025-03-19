import joblib
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.data.utils import prepare_target_and_weights
from src.models.scoring import model_summary, print_summary
from src.features.transformers import create_feature_pipeline
from xgboost import XGBRegressor

def train_and_evaluate_model(train, test, pipeline_features, regressor=DummyRegressor, params=None, verbose=False):
    y_train = train['ClaimNb']
    weights_train = train['Exposure']
    y_freq_train = y_train / weights_train

    y_test = test['ClaimNb']
    weights_test = test['Exposure']
    y_freq_test = y_test / weights_test

    params = params or {
        'strategy':'mean'
    }

    model = Pipeline([
        ('preprocessor', pipeline_features),
        ('regressor', regressor(**params))
    ])

    # Check if the regressor is XGBRegressor to handle exposure weights properly
    if isinstance(model.named_steps['regressor'], XGBRegressor):
        model.fit(
            train,
            y=y_freq_train,
            regressor__sample_weight=weights_train,
            regressor__eval_set=[(pipeline_features.transform(test), y_freq_test)],
            regressor__sample_weight_eval_set=[weights_test],
            regressor__verbose=verbose
        )
    else:
        model.fit(
            train,
            y=y_freq_train,
            regressor__sample_weight=weights_train
        )

    summary = model_summary(model, train, y_freq_train, test, y_freq_test, weights_train, weights_test)
    
    return model, summary


def save_model_with_pipeline(model, filename, compress=3):
    """
    Saves a trained model along with its preprocessing pipeline to a file.
    
    Parameters:
    -----------
    model : Pipeline or estimator
        The trained model to save. If it's a scikit-learn Pipeline, it will save
        the entire pipeline including preprocessing steps.
    
    filename : str
        Path where the model will be saved. If no extension is provided,
        '.joblib' will be added automatically.
    
    compress : int, default=3
        Compression level to use when saving. Higher values mean better
        compression but slower processing. Range is 0-9.
    
    Returns:
    --------
    str
        Path to the saved model file
    """
    # Add .joblib extension if not present
    if not filename.endswith('.joblib'):
        filename += '.joblib'
    
    # Save the model
    joblib.dump(model, filename, compress=compress)
    
    print(f"Model successfully saved to {filename}")
    return filename

#TODO: Work on this. I would like to use this metric for scoring in xgboost tuning and use it in early stopping.
def custom_metric_weighted(y_true, y_pred, sample_weight):
    eps = 1e-10
    y_pred = np.maximum(y_pred, eps)
    y_true = np.maximum(y_true, eps)
    
    dev = 2 * (y_true * np.log(y_true/y_pred) - (y_true - y_pred))
    return float(np.average(dev, weights=sample_weight))


def tune_glm_with_feature_pipeline(train, test, numerical_features, categorical_features, n_trials=10, random_state=42):
    """
    Przeprowadza optymalizację hiperparametrów dla modelu GLM (PoissonRegressor) 
    wraz z pipelineami przetwarzania cech.
    
    Parametry:
    -----------
    train : DataFrame
        Zbiór treningowy
    test : DataFrame
        Zbiór testowy
    numerical_features : list
        Lista nazw cech numerycznych
    categorical_features : list
        Lista nazw cech kategorycznych
    n_trials : int, default=10
        Liczba iteracji optymalizacji
    random_state : int, default=42
        Ziarno losowości
        
    Zwraca:
    --------
    dict
        Najlepsze znalezione parametry
    """
    y_train, weights_train, y_freq_train, y_test, weights_test, y_freq_test = prepare_target_and_weights(train, test)
    
    # Przestrzeń parametrów dla PoissonRegressor i pipeline cech
    space = {
        # Parametry dla PoissonRegressor
        'regressor__alpha': hp.loguniform('regressor__alpha', np.log(1e-6), np.log(1.0)),
        'regressor__max_iter': hp.quniform('regressor__max_iter', 20, 100, 20),
        'regressor__tol': hp.loguniform('regressor__tol', np.log(1e-6), np.log(1e-3)),
        
        # Parametry dla feature pipeline
        'preprocessor__n_bins_drivage': hp.quniform('preprocessor__n_bins_drivage', 8, 20, 2),
        'preprocessor__n_bins_vehage': hp.quniform('preprocessor__n_bins_vehage', 5, 15, 1),
        
        # Wartości graniczne dla CapTransformer
        'preprocessor__cap_veh_power': hp.quniform('preprocessor__cap_veh_power', 10, 15, 1),
        'preprocessor__cap_veh_age': hp.quniform('preprocessor__cap_veh_age', 15, 25, 1),
        'preprocessor__cap_driv_age': hp.quniform('preprocessor__cap_driv_age', 80, 100, 5),
        'preprocessor__cap_bonus_malus': hp.quniform('preprocessor__cap_bonus_malus', 100, 150, 10),
        'preprocessor__cap_density': hp.quniform('preprocessor__cap_density', 50000, 150000, 10000)
    }

    def objective(params):
        # Konwersja parametrów do odpowiednich typów
        params_regressor = {
            'alpha': float(params['regressor__alpha']),
            'max_iter': int(params['regressor__max_iter']),
            'tol': float(params['regressor__tol']),
            'solver': 'newton-cholesky'
        }
        
        params_preprocessor = {
            'n_bins_drivage': int(params['preprocessor__n_bins_drivage']),
            'n_bins_vehage': int(params['preprocessor__n_bins_vehage'])
        }
        # Parametry dla feature pipeline
        cap_values = {
            'VehPower': int(params['preprocessor__cap_veh_power']),
            'VehAge': int(params['preprocessor__cap_veh_age']),
            'DrivAge': int(params['preprocessor__cap_driv_age']),
            'BonusMalus': int(params['preprocessor__cap_bonus_malus']),
            'Density': int(params['preprocessor__cap_density'])
        }
        
        n_bins_drivage = params_preprocessor['n_bins_drivage']
        n_bins_vehage = params_preprocessor['n_bins_vehage']

        # Pipeline z preprocessorem i modelem
        model = Pipeline([
            ('preprocessor', create_feature_pipeline(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                n_bins_drivage=n_bins_drivage,
                n_bins_vehage=n_bins_vehage,
                cap_values=cap_values
            )),
            ('regressor', PoissonRegressor(**params_regressor))
        ])

        # Walidacja krzyżowa z podziałem na grupy
        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
        splits = list(gss.split(train, groups=train['GroupID']))
        
        def evaluate_fold(fold):
            train_idx, val_idx = splits[fold]
            X_train_cv = train.iloc[train_idx]
            X_val_cv = train.iloc[val_idx]
            
            y_train_cv = y_freq_train.iloc[train_idx]
            y_val_cv = y_freq_train.iloc[val_idx]
            
            weights_train_cv = weights_train.iloc[train_idx]
            weights_val_cv = weights_train.iloc[val_idx]

            fold_model = clone(model)
            
            fold_model.fit(
                X_train_cv,
                y_train_cv,
                regressor__sample_weight=weights_train_cv
            )

            # Predykcja z uwzględnieniem ekspozycji
            pred = fold_model.predict(X_val_cv)
            score = mean_poisson_deviance(y_val_cv, pred, sample_weight=weights_val_cv)
            return score
        
        scores = Parallel(n_jobs=-1)(delayed(evaluate_fold)(fold) for fold in range(len(splits)))
        
        return {'loss': np.mean(scores), 'status': STATUS_OK}

    # Optymalizacja
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        verbose=True,
        show_progressbar=True
    )
    
    return best


def apply_best_params_and_train_model(best, train, test, numerical_features, categorical_features, path_to_save):
    """
    Applies the best parameters from hyperparameter tuning to create and train a GLM model.
    
    Parameters:
    -----------
    best : dict
        Dictionary containing the best hyperparameters from optimization
    train : DataFrame
        Training dataset
    test : DataFrame
        Test dataset
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names
        
    Returns:
    --------
    tuple
        (trained_model, summary_tuple) where summary_tuple contains evaluation metrics and predictions
    """
    best_params_regressor = {
        'alpha': best['regressor__alpha'],
        'max_iter': int(best['regressor__max_iter']),
        'tol': float(best['regressor__tol']),
        'solver': 'newton-cholesky'
    }
    best_params_preprocessor = {
        'n_bins_drivage': int(best['preprocessor__n_bins_drivage']),
        'n_bins_vehage': int(best['preprocessor__n_bins_vehage']),
        'cap_values': {
            'VehPower': int(best['preprocessor__cap_veh_power']),
            'VehAge': int(best['preprocessor__cap_veh_age']),
            'DrivAge': int(best['preprocessor__cap_driv_age']),
            'BonusMalus': int(best['preprocessor__cap_bonus_malus']),
            'Density': int(best['preprocessor__cap_density'])
        }
    }

    final_model = Pipeline([
        ('preprocessor', create_feature_pipeline(numerical_features, categorical_features, **best_params_preprocessor)),
        ('regressor', PoissonRegressor(**best_params_regressor))
    ])

    pipeline_features = create_feature_pipeline(numerical_features, categorical_features, **best_params_preprocessor)
    model_class = PoissonRegressor

    model, summary_tuple = train_and_evaluate_model(train, test, pipeline_features, model_class, best_params_regressor)
    save_model_with_pipeline(model, path_to_save)
    print_summary(summary_tuple[0])
    
    return model, summary_tuple, pipeline_features


def tune_xgboost_model(train, test, pipeline_features, n_trials=5, random_state=42, early_stopping_rounds=30):
    """
    Tune XGBoost model with hyperparameter optimization.
    
    Parameters:
    -----------
    train : DataFrame
        Training dataset
    test : DataFrame
        Test dataset
    pipeline_features : Pipeline
        Preprocessor pipeline for feature transformation
    n_trials : int, default=5
        Number of trials for hyperparameter optimization
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Best hyperparameters from optimization
    """
    y_train, weights_train, y_freq_train, y_test, weights_test, y_freq_test = prepare_target_and_weights(train, test)

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'max_depth': hp.quniform('max_depth', 3, 9, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(1), np.log(100)),
        'gamma': hp.loguniform('gamma', np.log(0.01), np.log(10)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10)),
        'n_estimators': hp.quniform('n_estimators', 300, 500, 50)
    }

    def objective(params):
        params_xgb = {
            'objective': 'count:poisson',
            'eval_metric': 'poisson-nloglik',
            'learning_rate': float(params['learning_rate']),
            'max_depth': int(params['max_depth']),
            'subsample': float(params['subsample']),
            'colsample_bytree': float(params['colsample_bytree']),
            'min_child_weight': float(params['min_child_weight']),
            'gamma': float(params['gamma']),
            'reg_alpha': float(params['reg_alpha']),
            'reg_lambda': float(params['reg_lambda']),
            'tree_method': 'hist',
            'enable_categorical': True,
            'n_estimators': int(params['n_estimators']),
            'early_stopping_rounds': early_stopping_rounds,
            # 'eval_metric': custom_metric_weighted
        }
        
        model = Pipeline([
            ('preprocessor', pipeline_features),
            ('regressor', XGBRegressor(**params_xgb))
        ])

        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
        splits = list(gss.split(train, groups=train['GroupID']))
        
        def evaluate_fold(fold):
            train_idx, val_idx = splits[fold]
            X_train_cv = train.iloc[train_idx]
            X_val_cv = train.iloc[val_idx]
            
            y_train_cv = y_freq_train.iloc[train_idx]
            y_val_cv = y_freq_train.iloc[val_idx]
            
            weights_train_cv = weights_train.iloc[train_idx]
            weights_val_cv = weights_train.iloc[val_idx]

            fold_model = clone(model)
            
            fold_model.fit(
                X_train_cv,
                y_train_cv,
                regressor__eval_set=[(pipeline_features.transform(X_val_cv), y_val_cv)],
                regressor__sample_weight=weights_train_cv,
                regressor__verbose=100,
                regressor__sample_weight_eval_set=[weights_val_cv], # TODO: Check if this is correct. It needs to be applied in this way.
                # regressor__sample_weight_eval_set=weights_train_cv,
                # regressor__eval_metric=custom_poisson_deviance
            )

            # Predykcja z uwzględnieniem ekspozycji
            pred = fold_model.predict(X_val_cv)
            score = mean_poisson_deviance(y_val_cv, pred, sample_weight=weights_val_cv)
            return score

        scores = Parallel(n_jobs=-1)(delayed(evaluate_fold)(fold) for fold in range(len(splits)))
        
        return {'loss': np.mean(scores), 'status': STATUS_OK}

    # Optymalizacja
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        verbose=True,
        show_progressbar=True
    )
    
    return best


def apply_best_params_and_train_xgboost_model(best, train, test, pipeline_features, path_to_save, early_stopping_rounds=30):
    """
    Applies the best parameters from hyperparameter tuning to create and train an XGBoost model.
    
    Parameters:
    -----------
    best : dict
        Dictionary containing the best hyperparameters from optimization
    train : DataFrame
        Training dataset
    test : DataFrame
        Test dataset
    pipeline_features : Pipeline
        Preprocessor pipeline for feature transformation
    path_to_save : str
        Path to save the trained model
        
    Returns:
    --------
    tuple
        (trained_model, summary_tuple) where summary_tuple contains evaluation metrics and predictions
    """
    best_params_regressor = {
        'objective': 'count:poisson',
        'eval_metric': 'poisson-nloglik',
        'learning_rate': float(best['learning_rate']),
        'max_depth': int(best['max_depth']),
        'subsample': float(best['subsample']),
        'colsample_bytree': float(best['colsample_bytree']),
        'min_child_weight': float(best['min_child_weight']),
        'gamma': float(best['gamma']),
        'reg_alpha': float(best['reg_alpha']),
        'reg_lambda': float(best['reg_lambda']),
        'tree_method': 'hist',
        'enable_categorical': True,
        'n_estimators': int(best['n_estimators']),
        'early_stopping_rounds': early_stopping_rounds
    }

    final_model = Pipeline([
        ('preprocessor', pipeline_features),
        ('regressor', XGBRegressor(**best_params_regressor))
    ])

    model_class = XGBRegressor

    model, summary_tuple = train_and_evaluate_model(train, test, pipeline_features, model_class, best_params_regressor, verbose=50)
    save_model_with_pipeline(model, path_to_save)
    print_summary(summary_tuple[0])
    
    return model, summary_tuple