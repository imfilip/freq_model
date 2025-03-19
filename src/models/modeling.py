from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from src.models.scoring import model_summary


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

    model.fit(
        train,
        y=y_freq_train,
        regressor__sample_weight=weights_train
    )

    summary = model_summary(model, train, y_freq_train, test, y_freq_test, weights_train, weights_test)
    
    return model, summary