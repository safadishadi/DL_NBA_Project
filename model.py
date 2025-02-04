from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.metrics import accuracy_score
import optuna


#Define the Objective Function
def objective(trial, data, X_train, X_val, y_val):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Data Configuration
    data_config = DataConfig(
        target=['TARGET'],
        continuous_cols=[col for col in data.columns if col != 'TARGET'],
        categorical_cols=[],
    )

    # Model Configuration
    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="-".join([str(hidden_dim)] * hidden_layers),
        activation="ReLU",
        dropout=dropout,
    )

    # Trainer Configuration
    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=batch_size,
        max_epochs=10,# feel free to change epochs num while training
    )

    # Optimizer Configuration
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
    )

    # Initialize the Tabular Model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    # Train the model
    tabular_model.fit(train=X_train, validation=X_val)

    # Predict and evaluate
    preds = tabular_model.predict(X_val)
    print(preds.columns)
    accuracy = accuracy_score(y_val, preds['TARGET_prediction'])

    return accuracy



def model_config(data1, X_train1, X_val1, y_val1):
    data = data1
    X_train = X_train1
    X_val = X_val1
    y_val = y_val1
    #Create Study and Optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data, X_train, X_val, y_val), n_trials=10) # feel free to change num of optuna trials

    #Train Final Model with Best Parameters
    best_params = study.best_params
    print("Best Parameters:", best_params)

    data_config = DataConfig(
            target=['TARGET'],
            continuous_cols=[col for col in data.columns if col != 'TARGET'],
            categorical_cols=[],
        )

    final_model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="-".join([str(best_params['hidden_dim'])] * best_params['hidden_layers']),
        activation="ReLU",
        dropout=best_params['dropout'],
    )

    final_trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=best_params['batch_size'],
        max_epochs=20, # feel free to change max epochs upon the need
    )

    final_optimizer_config = OptimizerConfig(
        optimizer="Adam",
        #learning_rate=best_params['learning_rate']
    )

    final_model = TabularModel(
        data_config=data_config,
        model_config=final_model_config,
        optimizer_config=final_optimizer_config,
        trainer_config=final_trainer_config
    )

    return final_model, study