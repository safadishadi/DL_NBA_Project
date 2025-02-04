# Install necessary libraries in bash before executing main
# pip install nba_api
# pip install optuna pytorch-tabular

import optuna
from data_load import fetch_all_game_data, fetch_team_stats
from data_process import prepare_data_for_transformer, preprocess_data
from sklearn.metrics import accuracy_score
from plotting import plotting5
from model import model_config



if __name__ == "__main__":
    #Data fetch
    print("Fetching game data...")
    games = fetch_all_game_data()

    print("Fetching team stats...")
    team_stats = fetch_team_stats()

    #preserve original data, work on copies
    games_copy = games.copy()
    team_stats_copy = team_stats.copy()
    player_stats_copy = 0 #player_stats.copy(), can uncomment and use for future work

    print("Preprocessing data...")
    data = preprocess_data(games_copy, team_stats_copy, player_stats_copy)

    print("Preparing data for Tabular Transformer...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_transformer(data)

    print("Data is ready!")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    final_model, study = model_config(data, X_train, X_val, y_val)

    final_model.fit(train=X_train, validation=X_val)
    final_preds = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_preds['TARGET_prediction'])
    print(f"Final Test Accuracy: {final_accuracy}")

    plotting5(final_model, X_test, y_test, study)