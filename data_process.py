from util import add_opponent_stats, add_opponent_team_id, contains_alpha
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split





# Main preprocessing function
def preprocess_data(games_copy, team_stats_copy, player_stats):
    #remove in-game stats, leave all pre-game stats
    to_drop_g_col = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
                    'TOV', 'PF', 'PLUS_MINUS']
    games_copy.drop(columns=[col for col in to_drop_g_col if col in games_copy.columns], inplace=True)
    to_drop_t_col = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
                    'BLK', 'TOV', 'PF', 'PTS']
    team_stats_copy.drop(columns=[col for col in to_drop_t_col if col in team_stats_copy.columns], inplace=True)

    data = games_copy.merge(team_stats_copy, left_on=['GAME_ID','TEAM_ID'], right_on=['Game_ID', 'Team_ID'], how='left', suffixes=('', '_drop'))
    data = data[[c for c in data.columns if not c.endswith('_drop')]]

    data = add_opponent_team_id(data)
    data = add_opponent_stats(data, team_stats_copy)

    #create new features
    data['TARGET'] = (data['WL_x'] == 'W').astype(int)
    data['Home'] = data['MATCHUP_x'].apply(lambda x: 1 if '@' in x else 0)
    data['Win_Pct_Last_10'] = data.groupby('TEAM_ID')['W'].rolling(window=10).mean().reset_index(drop=True)
    data['Win_Streak'] = data.groupby('TEAM_ID')['W'].rolling(window=5).sum().reset_index(drop=True)

    #drop unnecessary columns
    columns_to_drop = ['GAME_DATE', 'GAME_DATE_x', 'MATCHUP', 'GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
                       'TEAM_ID_TEAM', 'Team_ID', 'WL', 'Team_ID_x',  'Team_ID_y']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    columns_to_drop2 = [col for col in data.columns if contains_alpha(data[col])]

    if columns_to_drop2:
        data.drop(columns=[col for col in columns_to_drop2 if col in data.columns], inplace=True)

    #remove irrelevant lines with NaN values
    data.fillna(0, inplace=True)

    return data


#Prepare Data for Tabular Transformer
def prepare_data_for_transformer(data):
    """
    Splits the data into train, validation, and test sets and scales numeric features.
    """
    # Split data into features and target
    X = data
    y = data['TARGET']

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Define features for Standard and MinMax normalization
    standard_features = ['W', 'L', 'OPPONENT_W', 'OPPONENT_L', 'W_PCT', 'Win_Pct_Last_10', 'Win_Streak']
    minmax_features = ['SEASON_ID', 'TEAM_ID', 'Game_ID', 'OPPONENT_TEAM_ID', 'OPPONENT_W_PCT', 'Home']

    # Standard Scaling
    existing_standard_columns =[col for col in standard_features if col in X_train]
    standard_scaler = StandardScaler()
    X_train[existing_standard_columns] = standard_scaler.fit_transform(X_train[existing_standard_columns])
    X_val[existing_standard_columns] = standard_scaler.transform(X_val[existing_standard_columns])
    X_test[existing_standard_columns] = standard_scaler.transform(X_test[existing_standard_columns])

    # MinMax Scaling
    existing_minmax_columns = [col for col in minmax_features if col in X_train]
    minmax_scaler = MinMaxScaler()
    X_train[existing_minmax_columns] = minmax_scaler.fit_transform(X_train[existing_minmax_columns])
    X_val[existing_minmax_columns] = minmax_scaler.transform(X_val[existing_minmax_columns])
    X_test[existing_minmax_columns] = minmax_scaler.transform(X_test[existing_minmax_columns])

    return X_train, X_val, X_test, y_train, y_val, y_test



