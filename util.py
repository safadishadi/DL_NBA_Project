from nba_api.stats.static import teams








def contains_alpha(col):
    if col.dtype == 'object':
        return col.str.contains(r'[a-zA-Z]', na=False).any()

    return False


# Add opponent team ID to each line of data df according to abbreviations
def add_opponent_team_id(data):
    nba_teams = teams.get_teams()
    team_abbr_to_id = {team['abbreviation']: int(team['id']) for team in nba_teams}
    data['OPPONENT_ABBR'] = data['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    data['OPPONENT_ABBR'] = data['OPPONENT_ABBR'].str.strip().str.upper()
    data['OPPONENT_TEAM_ID'] = data['OPPONENT_ABBR'].map(team_abbr_to_id).astype('Int64')
    data.drop(columns=['OPPONENT_ABBR'], inplace=True)

    return data


# Add relevant opponent stats to each line of data df
def add_opponent_stats(data, team_stats_copy):
    opponent_stats = team_stats_copy.copy()
    opponent_stats = opponent_stats.rename(columns={
        'TEAM_ID': 'OPPONENT_TEAM_ID',
        'W': 'OPPONENT_W',
        'L': 'OPPONENT_L',
        'W_PCT': 'OPPONENT_W_PCT',
        'Home': 'OPPONENT_Home',
        'Win_Pct_Last_10': 'OPPONENT_Win_Pct_Last_10',
        'Win_Streak': 'OPPONENT_Win_Streak',
        'TARGET': 'OPPONENT_TARGET'
    })
    data = data.merge(opponent_stats,
                      left_on=['Game_ID', 'OPPONENT_TEAM_ID'],
                      right_on=['Game_ID', 'OPPONENT_TEAM_ID'],
                      how='left')
    data.fillna(0, inplace=True)

    return data