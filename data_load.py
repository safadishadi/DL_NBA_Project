from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, playergamelog
from nba_api.stats.static import teams, players
import pandas as pd
import time


#Fetch NBA Stats from NBA official site using NBA-API
def fetch_all_game_data():
    """
    Fetches all NBA game data from the API for all teams.
    Returns a DataFrame with historical game data.
    """
    nba_teams = teams.get_teams()
    all_games = []
    for team in nba_teams:
        team_id = team['id']
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        team_games = gamefinder.get_data_frames()[0]
        team_games['TEAM_ID'] = team_id
        all_games.append(team_games)
        time.sleep(1)  # Prevent rate limiting
    return pd.concat(all_games, ignore_index=True)

def fetch_team_stats():
    """
    Fetches team stats for all teams.
    Returns a DataFrame with team stats.
    """
    nba_teams = teams.get_teams()
    all_team_stats = []
    for team in nba_teams:
        team_id = team['id']
        stats = teamgamelog.TeamGameLog(team_id=team_id).get_data_frames()[0]
        stats['TEAM_ID'] = team_id
        all_team_stats.append(stats)
        time.sleep(1)
    return pd.concat(all_team_stats, ignore_index=True)