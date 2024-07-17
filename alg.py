import db_functions as db_f
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from pytz import timezone, utc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

mexico_timezone = timezone('America/Mexico_City')
current_date = datetime.now(mexico_timezone).date()
previous_date = current_date - timedelta(days=30)
selected_date = current_date
pd_selected_date = pd.Timestamp(selected_date)

# Read data from each table
matches = pd.read_sql_query('SELECT * FROM matches;', db_f.conn)
next_matches = pd.read_sql_query('SELECT * FROM next_matches;', db_f.conn)
teams_stats = pd.read_sql_query('SELECT * FROM teams_games_stats;', db_f.conn)
players_stats = pd.read_sql_query('SELECT * FROM pitchers_games_stats;', db_f.conn)

# Create DataFrames for away and home teams renaming teams columns and adding the side
away_df = matches.rename(columns=lambda x: x.replace('away', 'team').replace('home', 'opponent')).rename(columns={'team_team': 'team'})
away_df['winner'] = away_df['winner'].map({'home': False, 'away': True})
away_df['side'] = 0 # 0 = Away
home_df = matches.rename(columns=lambda x: x.replace('home', 'team').replace('away', 'opponent')).rename(columns={'team_team': 'team'})
home_df['winner'] = home_df['winner'].map({'home': True, 'away': False})
home_df['side'] = 1 # 1 = Home
home_df = home_df[['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'team_score', 'opponent_score', 'link_id', 'side', 'winner',]]
away_df = away_df[['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'team_score', 'opponent_score', 'link_id', 'side', 'winner',]]
matches = pd.concat([away_df, home_df], ignore_index=True)
matches = matches.sort_values('game_date').reset_index(drop=True)

def add_target(team):
    team['target'] = team['winner'].shift(-1)
    return team
matches = matches.groupby('team', group_keys=False).apply(add_target)
teams_stats = teams_stats[teams_stats['season'] == '2024']
teams_stats.drop(columns=['id', 'competition', 'season', 'side', 'link_id'], inplace=True)

stats = pd.merge(matches, teams_stats, on=['game_date', 'team', 'opponent_team'], how='inner')
stats = stats[['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'side', 'team_score', 'opponent_score', 'runs', 'doubles', 'triples',
                               'hr', 'so', 'bb', 'hits', 'avg', 'obp', 'slg', 'ops', 'rbi', 'ab', 'lob', 'era', 'link_id', 'winner', 'target']]
stats.loc[pd.isnull(stats['target']), 'target'] = 2
stats['target'] = stats['target'].astype(int, errors='ignore')
stats['winner'] = stats['winner'].astype(int, errors='ignore')

rr = RidgeClassifier(alpha=10)
split = TimeSeriesSplit(n_splits = 4)
sfs = SequentialFeatureSelector(rr, n_features_to_select=23, direction='forward', cv=split)

no_scale_cols = ['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'link_id', 'winner', 'target']
scale_cols = stats.columns[~stats.columns.isin(no_scale_cols)]
scaler = MinMaxScaler()
stats[scale_cols] = scaler.fit_transform(stats[scale_cols])

# ALGORITHM 
def backtest(data, model, predictors, start=0, step=1):
    all_predictions = []
    games_dates = sorted(data['game_date'].unique())
    #print(games_dates)
    for i in range(start, len(games_dates), step):
        game_date = games_dates[i]
        train = data[data['game_date'] < game_date]
        test = data[data['game_date'] == game_date]

        if not train.empty and not train['target'].empty:
            model.fit(train[predictors], train['target'])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index)
            combined = pd.concat([test['target'], preds], axis=1)
            combined.columns = ['actual', 'prediction']

            all_predictions.append(combined)

    return pd.concat(all_predictions)


stats_last_10 = stats[list(scale_cols) + ['team', 'winner']]
def find_team_averages(team):
    rolling = team[scale_cols].rolling(10).mean()
    return rolling
stats_last_10 = stats_last_10.groupby('team', group_keys=False).apply(find_team_averages)
last_10_cols = [f'{col}_l10' for col in stats_last_10.columns]
stats_last_10.columns = last_10_cols
stats = pd.concat([stats, stats_last_10], axis=1)

stats_last_5 = stats[list(scale_cols) + ['team', 'winner']]
def find_team_averages(team):
    rolling = team[scale_cols].rolling(5).mean()
    return rolling
stats_last_5 = stats_last_5.groupby('team', group_keys=False).apply(find_team_averages)
last_5_cols = [f'{col}_l5' for col in stats_last_5.columns]
stats_last_5.columns = last_5_cols
stats = pd.concat([stats, stats_last_5], axis=1)

players_stats = players_stats[players_stats['substitute'] == False]
players_stats.drop(columns=['id', 'competition', 'season', 'side', 'jersey_number', 'player_id', 'position', 'substitute'], inplace=True)
players_stats.rename(columns={'name': 'pitcher'}, inplace=True)
#stats = pd.merge(players_stats, stats, on=['game_date', 'team', 'opponent_team', 'link_id'], how='inner', suffixes=('_p', ''))

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col
def add_col(df, col_name):
    return df.groupby('team', group_keys=False).apply(lambda x: shift_col(x, col_name))
stats['game_date_next'] = add_col(stats, 'game_date')
stats['side_next'] = add_col(stats, 'side')
stats['opponent_team_next'] = add_col(stats, 'opponent_team')
#stats['pitcher_next'] = add_col(stats, 'pitcher')

next_matches['game_date'] = pd.to_datetime(next_matches['game_date'])
next_matches = next_matches[next_matches['game_date'].dt.date == pd.to_datetime(pd_selected_date).date()]
away_df = next_matches.rename(columns=lambda x: x.replace('away', 'team').replace('home', 'opponent')).rename(columns={'team_team': 'team'})
away_df['side'] = 0
home_df = next_matches.rename(columns=lambda x: x.replace('home', 'team').replace('away', 'opponent')).rename(columns={'team_team': 'team'})
home_df['side'] = 1
home_df = home_df[['game_date', 'team', 'opponent_team', 'side']]
away_df = away_df[['game_date', 'team', 'opponent_team', 'side']]
next_matches = pd.concat([away_df, home_df], ignore_index=True)
next_matches = next_matches.sort_values('game_date').reset_index(drop=True)
for index, row in next_matches.iterrows():
    condition = (stats['team'] == row['team'])
    last_matching_row = stats[condition].iloc[-1]
    stats.loc[last_matching_row.name, 'side_next'] = row['side']
    stats.loc[last_matching_row.name, 'opponent_team_next'] = row['opponent_team']
    stats.loc[last_matching_row.name, 'game_date_next'] = row['game_date']
stats = stats.merge(stats[last_10_cols + ['game_date_next', 'team', 'opponent_team_next']], left_on=['game_date_next', 'team'], right_on=['game_date_next', 'opponent_team_next'], suffixes=('_x', '_z'))
stats = stats.merge(stats[last_5_cols + ['game_date_next', 'team_x', 'opponent_team_next_x']], left_on=['game_date_next', 'team_x'], right_on=['game_date_next', 'opponent_team_next_x'], suffixes=('_x', '_z'))

today_games = stats[(pd.to_datetime(stats['game_date_next']).dt.date == selected_date)][['game_date_next', 'team_x_x', 'opponent_team_next_x_x', 'target']]
print(today_games)
today_games_index = today_games.index

stats = stats.dropna()
removed_cols = list(stats.columns[stats.dtypes == 'object']) + no_scale_cols
selected_cols = stats.columns[~stats.columns.isin(removed_cols)]
sfs.fit(stats[selected_cols], stats['target'])
predictors = list(selected_cols[sfs.get_support()])
print(predictors)
predictions = backtest(stats, rr, predictors)
filtered_predictions = predictions[predictions.index.isin(today_games_index)]
print(filtered_predictions)

predictions = predictions[predictions['actual'] != 2]
print(predictions)
predictions_df = pd.merge(stats, predictions, left_index=True, right_index=True)
final_df = pd.merge(predictions_df, predictions_df[['game_date_next', 'team_x_x', 'actual', 'prediction']], left_on=['game_date_next', 'opponent_team_next_x_x'], right_on=['game_date_next', 'team_x_x'], suffixes=('_x', '_opponent'))
final_df = final_df.dropna()
#print(final_df)
model_accuracy = accuracy_score(predictions['actual'], predictions['prediction'])
print(model_accuracy)
#team_accuracies = stats.groupby('team_x')['actual', 'prediction'].apply(lambda x: accuracy_score(x['actual'], x['prediction']))
#print(team_accuracies)