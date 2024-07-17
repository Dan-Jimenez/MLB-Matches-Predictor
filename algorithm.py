from bs4 import BeautifulSoup
import requests
import re
import db_functions as db_f
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from pytz import timezone, utc
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
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
sportsbooks = pd.read_sql_query('SELECT * FROM sportsbooks;', db_f.conn)

# DATA TRANSFORMING AND CLEANING
sportsbooks_away = sportsbooks.rename(columns=lambda x: x.replace('away', 'team').replace('home', 'opponent')).rename(columns={'team_team': 'team'}).drop(columns=['team_points', 'opponent_points', 'book_id', 'game_id', 'opponent_team', 'opponent_spread'])
sportsbooks_home = sportsbooks.rename(columns=lambda x: x.replace('home', 'team').replace('away', 'opponent')).rename(columns={'team_team': 'team'}).drop(columns=['team_points', 'opponent_points', 'book_id', 'game_id', 'opponent_team', 'opponent_spread'])
sportsbooks = pd.concat([sportsbooks_away, sportsbooks_home], ignore_index=True)
sportsbooks = sportsbooks.sort_values('game_date').reset_index(drop=True)
sportsbooks['game_date'] = pd.to_datetime(sportsbooks['game_date']).dt.date

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
    team['target'] = team['team_score'].shift(-1)
    team['winner_next'] = team['winner'].shift(-1)
    return team
matches = matches.groupby('team', group_keys=False).apply(add_target)
#teams_stats = teams_stats[teams_stats['competition'] == 'MLB']
teams_stats = teams_stats[teams_stats['season'] == '2024']
teams_stats.drop(columns=['id', 'competition', 'season', 'side', 'link_id', 'lob'], inplace=True)

stats = pd.merge(matches, teams_stats, on=['game_date', 'team', 'opponent_team'], how='inner')
stats = stats.sort_values('game_date').reset_index(drop=True)
stats = stats[['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'side', 'team_score', 'opponent_score', 'runs', 'doubles', 'triples',
                               'hr', 'so', 'bb', 'hits', 'avg', 'obp', 'slg', 'ops', 'rbi', 'ab', 'era', 'link_id', 'winner', 'winner_next', 'target']]
stats.loc[pd.isnull(stats['target']), 'target'] = 50
stats.loc[pd.isnull(stats['winner_next']), 'winner_next'] = 2
stats['target'] = stats['target'].astype(int, errors='ignore')
stats['winner'] = stats['winner'].astype(int, errors='ignore')
stats['winner_next'] = stats['winner_next'].astype(int, errors='ignore')

no_scale_cols = ['id', 'game_date', 'competition', 'season', 'team', 'opponent_team', 'link_id', 'winner', 'winner_next','target']
scale_cols = stats.columns[~stats.columns.isin(no_scale_cols)]
scaler = MinMaxScaler()
stats[scale_cols] = scaler.fit_transform(stats[scale_cols])

# ALGORITHM 
""" def backtest(data, model, predictors, start=0, step=1):
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

    return pd.concat(all_predictions) """
def backtest(data, model, predictors, start=0, step=1):
    all_predictions = []
    if 'game_date_next' in data.columns:
        data['game_date_next'] = pd.to_datetime(data['game_date_next'])  # Convert 'game_date_next' to datetime if it's not already
    games_dates = sorted(data['game_date_next'].dt.date.unique())  # Extract unique dates and convert to date objects
    for game_date in games_dates:
        test = data[data['game_date_next'].dt.date == game_date]  # Filter data for the current date
        if not test.empty and not test['target'].empty:
            actual_scores = test['target']
            train = data[data['game_date_next'].dt.date < game_date]  # Filter training data for dates before current date
            if not train.empty:  # Check if training data is not empty
                model.fit(train[predictors], train['target'])
                preds = model.predict(test[predictors])
                preds = pd.Series(preds, index=test.index)
                combined = pd.concat([test['target'], preds], axis=1)
                combined.columns = ['actual_scores', 'prediction_scores']

                all_predictions.append(combined)
            #else:
                #print(f"No training data available for {game_date}. Skipping prediction.")

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

def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col
def add_col(df, col_name):
    return df.groupby('team', group_keys=False).apply(lambda x: shift_col(x, col_name))
stats['game_date_next'] = add_col(stats, 'game_date')
stats['side_next'] = add_col(stats, 'side')
stats['opponent_team_next'] = add_col(stats, 'opponent_team')

next_matches['game_date'] = pd.to_datetime(next_matches['game_date'])
next_matches = next_matches[next_matches['game_date'].dt.date == pd.to_datetime(pd_selected_date).date()]
away_df = next_matches.rename(columns=lambda x: x.replace('away', 'team').replace('home', 'opponent')).rename(columns={'team_team': 'team'})
away_df['side'] = 0
home_df = next_matches.rename(columns=lambda x: x.replace('home', 'team').replace('away', 'opponent')).rename(columns={'team_team': 'team'})
home_df['side'] = 1
home_df = home_df[['game_date', 'team', 'opponent_team', 'side']]
away_df = away_df[['game_date', 'team', 'opponent_team', 'side']]
next_matches = pd.concat([away_df, home_df], ignore_index=True).sort_values('game_date').reset_index(drop=True)
for index, row in next_matches.iterrows():
    condition = (stats['team'] == row['team'])
    last_matching_row = stats[condition].iloc[-1]
    stats.loc[last_matching_row.name, 'side_next'] = row['side']
    stats.loc[last_matching_row.name, 'opponent_team_next'] = row['opponent_team']
    stats.loc[last_matching_row.name, 'game_date_next'] = row['game_date']

""" players_stats = players_stats.sort_values('game_date').reset_index(drop=True)
players_stats = players_stats[(players_stats['substitute'] == False) & (players_stats['season'] == '2024')]
players_stats.drop(columns=['id', 'competition', 'season', 'side', 'jersey_number', 'player_id', 'position', 'substitute'], inplace=True)
#players_stats = players_stats.loc[:, ['game_date', 'team', 'opponent_team', 'link_id', 'player_id']]
players_stats.rename(columns={'name': 'pitcher'}, inplace=True)
players_no_scale = ['game_date', 'team', 'opponent_team', 'link_id', 'pitcher']
players_scale_cols = players_stats.columns[~players_stats.columns.isin(players_no_scale)]
scaler = MinMaxScaler()
players_stats[players_scale_cols] = scaler.fit_transform(players_stats[players_scale_cols])
#stats = pd.merge(players_stats, stats, on=['game_date', 'team', 'opponent_team', 'link_id'], how='inner', suffixes=('_p', ''))
players_last_3 = players_stats[list(players_scale_cols) + ['pitcher']]
rol = 2
def find_team_averages(pitcher):
    rolling = pitcher[players_scale_cols].rolling(rol).mean()
    return rolling
players_last_3 = players_last_3.groupby('pitcher', group_keys=False).apply(find_team_averages)
players_last_3_cols = [f'{col}_l3_p' for col in players_last_3.columns]
players_last_3.columns = players_last_3_cols
players_stats = pd.concat([players_stats, players_last_3], axis=1)
#nan_percentage = (players_stats.isna().mean() * 100).round(2)
#print(nan_percentage)
#print(players_stats.columns)
def shift_col(pitcher, col_name):
    next_col = pitcher[col_name].shift(-1)
    return next_col
def add_col(df, col_name):
    return df.groupby('pitcher', group_keys=False).apply(lambda x: shift_col(x, col_name))
players_stats['game_date_next'] = add_col(players_stats, 'game_date')
players_stats['opponent_team_next'] = add_col(players_stats, 'opponent_team')
players_stats.drop(columns=['game_date', 'opponent_team', 'link_id'], inplace=True)
stats = pd.merge(stats, players_stats, on=['team', 'game_date_next', 'opponent_team_next'], how='inner', suffixes=['','_p'])
#print(stats[['game_date_next', 'team', 'opponent_team_next', 'pitcher']].tail(60))
#print(stats[['game_date_next', 'team', 'pitcher', 'opponent_team', 'era_l3_p']])
stats = stats.dropna() """

stats = stats.merge(stats[last_10_cols + ['game_date_next', 'team', 'opponent_team_next']], left_on=['game_date_next', 'team'], right_on=['game_date_next', 'opponent_team_next'], suffixes=('_x', '_z'))
stats = stats.merge(stats[last_5_cols + ['game_date_next', 'team_x', 'opponent_team_next_x']], left_on=['game_date_next', 'team_x'], right_on=['game_date_next', 'opponent_team_next_x'], suffixes=('_x', '_z'))

today_games = stats[(pd.to_datetime(stats['game_date_next']).dt.date == selected_date)][['team_x_x', 'opponent_team_next_x_x', 'game_date_next', 'target']]
print(today_games)
today_games_index = today_games.index

i = 1
j = 7
n = 26
rr = Ridge(alpha=i)
split = TimeSeriesSplit(n_splits = j)
sfs = SequentialFeatureSelector(rr, n_features_to_select=n, direction='forward', cv=split)

stats = stats.dropna()
training_data = stats[stats['target'] != 50]
removed_cols = list(training_data.columns[training_data.dtypes == 'object']) + no_scale_cols
selected_cols = training_data.columns[~training_data.columns.isin(removed_cols)]
sfs.fit(training_data[selected_cols], training_data['target'])
predictors = list(selected_cols[sfs.get_support()])
predictions = backtest(stats, rr, predictors)
filtered_predictions = predictions[predictions.index.isin(today_games_index)]
rounded_predictions = filtered_predictions['prediction_scores'].round(2)
print()
print(rounded_predictions)

predictions = predictions[predictions['actual_scores'] != 50]
predictions_df = pd.merge(stats, predictions, left_index=True, right_index=True)
final_df = pd.merge(predictions_df, predictions_df[['game_date_next', 'team_x_x', 'actual_scores', 'prediction_scores']], left_on=['game_date_next', 'opponent_team_next_x_x'], right_on=['game_date_next', 'team_x_x'], suffixes=('_x', '_opponent'))
final_df['prediction_winner_next'] = final_df.apply(lambda row: 1 if row['prediction_scores_x'] > row['prediction_scores_opponent'] else 0, axis=1)
final_df['game_date_next'] = pd.to_datetime(final_df['game_date_next']).dt.date
final_df = final_df.merge(sportsbooks, left_on=['game_date_next', 'team_x_x_x'], right_on=['game_date', 'team']).drop(columns=['team']).rename(columns={'team_spread': 'team_spread_next', 'total': 'total_next'}).sort_values(by='game_date_next')
final_df['actual_total_next'] = final_df['actual_scores_x'] + final_df['actual_scores_opponent']
final_df['prediction_total_next'] = final_df['prediction_scores_x'] + final_df['prediction_scores_opponent']
final_df['actual_total_next_result'] = np.where((final_df['actual_scores_x'] + final_df['actual_scores_opponent']) > final_df['total_next'], 1, 0)
final_df['prediction_total_next_result'] = np.where(final_df['prediction_total_next'] > final_df['total_next'], 1, 0)
final_df = final_df.dropna()

# Print classification report for winner prediction
print(i, j, n)
print('Classification Report for Winner Prediction:')
print(classification_report(final_df['winner_next'], final_df['prediction_winner_next']))

# Print classification report for total result prediction
print('Classification Report for Total Prediction:')
print(classification_report(final_df['actual_total_next_result'], final_df['prediction_total_next_result']))

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def calculate_classification_report(group):
    report = classification_report(group['winner_next'], group['prediction_winner_next'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    return report_df
# Calculate and print classification report for winner prediction
team_classification_report_winner = final_df.groupby('team_x_x_x').apply(calculate_classification_report)
print('\nTeam Classification Report for Winner Prediction:')
print(team_classification_report_winner)

""" def calculate_classification_report(group):
    report = classification_report(group['actual_total_next_result'], group['prediction_total_next_result'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    return report_df
# Calculate and print classification report for total result prediction
print('\nTeam Classification Report for Total Prediction:')
team_classification_report_total = final_df.groupby('team_x_x_x').apply(calculate_classification_report)
print(team_classification_report_total) """

# Reset display options if needed
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')




#team_accuracies = stats.groupby('team_x')['actual', 'prediction'].apply(lambda x: accuracy_score(x['actual'], x['prediction']))
#print(team_accuracies)


# Code for best params and model
""" stats = stats.dropna()
# Fit SequentialFeatureSelector
removed_cols = list(stats.columns[stats.dtypes == 'object']) + no_scale_cols
selected_cols = stats.columns[~stats.columns.isin(removed_cols)]
sfs.fit(stats[selected_cols], stats['target'])
predictors = list(selected_cols[sfs.get_support()])

# GridSearchCV for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1, 10]  # Example values for alpha parameter in Ridge regression
}
grid_search = GridSearchCV(rr, param_grid, cv=split, scoring='accuracy')
grid_search.fit(stats[selected_cols], stats['target'])
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)
best_model = grid_search.best_estimator_
print(best_model)
# Use the best model to make predictions
predictions = backtest(stats, best_model, predictors) """
