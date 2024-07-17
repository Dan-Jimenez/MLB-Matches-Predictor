
import requests
import db_functions as db_f
from datetime import datetime, timedelta
from pytz import timezone, utc
import csv
import os

mexico_timezone = timezone('America/Mexico_City')
current_date = datetime.now(mexico_timezone).date()

headers = {
    'authority': 'api.sofascore.com',
    'accept': '/',
    'accept-language': 'en-US,en;q=0.9,es-US;q=0.8,es;q=0.7',
    'cache-control': 'max-age=0',
    'if-none-match': 'W/"3f3cf2c31d"',
    'origin': 'https://www.sofascore.com',
    'referer': 'https://www.sofascore.com/',
    'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36',
}
headers['If-Modified-Since'] = 'sat, 10 Jun 2024 00:00:00 GMT'

def save_mlb_next_matches(game_number):
    mlb_next_matches = []
    table_name = 'next_matches'
    response_mlb = requests.get(f'https://www.sofascore.com/api/v1/unique-tournament/11205/season/57577/events/next/{game_number}', headers=headers)
    matches = response_mlb.json()
    if not 'error' in matches:
        for item in matches['events']:
            link_id = item['id']
            link = f"https://www.sofascore.com/{item['slug']}/{item['customId']}#id:{link_id}"
            timestamp = item['startTimestamp']
            mexico_date = datetime.fromtimestamp(timestamp)
            #mexico_date = utc_date.replace(tzinfo=utc).astimezone(mexico_timezone)
            game_date = mexico_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3]
            if not db_f.avoid_id_matches_duplicates(table_name, link_id):
                mlb_next_matches.append([
                    game_date,
                    item['tournament']['name'],
                    item['season']['year'],
                    item['awayTeam']['name'],
                    item['homeTeam']['name'],
                    link,
                    link_id
                ])
    columns = ['game_date', 'competition', 'season', 'away_team', 'home_team', 'link', 'link_id']
    if mlb_next_matches:
        db_f.insert_data(table_name, columns, mlb_next_matches)

def save_mlb_matches(game_number):
    current_datetime_trunc = datetime.now(mexico_timezone).date()
    mlb_links = []
    table_name = 'matches'
    response_mlb = requests.get(f'https://www.sofascore.com/api/v1/unique-tournament/11205/season/57577/events/last/{game_number}', headers=headers)
    matches = response_mlb.json()
    if not 'error' in matches:
        for item in matches['events']:
            link_id = item['id']
            link = f"https://www.sofascore.com/{item['slug']}/{item['customId']}#id:{link_id}"
            away_score = item.get('awayScore', {}).get('current')
            home_score = item.get('homeScore', {}).get('current')
            if away_score is not None and home_score is not None:
                if away_score > home_score:
                    winner = 'away'
                elif away_score < home_score:
                    winner = 'home'
            timestamp = item['startTimestamp']
            mexico_date = datetime.fromtimestamp(timestamp)
            #mexico_date = utc_date.replace(tzinfo=utc).astimezone(mexico_timezone)
            game_date = mexico_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3]
            game_date_trunc = datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%S.%f').date()
            if not db_f.avoid_id_matches_duplicates(table_name, link_id) and game_date_trunc != current_datetime_trunc and away_score is not None and home_score is not None:
                mlb_links.append([
                    game_date,
                    item['tournament']['name'],
                    item['season']['year'],
                    item['awayTeam']['name'],
                    item['homeTeam']['name'],
                    away_score,
                    home_score,
                    winner,
                    link,
                    link_id
                ])
    columns = ['game_date', 'competition', 'season', 'away_team', 'home_team', 'away_score', 'home_score', 'winner', 'link', 'link_id']
    if mlb_links:
        db_f.insert_data(table_name, columns, mlb_links)



def get_teams_game_stats(statistics, game_date, competition, season, away_team, home_team, link_id):
    away_team_stats = []
    home_team_stats = []

    print(game_date, away_team, home_team, link_id)
    if not 'error' in statistics:
        for period_data in statistics.get('statistics', []):
            if period_data.get('period') == 'ALL':
                for group_data in period_data.get('groups', []):
                    if group_data.get('groupName') == 'Batting':
                        for item_data in group_data.get('statisticsItems', []):
                            stat_name = item_data.get('name', None)
                            away_value = item_data.get('awayValue', None)
                            home_value = item_data.get('homeValue', None)
                            if stat_name == 'Runs':
                                away_runs = away_value
                                home_runs = home_value
                            elif stat_name == 'Doubles':
                                away_doubles = away_value
                                home_doubles = home_value
                            elif stat_name == 'Triples':
                                away_triples = away_value
                                home_triples = home_value
                            elif stat_name == 'Home runs':
                                away_hr = away_value
                                home_hr = home_value
                            elif stat_name == 'Strike outs':
                                away_so = away_value
                                home_so = home_value
                            elif stat_name == 'Base on balls':
                                away_bb = away_value
                                home_bb = home_value
                            elif stat_name == 'Hits':
                                away_hits = away_value
                                home_hits = home_value
                            elif stat_name == 'AVG':
                                away_avg = away_value
                                home_avg = home_value
                            elif stat_name == 'OBP':
                                away_obp = away_value
                                home_obp = home_value
                            elif stat_name == 'SLG':
                                away_slg = away_value
                                home_slg = home_value
                            elif stat_name == 'OPS':
                                away_ops = away_value
                                home_ops = home_value
                            elif stat_name == 'RBI':
                                away_rbi = away_value
                                home_rbi = home_value
                            elif stat_name == 'At bats':
                                away_ab = away_value
                                home_ab = home_value
                            elif stat_name == 'Left on base':
                                away_lob = away_value
                                home_lob = home_value
                    if group_data.get('groupName') == 'Pitching':
                        for item_data in group_data.get('statisticsItems', []):
                            stat_name = item_data.get('name', None)
                            away_value = item_data.get('awayValue', None)
                            home_value = item_data.get('homeValue', None)
                            if stat_name == 'ERA':
                                away_era = away_value
                                home_era = home_value


        # Assuming you have initialized the lists before this code
        away_team_stats.append([game_date, competition, season, 'away', away_team, home_team, away_runs, away_doubles, away_triples, away_hr, away_so,
                                away_bb, away_hits, away_avg, away_obp, away_slg, away_ops, away_rbi, away_ab, away_lob, away_era, link_id])

        home_team_stats.append([game_date, competition, season, 'home', home_team, away_team, home_runs, home_doubles, home_triples, home_hr, home_so,
                                home_bb, home_hits, home_avg, home_obp, home_slg, home_ops, home_rbi, home_ab, home_lob, home_era, link_id])
    
    if away_team_stats and home_team_stats:
        return away_team_stats, home_team_stats
    else:
        return [], []

def save_teams_games_stats():
    table_name = 'matches'
    columns_name = 'game_date, competition, season, away_team, home_team, link_id'
    limit_number = 750
    _45_days_ago_date = current_date - timedelta(days=45)
    recent_games_data = db_f.get_column_data_limit(table_name, columns_name, _45_days_ago_date ,limit_number)
    #recent_games_data = db_f.get_column_data(table_name, columns_name)

    table = 'teams_games_stats'
    for date, competition, season, away_team, home_team, link_id in recent_games_data:
        if not db_f.avoid_players_stats_duplicates(table, link_id):
            response = requests.get(f'https://www.sofascore.com/api/v1/event/{link_id}/statistics', headers=headers)
            statistics = response.json()
            if not "error" in statistics:
                away_team_stats, home_team_stats = get_teams_game_stats(statistics, date, competition, season, away_team, home_team, link_id)

                if away_team_stats and home_team_stats:

                    columns = ['game_date', 'competition', 'season', 'side', 'team', 'opponent_team', 'runs', 'doubles', 'triples',
                               'hr', 'so', 'bb', 'hits', 'avg', 'obp', 'slg', 'ops', 'rbi', 'ab', 'lob', 'era', 'link_id']
                    
                    if away_team_stats:
                        db_f.insert_data(table, columns, away_team_stats)

                    if home_team_stats:
                        db_f.insert_data(table, columns, home_team_stats)



def get_players_game_stats(lineups, game_date=None, competition=None, season=None, away_team=None, home_team=None, link_id=None):
    away_players_stats = []
    home_players_stats = []

    if not 'error' in lineups:
        for player in lineups['away']['players']:
            position = player['player'].get('position', None)
            if position == 'P':
                player_id = player['player'].get('id', None)
                name = player['player'].get('name', None)
                jersey_number = player['player'].get('jerseyNumber', None)
                substitute = player.get('substitute', None)
                statistics = player.get('statistics', {})
                runs = statistics.get('pitchingRuns', None)
                hr = statistics.get('pitchingHomeRuns', None)
                so = statistics.get('pitchingStrikeOuts', None)
                bb = statistics.get('pitchingBaseOnBalls', None)
                hits = statistics.get('pitchingHits', None)
                ip = statistics.get('pitchingInningsPitched', None)
                er = statistics.get('pitchingEarnedRuns', None)
                bf = statistics.get('pitchingBattersFaced', None)
                outs = statistics.get('pitchingOuts', None)
                pitches = statistics.get('pitchingPitchesThrown', None)
                strikes = statistics.get('pitchingStrikes', None)
                era = statistics.get('pitchingEarnedRunsAverage', None)
            
                away_players_stats.append([game_date, competition, season, 'away', away_team, home_team, player_id, name, position,
                                           jersey_number, substitute, runs, hr, so, bb, hits, ip, er, bf, outs,
                                           pitches, strikes, era, link_id])

    if not 'error' in lineups:
        for player in lineups['home']['players']:
            position = player['player'].get('position', None)
            if position == 'P':
                player_id = player['player'].get('id', None)
                name = player['player'].get('name', None)
                jersey_number = player['player'].get('jerseyNumber', None)
                substitute = player.get('substitute', None)
                statistics = player.get('statistics', {})
                runs = statistics.get('pitchingRuns', None)
                hr = statistics.get('pitchingHomeRuns', None)
                so = statistics.get('pitchingStrikeOuts', None)
                bb = statistics.get('pitchingBaseOnBalls', None)
                hits = statistics.get('pitchingHits', None)
                ip = statistics.get('pitchingInningsPitched', None)
                er = statistics.get('pitchingEarnedRuns', None)
                bf = statistics.get('pitchingBattersFaced', None)
                outs = statistics.get('pitchingOuts', None)
                pitches = statistics.get('pitchingPitchesThrown', None)
                strikes = statistics.get('pitchingStrikes', None)
                era = statistics.get('pitchingEarnedRunsAverage', None)
            
                home_players_stats.append([game_date, competition, season, 'home', home_team, away_team, player_id, name, position,
                                           jersey_number, substitute, runs, hr, so, bb, hits, ip, er, bf, outs,
                                           pitches, strikes, era, link_id])

    if away_players_stats and home_players_stats:
        return away_players_stats, home_players_stats
    else:
        return [], []

def save_players_stats():
    table_name = 'matches'
    columns_name = 'game_date, competition, season, away_team, home_team, link_id'
    limit_number = 750
    _45_days_ago_date = current_date - timedelta(days=45)
    recent_games_data = db_f.get_column_data_limit(table_name, columns_name, _45_days_ago_date ,limit_number)
    #recent_games_data = db_f.get_column_data(table_name, columns_name)

    table = 'pitchers_games_stats'
    for date, competition, season, away_team, home_team, link_id in recent_games_data:
        if not db_f.avoid_players_stats_duplicates(table, link_id):
            response = requests.get(f'https://www.sofascore.com/api/v1/event/{link_id}/lineups', headers=headers)
            lineups = response.json()
            if not "error" in lineups:
                away_players_stats, home_players_stats = get_players_game_stats(lineups, date, competition, season, away_team, home_team, link_id)

                if away_players_stats and home_players_stats:
                    columns = ['game_date', 'competition', 'season', 'side', 'team', 'opponent_team', 'player_id', 'name',
                               'position', 'jersey_number', 'substitute', 'runs', 'hr', 'so', 'bb', 'hits', 'ip', 'er', 'bf',
                               'outs', 'pitches', 'strikes', 'era', 'link_id']
                    
                    if away_players_stats:
                        db_f.insert_data(table, columns, away_players_stats)
                        
                    if home_players_stats:
                        db_f.insert_data(table, columns, home_players_stats)



def save_team_info():
    table_name = 'teams'
    mlb_teams = []
    response_mlb = requests.get(f'https://api.sofascore.com/api/v1/unique-tournament/132/season/54105/teams', headers=headers)
    teams = response_mlb.json()
    if not 'error' in teams:
        for item in teams['teams']:
            mlb_teams.append([
                    item['name'],
                    item['nameCode'],
                    item['id'],
                ])
    columns = ['team', 'abbr', 'team_id']
    if mlb_teams:
        db_f.insert_data(table_name, columns, mlb_teams)

def save_players_info():
    table_name = 'players'
    mlb_players = []
    teams_id = db_f.get_teams_id()
    for id in teams_id:
        response = requests.get(f'https://api.sofascore.com/api/v1/team/{id[0]}/players', headers=headers)
        players = response.json()
        if 'error' not in players:
            for player_info in players['players']:
                player_id = player_info.get('player', {}).get('id')
                if not db_f.avoid_players_duplicates(table_name, player_id):
                    item = player_info['player']
                    mlb_players.append([
                        item.get('name'),
                        item.get('team', {}).get('name'),
                        item.get('team', {}).get('nameCode'),
                        item.get('position'),
                        item.get('jerseyNumber'),
                        item.get('country', {}).get('name'),
                        item.get('height'),
                        item.get('dateOfBirthTimestamp'),
                        player_id
                    ])
    columns = ['name', 'team', 'team_abbr', 'position', 'jersey_number', 'country', 'height', 'birth_date', 'player_id']
    if mlb_players:
        db_f.insert_data(table_name, columns, mlb_players)

def save_sportsbooks_data(formatted_date, date):
    table_name = 'sportsbooks'
    sportsbooks = []
    response = requests.get(f'https://api.actionnetwork.com/web/v2/scoreboard/mlb?bookIds=15,30,75,123,69,68,972,71,247,79&date={formatted_date}&periods=event', headers=headers)
    sportsbooks_data = response.json()
    if 'error' not in sportsbooks_data:
        for game in sportsbooks_data.get('games'):
            game_id = game.get('id')
            if not db_f.avoid_sportsbooks_duplicates(table_name, game_id):
                book_id = 79
                spread_values = {'home': None, 'away': None}
                total_values = {'under': None, 'over': None}

                for outcome in game['markets'].get(str(book_id), {}).get('event', {}).get('spread', []) + game['markets'].get(str(book_id), {}).get('event', {}).get('total', []):
                    if outcome.get('book_id') == book_id:
                        if outcome['type'] == 'spread':
                            spread_values[outcome.get('side')] = outcome.get('value')
                        elif outcome['type'] == 'total':
                            total_values[outcome.get('side')] = outcome.get('value')

                away_team = next((team for team in game.get('teams', []) if team.get('id') == game.get('away_team_id')), None)
                away_team_full_name = away_team.get('full_name') if away_team else None
                home_team = next((team for team in game.get('teams', []) if team.get('id') == game.get('home_team_id')), None)
                home_team_full_name = home_team.get('full_name') if home_team else None
                
                sportsbooks.append([
                    game_id,
                    date,
                    away_team_full_name,
                    home_team_full_name,
                    book_id,
                    spread_values['away'],
                    spread_values['home'],
                    total_values['over'],
                    game.get('boxscore', {}).get('stats', {}).get('away', {}).get('runs'),
                    game.get('boxscore', {}).get('stats', {}).get('home', {}).get('runs')
                ])
    columns = ['game_id', 'game_date', 'away_team', 'home_team', 'book_id', 'away_spread', 'home_spread', 'total', 'away_points', 'home_points']
    if sportsbooks:
        db_f.insert_data(table_name, columns, sportsbooks)

def get_sportsbooks_dates():
    start_date = current_date - timedelta(days=45)
    next_day = start_date
    while next_day < current_date:
        formatted_date = next_day.strftime('%Y%m%d')
        save_sportsbooks_data(formatted_date, next_day)
        next_day += timedelta(days=1)

def save_sportsbooks_data_as_csv(csv_filename):
    table_name = 'sportsbooks'
    rows, columns = db_f.export_table_to_csv(table_name)
    directory = os.path.dirname(__file__)
    subdirectory = 'Sportsbooks Data csv'
    file_path = os.path.join(directory, subdirectory, f'{csv_filename}.csv')
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(columns)
        # Write the data
        csv_writer.writerows(rows)

def save_jba_predictions_as_csv(csv_filename):
    table_name = 'jba_predictions'
    rows, columns = db_f.export_table_to_csv(table_name)
    directory = os.path.dirname(__file__)
    subdirectory = 'JBA Predictions csv'
    file_path = os.path.join(directory, subdirectory, f'{csv_filename}.csv')
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(columns)
        # Write the data
        csv_writer.writerows(rows)

for i in range (18, -1, -1):
    save_mlb_matches(i)
save_mlb_next_matches(0)
save_teams_games_stats()
save_players_stats()
db_f.commit_and_close()
#get_sportsbooks_dates()
#save_sportsbooks_data_as_csv('sportsbooks_data')
#save_jba_predictions_as_csv('jba_predictions')
#save_players_info()