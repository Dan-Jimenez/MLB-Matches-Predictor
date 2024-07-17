import sqlite3


conn = sqlite3.connect('mlb.db')
c = conn.cursor()


def commit_and_close():
    conn.commit()
    conn.close()

def _commit():
    conn.commit()

def _close():
    conn.close()

def avoid_id_matches_duplicates(table, link_id):
    query = f'SELECT 1 FROM {table} WHERE link_id = ?'
    c.execute(query, (link_id,))
    return c.fetchone() is not None

def avoid_players_stats_duplicates(table, link_id):
    query = f'SELECT 1 FROM {table} WHERE link_id = ?'
    c.execute(query, (link_id,))
    return c.fetchone() is not None

def avoid_players_duplicates(table, player_id):
    query = f'SELECT 1 FROM {table} WHERE player_id = ?'
    c.execute(query, (player_id,))
    return c.fetchone() is not None

def avoid_sportsbooks_duplicates(table, game_id):
    query = f'SELECT 1 FROM {table} WHERE game_id = ?'
    c.execute(query, (game_id,))
    return c.fetchone() is not None

def insert_data(table, columns, data):
    placeholders = ",".join(["?"] * len(columns))
    column_names = ",".join(columns)
    query = f'INSERT INTO {table} ({column_names}) VALUES ({placeholders})'
    c.executemany(query, data)

def delete_data(table, date):
    c.execute(f"DELETE FROM {table} WHERE DATE(game_date) = ?", (date,))

def get_column_data(table, column):
    query = f'SELECT {column} FROM {table}'
    c.execute(query)
    return c.fetchall()

def get_column_data_limit(table, column, game_date, limit_number):
    query = f"SELECT {column} FROM {table} WHERE game_date > ? ORDER BY id ASC LIMIT ?"
    c.execute(query, (game_date, limit_number,))
    return c.fetchall()

def get_next_matches_by_date(prediction_date):
    query = f"SELECT * FROM matches WHERE DATE(game_date) = DATE(?)"
    c.execute(query, (prediction_date,))
    return c.fetchall()

def get_teams_data(competition, season, team, game_date, columns):
    all_teams_data = {}
    columns_str = ', '.join(columns)
    query = f"SELECT side, {columns_str} FROM teams_games_stats WHERE competition = '{competition}' AND season = '{season}' AND team = '{team}' AND game_date < '{game_date}' ORDER BY game_date ASC"
    c.execute(query)
    results = c.fetchall()

    # Initialize the data structure if the team doesn't exist
    if team not in all_teams_data:
        all_teams_data[team] = {'away': {col: [] for col in columns}, 'home': {col: [] for col in columns}}

    # Accumulate the data for each game
    for result in results:
        side = result[0]
        side_data = {col: result[i + 1] for i, col in enumerate(columns)}
        for col, value in side_data.items():
            all_teams_data[team][side.lower()][col].append(value)

    return all_teams_data

def get_teams_id():
    query = "SELECT team_id FROM teams"
    c.execute(query)
    return c.fetchall()

def get_team_id(team_name):
    query = "SELECT team_id FROM teams WHERE team = ?"
    c.execute(query, (team_name,))
    return c.fetchall()

def get_team_roster(team):
    query = f"SELECT name, id FROM players WHERE team = '{team}'"
    c.execute(query)
    return c.fetchall()

def get_players_data(competition, season, team, game_date, columns, lineup):
    all_players_data = {}
    columns_str = ', '.join(['name', 'position', 'side'] + columns)
    query = f"SELECT {columns_str} FROM players_games_stats WHERE competition = '{competition}' AND season = '{season}' AND team = '{team}' AND game_date < '{game_date}' ORDER BY game_date ASC"
    c.execute(query)
    results = c.fetchall()

    for result in results:
        player_name = result[0]
        player_position = result[1]
        side = result[2]
        player_data = {col: result[i + 3] for i, col in enumerate(columns)}
        
        if player_name not in all_players_data:
            all_players_data[player_name] = {'position' : player_position, 'away': {col: [] for col in columns}, 'home': {col: [] for col in columns}}

        for col, value in player_data.items():
            all_players_data[player_name][side][col].append(value)

    return all_players_data

def get_teams_last_5_data(competition, season, team, game_date, columns):
    all_teams_data = {}
    columns_str = ', '.join(columns)
    query = f"SELECT side, {columns_str} FROM teams_games_stats WHERE competition = '{competition}' AND season = '{season}' AND team = '{team}' AND game_date < '{game_date}' ORDER BY game_date DESC LIMIT 10"
    c.execute(query)
    results = c.fetchall()

    # Initialize the data structure if the team doesn't exist
    if team not in all_teams_data:
        all_teams_data[team] = {col: [] for col in columns}

    # Accumulate the data for each game
    for result in results:
        side_data = {col: result[i + 1] for i, col in enumerate(columns)}
        for col, value in side_data.items():
            all_teams_data[team][col].append(value)
    return all_teams_data

def get_players_last_5_data(competition, season, team, game_date, columns, lineups):
    all_players_data = {}
    for player_name, id in lineups:
        query = f"SELECT {', '.join(['name', 'position', 'side'] + columns)} FROM players_games_stats WHERE competition = ? AND season = ? AND team = ? AND name = ? AND game_date < ? ORDER BY game_date DESC LIMIT 10"
        c.execute(query, (competition, season, team, player_name, game_date))
        results = c.fetchall()

        # Initialize the data structure if the player doesn't exist
        if team not in all_players_data:
            all_players_data[team] = {}

        if player_name not in all_players_data[team]:
            all_players_data[team][player_name] = {'stats': {col: [] for col in columns}}

        # Accumulate player data
        for result in results:
            player_position = result[1]  # Move this line here
            side = result[2]
            player_data = {col: result[i + 3] for i, col in enumerate(columns)}

            for col, value in player_data.items():
                all_players_data[team][player_name]['stats'][col].append(value)

    return all_players_data

def get_opponent_teams(competition, season, team, game_date):
    query = f"SELECT opponent_team, DATE(game_date) FROM teams_games_stats WHERE competition = '{competition}' AND season = '{season}' AND team = '{team}' AND game_date < '{game_date}' ORDER BY game_date ASC"
    c.execute(query)
    return c.fetchall()

def get_opponent_team_data(opponent_team, team, game_date, columns):
    all_opponents_data = {}
    columns_str = ', '.join(columns)
    query = f"SELECT {columns_str} FROM teams_games_stats WHERE team = '{opponent_team}' AND opponent_team = '{team}' AND DATE(game_date) = DATE('{game_date}')"
    c.execute(query)
    results = c.fetchall()

    # Initialize the data structure if the team doesn't exist
    all_opponents_data[opponent_team] = {col: [] for col in columns}    
    for result in results:
        side_data = {col: result[i] for i, col in enumerate(columns)}
        for col, value in side_data.items():
            all_opponents_data[opponent_team][col].append(value)

    return all_opponents_data

def export_table_to_csv(table_name):
    query = f"SELECT * FROM {table_name} ORDER BY id DESC"
    c.execute(query)
    rows = c.fetchall()
    columns = [description[0] for description in c.description]
    return rows, columns


""" c.execute('''
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    game_date TIMESTAMP,
    competition TEXT,
    season TEXT,
    away_team TEXT,
    home_team TEXT,
    away_score INTEGER,
    home_score INTEGER,
    winner TEXT,
    link TEXT,
    link_id INTEGER
);
''') """

""" c.execute('''
CREATE TABLE IF NOT EXISTS next_matches (
    id INTEGER PRIMARY KEY,
    game_date TIMESTAMP,
    competition TEXT,
    season TEXT,
    away_team TEXT,
    home_team TEXT,
    link TEXT,
    link_id INTEGER
);
''') """

""" c.execute('''
CREATE TABLE IF NOT EXISTS teams_games_stats (
    id INTEGER PRIMARY KEY,
    game_date TIMESTAMP,
    competition TEXT,
    season TEXT,
    side TEXT,
    team TEXT,
    opponent_team TEXT,
    runs INTEGER,
    doubles INTEGER,
    triples INTEGER,
    hr INTEGER,
    so INTEGER,
    bb INTEGER,
    hits INTEGER,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    rbi INTEGER,
    ab INTEGER,
    lob INTEGER,
    era REAL,
    link_id INTEGER
);
''') """

""" c.execute('''
CREATE TABLE IF NOT EXISTS pitchers_games_stats (
    id INTEGER PRIMARY KEY,
    game_date TIMESTAMP,
    competition TEXT,
    season TEXT,
    side TEXT,
    team TEXT,
    opponent_team TEXT,
    player_id INTEGER,
    name TEXT,
    position TEXT,
    jersey_number INTEGER,
    substitute INTEGER,
    runs INTEGER,
    hr INTEGER,
    so INTEGER,
    bb INTEGER,
    hits INTEGER,
    ip REAL,
    er INTEGER,
    bf INTEGER,
    outs INTEGER,
    pitches INTEGER,
    strikes INTEGER,
    era REAL,
    link_id INTEGER
);
''') """

""" c.execute('''
CREATE TABLE IF NOT EXISTS sportsbooks (
    id INTEGER PRIMARY KEY,
    game_date TIMESTAMP,
    game_id INTEGER,
    away_team TEXT,
    home_team TEXT,
    book_id INTEGER,
    away_spread REAL,
    home_spread REAL,
    total REAL,
    away_points INTEGER,
    home_points INTEGER
);
''') """