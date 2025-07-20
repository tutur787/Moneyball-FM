import re
from matplotlib import cm
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ====================================================================================
# CONFIGURATION AND CONSTANTS
# ====================================================================================

# Position-specific metrics with their weights
POSITION_METRICS = {
    'GK': {
        'Sv %': 0.2, 'xSv %': 0.2, 'Shutouts': 0.15, 'Pens Saved': 0.1, 
        'Conc': 0.15, 'Av Rat': 0.2
    },
    'CB': {
        'Tck/90': 0.15, 'Hdr %': 0.1, 'Int/90': 0.1, 'Clear': 0.1, 
        'Conc': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Ps C/90': 0.1, 
        'Av Rat': 0.15
    },
    'RB/LB': {
        'Tck/90': 0.15, 'Drb/90': 0.1, 'Asts/90': 0.1, 'Cr A': 0.1, 
        'Cr C': 0.1, 'Pas %': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 
        'Av Rat': 0.15
    },
    'RWB/LWB': {
        'Tck/90': 0.1, 'Drb/90': 0.1, 'Cr A': 0.1, 'Cr C': 0.1, 
        'Pas %': 0.1, 'Asts/90': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 
        'Av Rat': 0.2
    },
    'CDM': {
        'Tck/90': 0.1, 'Int/90': 0.1, 'Pas %': 0.1, 'Tck R': 0.1, 
        'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Ps C/90': 0.1, 'Asts/90': 0.1, 
        'Av Rat': 0.2
    },
    'CM': {
        'Pas %': 0.1, 'Asts/90': 0.1, 'K Ps/90': 0.1, 'Drb/90': 0.1, 
        'Poss Won/90': 0.1, 'Tck/90': 0.1, 'Int/90': 0.1, 'Cr A': 0.1, 
        'Cr C': 0.1, 'Av Rat': 0.2
    },
    'CAM': {
        'Asts/90': 0.15, 'Gls/90': 0.1, 'xG': 0.1, 'Cr A': 0.1, 
        'K Ps/90': 0.1, 'Drb/90': 0.1, 'Pas %': 0.1, 'Cr C': 0.1, 
        'Poss Won/90': 0.05, 'Av Rat': 0.2
    },
    'LM/RM': {
        'Asts/90': 0.1, 'Drb/90': 0.1, 'Cr A': 0.1, 'Ps C/90': 0.1, 
        'xG': 0.1, 'Gls/90': 0.1, 'K Ps/90': 0.1, 'Pas %': 0.1, 
        'Cr C': 0.05, 'Poss Won/90': 0.05, 'Av Rat': 0.2
    },
    'RW/LW': {
        'Gls/90': 0.15, 'Cr A': 0.1, 'Drb/90': 0.1, 'xG': 0.1, 
        'Shot/90': 0.1, 'K Ps/90': 0.1, 'Pas %': 0.05, 'Cr C': 0.05, 
        'Asts/90': 0.1, 'Poss Won/90': 0.05, 'Av Rat': 0.1
    },
    'ST': {
        'Gls/90': 0.2, 'xG': 0.15, 'Shots': 0.1, 'ShT %': 0.1, 
        'PoM': 0.1, 'Asts/90': 0.1, 'K Ps/90': 0.05, 'Drb/90': 0.05, 
        'Pas %': 0.05, 'Cr C': 0.05, 'Poss Won/90': 0.05, 'Av Rat': 0.1
    }
}

# League strength coefficients (1.0 = strongest)
LEAGUE_COEFFICIENTS = {
    'Premier League': 1.0000,
    'Serie A TIM': 0.9925,
    'LaLiga EA Sports': 0.9850,
    'Bundesliga': 0.9775,
    'Ligue 1 McDonald\'s': 0.9700,
    'Eredivisie': 0.9625,
    'Liga Portugal Betclic': 0.9550,
    '3F Superliga': 0.9475,
    'Allsvenskan': 0.9400,
    'Trendyol Süper Lig': 0.9325,
    'cinch Premiership': 0.9250,
    'Admiral Bundesliga': 0.9175,
    'Jupiler Pro League': 0.9100,
    'Raiffeisen Super League': 0.9025,
    'Super League Interwetten': 0.8950,
    'Favbet Liha': 0.8875,
    'Mozzart Super Liga': 0.8800,
    'Sky Bet Championship': 0.8725,
    'SuperSport HNL': 0.8650,
    'Cyta Championship': 0.8575,
    'Ligat Tel Aviv Stock Exchange': 0.8500,
    'FORTUNA:LIGA': 0.8425,
    'NIKÉ Liga': 0.8350,
    'PKO Ekstraklasa': 0.8275,
    '2. Bundesliga': 0.8200,
    'OTP Bank Liga': 0.8125,
    'Eliteserien': 0.8050,
    'Serie BKT': 0.7975,
    'Casa Liga I': 0.7900,
    'Prva Liga Telemach': 0.7825,
    'Ligue 2 BKT': 0.7750,
    'Tinkoff Russian Premier Liga': 0.7675,
    'Efbet League': 0.7600,
    'm:tel Premijer liga Bosne i Hercegovine': 0.7525,
    'Qazaqstan Premer Lïgası': 0.7450,
    'LaLiga 2 Hypermotion': 0.7375,
    'SSE Airtricity League Premier Division': 0.7300,
    'Keuken Kampioen Divisie': 0.7225,
    'Superettan': 0.7150,
    'Sky Bet League One': 0.7075,
    'Crystalbet Erovnuli Liga': 0.7000,
    'Azərbaycan Premyer Liqası': 0.6925,
    'Liga Portugal 2 SABSEG': 0.6850,
    'Brack.ch Challenge League': 0.6775,
    'Spor Toto 1. Lig': 0.6700,
    'Belarusbank Vyšejšaja Liha': 0.6625,
    'Pepsi Max Deild': 0.6550,
    'OBOS-Ligaen': 0.6475,
    'Veikkausliiga': 0.6400,
    'Sports Direct Premiership': 0.6325,
    'Other Leagues': 0.6225
}

# FM position mappings for auto-detection
FM_POSITION_MAP = {
    'GK': ['GK'],
    'CB': ['CB', 'DC', 'DCL', 'DCR'],
    'RB': ['RB', 'LB'],
    'RWB/ LWB': ['RWB', 'LWB'],
    'CDM': ['CDM', 'DMC'],
    'CM': ['CM', 'MC', 'DMC/CM'],
    'CAM': ['CAM', 'AMC'],
    'LM/RM': ['LM', 'LW', 'RM', 'RW'],
    'RW/LW': ['RW', 'RWM', 'LW', 'LWM'],
    'ST': ['ST', 'CF']
}

ROLE_PROFILES = {
    'GK': {
        'Goalkeeper': {},
        'Sweeper Keeper': {
            'Pas %': 70,
            'Av Rat': 6.8,
            'Conc': 10
        },
    },
    'CB': {
        'Central Defender': {
            'Tck/90': 2.0,
            'Hdr %': 60
        },
        'Ball-Playing Defender': {
            'Pas %': 80,
            'Ps C/90': 40,
            'K Ps/90': 0.5
        },
        'No-Nonsense Centre-Back': {
            'Clear': 5.0,
            'Hdr %': 70,
            'Pas %': 75
        },
        'Libero': {
            'Ps C/90': 45,
            'Drb/90': 0.5,
            'K Ps/90': 0.4
        },
        'Wide Centre-Back': {
            'Cr C': 0.5,
            'Ps C/90': 35
        },
    },
    'RB/LB': {
        'Full-Back': {
            'Tck/90': 2.0,
            'Pas %': 70
        },
        'Wing-Back': {
            'Cr A': 0.8,
            'Asts/90': 0.15,
            'Drb/90': 1.0
        },
        'Complete Wing-Back': {
            'Cr A': 1.2,
            'Tck/90': 2.0,
            'Asts/90': 0.2
        },
        'Inverted Full-Back': {
            'Pas %': 82,
            'Ps C/90': 35,
            'Cr C': 1.0
        },
        'No-Nonsense Full-Back': {
            'Tck/90': 2.5,
            'Clear': 3.0,
            'Cr A': 0.5
        },
    },
    'RWB/LWB': {
        'Wing-Back': {
            'Cr A': 1.0,
            'Drb/90': 1.5
        },
        'Complete Wing-Back': {
            'Cr A': 1.5,
            'Asts/90': 0.25,
            'Tck/90': 1.8
        },
        'Inverted Wing-Back': {
            'Pas %': 80,
            'K Ps/90': 0.7,
            'Ps C/90': 40
        },
        'Defensive Winger': {
            'Tck/90': 2.0,
            'Poss Won/90': 2.0,
            'Cr A': 1.0
        },
    },
    'CDM': {
        'Defensive Midfielder': {
            'Tck/90': 2.2,
            'Pas %': 75
        },
        'Anchor Man': {
            'Tck/90': 2.8,
            'Int/90': 2.2,
            'K Ps/90': 0.5
        },
        'Deep-Lying Playmaker': {
            'Pas %': 83,
            'K Ps/90': 0.8,
            'Ps C/90': 45
        },
        'Ball-Winning Midfielder': {
            'Tck/90': 3.0,
            'Poss Won/90': 2.5
        },
        'Regista': {
            'Pas %': 85,
            'K Ps/90': 1.0,
            'Asts/90': 0.15
        },
        'Half-Back': {
            'Int/90': 2.0,
            'Ps C/90': 40,
            'Tck/90': 2.0
        },
        'Segundo Volante': {
            'Tck/90': 2.0,
            'Gls/90': 0.08,
            'Shot/90': 0.8
        },
        'Roaming Playmaker': {
            'K Ps/90': 0.9,
            'Drb/90': 1.0,
            'Pas %': 82
        },
    },
    'CM': {
        'Central Midfielder': {
            'Pas %': 75,
            'Tck/90': 1.5
        },
        'Box-to-Box Midfielder': {
            'Poss Won/90': 1.8,
            'Shot/90': 1.0,
            'Asts/90': 0.1
        },
        'Advanced Playmaker': {
            'K Ps/90': 1.0,
            'Asts/90': 0.2,
            'Cr A': 0.8
        },
        'Mezzala': {
            'Drb/90': 1.5,
            'Cr C': 0.8,
            'Gls/90': 0.1
        },
        'Carrilero': {
            'Tck/90': 2.0,
            'Pas %': 82,
            'Poss Won/90': 1.5
        },
        'Ball-Winning Midfielder': {
            'Tck/90': 2.8,
            'Poss Won/90': 2.2
        },
        'Deep-Lying Playmaker': {
            'Pas %': 84,
            'K Ps/90': 0.9
        },
    },
    'CAM': {
        'Attacking Midfielder': {
            'K Ps/90': 0.8,
            'Asts/90': 0.25
        },
        'Advanced Playmaker': {
            'K Ps/90': 1.2,
            'Pas %': 85,
            'Asts/90': 0.3
        },
        'Shadow Striker': {
            'Gls/90': 0.3,
            'Shot/90': 2.0,
            'xG': 0.25
        },
        'Trequartista': {
            'Drb/90': 2.0,
            'K Ps/90': 1.0,
            'Cr A': 1.2
        },
        'Enganche': {
            'Pas %': 88,
            'K Ps/90': 1.3,
            'Drb/90': 1.5
        },
    },
    'LM/RM': {
        'Wide Midfielder': {
            'Cr C': 1.0,
            'Tck/90': 1.0
        },
        'Winger': {
            'Cr A': 1.5,
            'Drb/90': 2.2,
            'Pas %': 70
        },
        'Wide Playmaker': {
            'K Ps/90': 1.0,
            'Asts/90': 0.25,
            'Pas %': 82
        },
        'Inverted Winger': {
            'Gls/90': 0.2,
            'Shot/90': 1.8,
            'Cr C': 1.0
        },
        'Defensive Winger': {
            'Tck/90': 1.8,
            'Poss Won/90': 1.5
        },
    },
    'RW/LW': {
        'Winger': {
            'Cr A': 1.2,
            'Drb/90': 2.5
        },
        'Inside Forward': {
            'Shot/90': 2.2,
            'Gls/90': 0.25,
            'xG': 0.2
        },
        'Inverted Winger': {
            'Cr C': 1.5,
            'Shot/90': 2.0,
            'K Ps/90': 0.8
        },
        'Wide Target Forward': {
            'Hdr %': 60,
            'Av Rat': 6.8,
            'Shot/90': 1.5
        },
        'Raumdeuter': {
            'Gls/90': 0.3,
            'xG': 0.25,
            'Poss Won/90': 1.0
        },
        'Wide Playmaker': {
            'K Ps/90': 1.1,
            'Asts/90': 0.3
        },
    },
    'ST': {
        'Advanced Forward': {
            'Gls/90': 0.35,
            'Shot/90': 2.0
        },
        'Poacher': {
            'Gls/90': 0.5,
            'ShT %': 40,
            'xG': 0.35
        },
        'Target Man': {
            'Hdr %': 65,
            'K Hdrs/90': 0.8,
            'Av Rat': 6.8
        },
        'Complete Forward': {
            'Gls/90': 0.35,
            'Asts/90': 0.15,
            'Drb/90': 1.2
        },
        'Pressing Forward': {
            'Poss Won/90': 2.0,
            'Tck/90': 1.0
        },
        'Deep-Lying Forward': {
            'K Ps/90': 0.8,
            'Asts/90': 0.2,
            'Pas %': 75
        },
        'False Nine': {
            'K Ps/90': 1.0,
            'Asts/90': 0.25,
            'Pas %': 80
        },
    },
}

# ====================================================================================
# UTILITY FUNCTIONS
# ====================================================================================

def extract_value(val):
    """Extract numeric value from transfer value strings like '$1.5M' or '$500K'"""
    if isinstance(val, str) and "$" in val:
        try:
            parts = re.findall(r"\$([\d\.]+)([MK])", val)
            if not parts:
                return None
            values = []
            for num, suffix in parts:
                multiplier = 1_000_000 if suffix == 'M' else 1_000
                values.append(float(num) * multiplier)
            return sum(values) / len(values) if values else None
        except:
            return None
    return None

def get_metric_value(row, metric):
    """Safely extract and convert metric values from dataframe row"""
    try:
        raw_val = str(row[metric]).strip()
        if raw_val == '-' or raw_val == '' or raw_val.lower() == 'nan':
            return None
        if '%' in raw_val:
            return float(raw_val.replace('%', '').replace(',', '')) / 100
        else:
            return float(raw_val.replace(',', ''))
    except:
        return None

def get_league_coefficient(row):
    """Get league coefficient based on division"""
    if 'Division' in row and isinstance(row['Division'], str):
        for league, factor in LEAGUE_COEFFICIENTS.items():
            if league.lower() in row['Division'].lower():
                return factor
    return LEAGUE_COEFFICIENTS.get('Other Leagues', 0.6225)

def detect_dominant_position_from_filename(filename):
    """Auto-detect position from uploaded filename"""
    if not filename:
        return None
    first_word = filename.split()[0].upper()
    for position, aliases in FM_POSITION_MAP.items():
        if first_word in aliases:
            return position
    return None

# ====================================================================================
# SCORING FUNCTIONS
# ====================================================================================

def compute_moneyball_score(row, metric_weights, value_weight=0.3, league_weight=0.3):
    """Calculate the Moneyball score for a player"""
    # Get transfer value
    try:
        value = extract_value(row['Transfer Value'])
        if not value:
            return 0
    except:
        return 0

    # Calculate performance score
    perf_score = 0
    total_weight = sum(metric_weights.values())
    
    for metric, weight in metric_weights.items():
        val = get_metric_value(row, metric)
        if val is not None:
            perf_score += weight * val

    # Apply league coefficient
    coeff = get_league_coefficient(row)
    coeff = (1 - league_weight) * 1.0 + league_weight * coeff

    # Age penalty (increases with age)
    age_penalty = 1 + (row['Age'] - 18) * 0.02 if 'Age' in row else 1

    # Minutes reliability bonus
    mins_raw = str(row['Mins']).strip() if 'Mins' in row else '0'
    try:
        minutes_played = float(mins_raw.replace(',', '')) if mins_raw != '-' else 0
    except:
        minutes_played = 0
    reliability_bonus = min(1.0, minutes_played / 2000)

    # Value factor (higher value = lower score)
    value_factor = 1.0 if value_weight == 0 else 1 + ((value / 1_000_000) * value_weight)
    
    # Final score calculation
    score = ((perf_score / total_weight) * coeff * reliability_bonus) / (value_factor * age_penalty)
    return round(score, 3)

def project_moneyball_score_at_age(row, target_age=25, metric_weights=None):
    """Project a player's Moneyball score at a target age"""
    if not metric_weights or 'Age' not in row or 'Moneyball Score' not in row:
        return 0
    try:
        current_age_penalty = 1 + (row['Age'] - 18) * 0.02
        projected_age_penalty = 1 + (target_age - 18) * 0.02
        projected_score = row['Moneyball Score'] * (projected_age_penalty / current_age_penalty)
        return round(projected_score, 3)
    except:
        return 0

def assign_player_tags(row, position, metrics):
    """Assign style tags based on player's stats and position"""
    tags = []
    
    def val(metric):
        v = get_metric_value(row, metric)
        return v if v is not None else 0

    # Attacking positions
    if position in ['CM', 'CAM', 'LM/RM', 'RW/LW']:
        if val('Asts/90') > 0.3 or val('Cr A') > 1.0:
            tags.append("Creator")
        if val('Drb/90') > 2.0:
            tags.append("Dribbler")
        if val('Gls/90') > 0.3 or val('xG') > 3:
            tags.append("Finisher")

    # Defensive positions
    if position in ['CDM', 'CB', 'RB/LB', 'RWB/LWB']:
        if val('Tck/90') > 2.5 or val('K Tck/90') > 1.0:
            tags.append("Ball-Winner")
        if val('Int/90') > 2.0:
            tags.append("Interceptor")
        if val('Clear') > 50:
            tags.append("Defender")

    # Striker specific
    if position == 'ST':
        if val('Gls/90') > 0.5:
            tags.append("Poacher")
        if val('ShT %') > 50:
            tags.append("Clinical")
        if val('Drb/90') > 1.5:
            tags.append("Mobile Striker")
    
    # Performance tag
    if 'Av Rat' in metrics and val('Av Rat') >= 7.5:
        tags.append("Performer")

    return ", ".join(tags) if tags else "Balanced"

def detect_player_roles(row, position, return_scores=False):
    """
    Detect suitable roles for a player based on their stats.
    Returns either the best role or all suitable roles with scores.
    """
    def val(metric):
        try:
            v = row.get(metric, '-')
            if isinstance(v, str):
                v = v.strip().replace('%', '').replace(',', '')
                if v == '-' or v == '' or v.lower() == 'nan':
                    return 0.0
            return float(v)
        except:
            return 0.0
    
    roles = ROLE_PROFILES.get(position, {})
    role_scores = {}
    
    # Handle special cases
    if position == 'GK' and 'Goalkeeper' in roles:
        role_scores['Goalkeeper'] = 0.5  # Base score for default role
    
    for role, thresholds in roles.items():
        if not thresholds:
            continue
            
        score = 0
        met_requirements = 0
        total_requirements = len(thresholds)
        
        for metric, threshold in thresholds.items():
            player_val = val(metric)
            
            # Handle reverse comparisons (where lower is better)
            if metric in ['Conc', 'Cr C', 'Cr A', 'Poss Won/90', 'Drb/90']:
                # Check context for reverse comparison
                if (role == 'Sweeper Keeper' and metric == 'Conc') or \
                   (role == 'Inverted Full-Back' and metric == 'Cr C') or \
                   (role == 'No-Nonsense Full-Back' and metric == 'Cr A') or \
                   (role == 'Defensive Winger' and metric == 'Cr A') or \
                   (role == 'Raumdeuter' and metric == 'Poss Won/90') or \
                   (role == 'Enganche' and metric == 'Drb/90'):
                    if player_val <= threshold:
                        met_requirements += 1
                        score += 1.5 - (player_val / threshold if threshold > 0 else 0)
                else:
                    # Normal comparison
                    if player_val >= threshold:
                        met_requirements += 1
                        score += min((player_val / threshold), 2.0)
            else:
                # Normal comparison (higher is better)
                if player_val >= threshold:
                    met_requirements += 1
                    # Bonus for exceeding threshold
                    score += min((player_val / threshold), 2.0)
        
        # Calculate final score - must meet at least 60% of requirements
        if met_requirements >= (total_requirements * 0.6):
            role_scores[role] = score / total_requirements
        
    # If no roles qualify, assign default
    if not role_scores:
        default_roles = {
            'GK': 'Goalkeeper',
            'CB': 'Central Defender',
            'RB/LB': 'Full-Back',
            'RWB/LWB': 'Wing-Back',
            'CDM': 'Defensive Midfielder',
            'CM': 'Central Midfielder',
            'CAM': 'Attacking Midfielder',
            'LM/RM': 'Wide Midfielder',
            'RW/LW': 'Winger',
            'ST': 'Advanced Forward'
        }
        return default_roles.get(position, 'Unclassified')
    
    if return_scores:
        # Return all roles with scores > 0.5, sorted by score
        suitable_roles = {k: v for k, v in role_scores.items() if v > 0.5}
        return sorted(suitable_roles.items(), key=lambda x: x[1], reverse=True)
    else:
        # Return the best role
        return max(role_scores, key=role_scores.get)

# ====================================================================================
# DATA PROCESSING FUNCTIONS
# ====================================================================================

def calculate_positional_benchmarks(df, metrics):
    """Calculate benchmark stats for the selected position"""
    benchmarks = {}
    for metric in metrics:
        try:
            positional_values = pd.to_numeric(
                df[metric].replace('-', None), 
                errors='coerce'
            )
            benchmarks[metric] = {
                'mean': positional_values.mean(),
                'median': positional_values.median(),
                '80th': positional_values.quantile(0.80)
            }
        except:
            benchmarks[metric] = {'mean': None, 'median': None, '80th': None}
    return benchmarks

def add_percentile_columns(df, metrics):
    """Add percentile ranking columns for each metric"""
    for metric in metrics:
        col_name = f"{metric} Percentile"
        try:
            df[col_name] = pd.to_numeric(
                df[metric].astype(str).str.replace('%', '').str.replace(',', ''), 
                errors='coerce'
            )
            df[col_name] = df[col_name].rank(pct=True) * 100
            df[col_name] = df[col_name].round(1)
        except:
            df[col_name] = None
    return df

def prepare_display_columns(df, metrics, show_percentiles=False):
    """Prepare columns for display based on user preferences"""
    available_cols = df.columns.tolist()
    base_cols = [col for col in ['Name', 'Club', 'Nat', 'Division', 'Age', 'Salary', 
                                  'Transfer Value', 'Apps', 'Moneyball Score', 
                                  'Style Tags', 'Best Role'] if col in available_cols]
    
    if show_percentiles:
        percentile_cols = [f"{m} Percentile" for m in metrics if f"{m} Percentile" in df.columns]
        return base_cols + percentile_cols
    else:
        metric_cols = [col for col in metrics if col in df.columns]
        return base_cols + metric_cols

# ====================================================================================
# VISUALIZATION FUNCTIONS
# ====================================================================================

def color_text_by_percentile(val):
    """Apply color gradient based on percentile value"""
    try:
        norm_val = float(val) / 100
        rgba = cm.RdYlGn(norm_val)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f'color: rgb({r},{g},{b})'
    except:
        return ''

def create_value_scatter_plot(df):
    """Create scatter plot for value vs performance"""
    scatter_df = df.dropna(subset=['Moneyball Score', 'Numeric Value'])
    fig = px.scatter(
        scatter_df.head(50), 
        x='Numeric Value', 
        y='Moneyball Score', 
        color='Age', 
        hover_data=['Name', 'Club', 'Division'],
        title='Transfer Value vs. Moneyball Score', 
        labels={
            'Numeric Value': 'Transfer Value (M)', 
            'Moneyball Score': 'Performance-to-Value Ratio'
        }
    )
    fig.update_traces(marker=dict(size=12, opacity=0.7))
    fig.update_layout(height=600)
    return fig

def create_hidden_gems_plot(gems_df):
    """Create visualization for hidden gems"""
    fig = px.scatter(
        gems_df, 
        x='Numeric Value', 
        y='Moneyball Score', 
        hover_data=['Name', 'Club'],
        color='Score Percentile', 
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Hidden Gems Highlighted in Market',
        labels={
            'Numeric Value': 'Transfer Value (M)', 
            'Moneyball Score': 'Performance-to-Value Ratio', 
            'Score Percentile': 'Performance Score Percentile'
        }
    )
    fig.update_traces(marker=dict(size=10, opacity=0.6))
    fig.update_layout(legend_title_text='Hidden Gem', height=500)
    return fig

# ====================================================================================
# MAIN APPLICATION
# ====================================================================================

def main():
    # Page configuration
    st.set_page_config(page_title="Moneyball Football Dashboard", layout="wide")
    st.title(":soccer: Football Manager Moneyball Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload HTML file with player data", type=["html"])

    if not uploaded_file:
        st.info("Upload an HTML file containing a table of players with stats.")
        return

    # Load and parse data
    try:
        raw_html = uploaded_file.read()
        decoded_html = raw_html.decode(errors='replace')
        tables = pd.read_html(decoded_html)
        df = tables[0]
    except Exception as e:
        st.error(f"Failed to parse HTML file: {e}")
        return

    # Initial data preparation
    df['Numeric Value'] = df['Transfer Value'].apply(extract_value)

    # Auto-detect position from filename
    detected_position = detect_dominant_position_from_filename(uploaded_file.name)

    # Sidebar filters
    create_sidebar_filters(df)

    # Apply filters
    df = apply_filters(df)

    # Main content area
    position = st.selectbox(
        "Select Position", 
        list(POSITION_METRICS.keys()), 
        index=list(POSITION_METRICS.keys()).index(detected_position) if detected_position else 0
    )
    
    metrics = POSITION_METRICS[position]
    
    # Get user-adjusted weights
    normalized_weights = get_user_weights(position, metrics)
    
    # Get global factors
    value_impact = st.session_state.get('value_impact', 0.3)
    league_impact = st.session_state.get('league_impact', 0.3)
    
    # Process data
    df = process_player_data(df, position, metrics, normalized_weights, value_impact, league_impact)
    
    # Display results
    display_results(df, position, metrics)
    
    # Visualizations
    display_visualizations(df)
    
    # Hidden gems analysis
    display_hidden_gems(df)
    
    # Age projection analysis
    display_age_projections(df)

def create_sidebar_filters(df):
    """Create all sidebar filters"""
    st.sidebar.header("Filters")
    
    # Age filters
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    st.session_state['age_lower'] = st.sidebar.number_input(
        "Min Age", min_value=15, max_value=25, value=age_min
    )
    st.session_state['age_upper'] = st.sidebar.number_input(
        "Max Age", min_value=15, max_value=45, value=25
    )

    # Value filters
    value_max = float(df['Numeric Value'].max() / 1_000_000) if df['Numeric Value'].notna().any() else 100
    st.session_state['val_lower'] = st.sidebar.number_input(
        "Min Transfer Value (Millions)", 
        min_value=0.0, max_value=500.0, value=1.5, step=0.5
    )
    st.session_state['val_upper'] = st.sidebar.number_input(
        "Max Transfer Value (Millions)", 
        min_value=0.0, max_value=500.0, value=value_max, step=0.5
    )

    # Nationality filter
    if 'Nat' in df.columns:
        nat_options = sorted(df['Nat'].dropna().unique())
        st.session_state['selected_nations'] = st.sidebar.multiselect(
            "Filter by Nationality", options=nat_options
        )

    # League filter
    if 'Division' in df.columns:
        league_options = sorted(df['Division'].dropna().unique())
        st.session_state['selected_leagues'] = st.sidebar.multiselect(
            "Filter by League/Division", options=league_options
        )

    st.sidebar.markdown("---")
    st.session_state['show_percentiles'] = st.sidebar.checkbox(
        "Show Percentile Rankings", value=False
    )

def apply_filters(df):
    """Apply all active filters to dataframe"""
    # Age filter
    df = df[
        (df['Age'] >= st.session_state.get('age_lower', 15)) & 
        (df['Age'] <= st.session_state.get('age_upper', 25))
    ]
    
    # Value filter
    val_lower = st.session_state.get('val_lower', 0) * 1_000_000
    val_upper = st.session_state.get('val_upper', 500) * 1_000_000
    df = df[df['Numeric Value'].between(val_lower, val_upper, inclusive='both')]
    
    # Nationality filter
    selected_nations = st.session_state.get('selected_nations', [])
    if selected_nations and 'Nat' in df.columns:
        df = df[df['Nat'].isin(selected_nations)]
    
    # League filter
    selected_leagues = st.session_state.get('selected_leagues', [])
    if selected_leagues and 'Division' in df.columns:
        df = df[df['Division'].isin(selected_leagues)]
    
    return df

def get_user_weights(position, metrics):
    """Get user-adjusted metric weights from sidebar"""
    st.sidebar.markdown("### Adjust Metric Weights")
    with st.sidebar.expander(f"⚖️ Weights for {position}", expanded=False):
        weight_inputs = {}
        for metric, default_weight in metrics.items():
            weight_inputs[metric] = st.slider(
                label=metric,
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                key=f"{position}_{metric}"
            )
    
    st.sidebar.markdown("### Global Factors")
    st.session_state['value_impact'] = st.sidebar.slider(
        "Transfer Value Impact", 0.0, 1.0, value=0.3, step=0.01
    )
    st.session_state['league_impact'] = st.sidebar.slider(
        "League Coefficient Impact", 0.0, 1.0, value=0.3, step=0.01
    )
    
    # Normalize weights
    total = sum(weight_inputs.values()) or 1
    return {k: v / total for k, v in weight_inputs.items()}

def process_player_data(df, position, metrics, normalized_weights, value_impact, league_impact):
    """Process all player calculations and add computed columns"""
    # Calculate benchmarks
    benchmarks = calculate_positional_benchmarks(df, metrics)
    
    # Add percentile columns
    df = add_percentile_columns(df, metrics)
    
    # Calculate Moneyball score
    df['Moneyball Score'] = df.apply(
        lambda row: compute_moneyball_score(
            row, normalized_weights, value_impact, league_impact
        ),
        axis=1
    )
    
    # Sort by score
    df = df.sort_values(by='Moneyball Score', ascending=False).reset_index(drop=True)
    
    # Add projected score
    df['Projected Score (25)'] = df.apply(
        lambda row: project_moneyball_score_at_age(
            row, target_age=25, metric_weights=metrics
        ), 
        axis=1
    )
    
    # Add style tags
    df['Style Tags'] = df.apply(
        lambda row: assign_player_tags(row, position, metrics), 
        axis=1
    )
    
    # Apply style tag filter if selected
    selected_tags = st.sidebar.multiselect(
        "Filter by Style Tags", 
        options=df['Style Tags'].unique()
    )
    if selected_tags:
        df = df[df['Style Tags'].isin(selected_tags)]

    # Apply inferred roles

    df['Best Role'] = df.apply(
        lambda row: detect_player_roles(row, position), 
        axis=1
    )
    selected_roles = st.sidebar.multiselect(
        "Filter by Best Role", 
        options=df['Best Role'].unique()
    )
    if selected_roles:
        df = df[df['Best Role'].isin(selected_roles)]
    
    # Store benchmarks for display
    st.session_state['benchmarks'] = benchmarks
    
    return df

def display_results(df, position, metrics):
    """Display main results table and benchmarks"""
    st.markdown(f"**Evaluating {position}** using: {', '.join(metrics)}")
    
    # Prepare display columns
    show_percentiles = st.session_state.get('show_percentiles', False)
    display_cols = prepare_display_columns(df, metrics, show_percentiles)
    
    # Display top players
    st.subheader(f"Top {position} Players")
    top_n = 10
    
    if show_percentiles:
        percentile_cols = [f"{m} Percentile" for m in metrics if f"{m} Percentile" in df.columns]
        styled_df = df[display_cols].head(top_n).style\
            .applymap(color_text_by_percentile, subset=percentile_cols)\
            .format("{:.1f}", subset=percentile_cols)
        st.dataframe(styled_df)
    else:
        st.dataframe(df[display_cols].head(top_n))
    
    # Display benchmarks
    if 'benchmarks' in st.session_state:
        st.subheader(f"{position} Benchmarks")
        benchmarks = st.session_state['benchmarks']
        bm_df = pd.DataFrame(benchmarks).T.rename(columns={
            'mean': 'Mean',
            'median': 'Median',
            '80th': '80th Percentile'
        }).dropna()
        st.dataframe(bm_df.style.format("{:.2f}"))
    
    # Download button
    st.download_button(
        "Download Ranked Players CSV", 
        df[display_cols].to_csv(index=False), 
        file_name="ranked_players.csv"
    )

def display_visualizations(df):
    """Display main visualizations"""
    st.subheader("Value for Money")
    fig = create_value_scatter_plot(df)
    st.plotly_chart(fig)

def display_hidden_gems(df):
    """Display hidden gems analysis"""
    st.subheader("🔍 Hidden Gems: High Performers, Low Cost")
    
    # Filter valid data
    valid_df = df[(df['Moneyball Score'] > 0) & (df['Numeric Value'].notnull())]
    
    # Calculate percentiles
    valid_df['Value Percentile'] = valid_df['Numeric Value'].rank(pct=True)
    valid_df['Score Percentile'] = valid_df['Moneyball Score'].rank(pct=True)
    
    # Find hidden gems
    gems_df = valid_df[
        (valid_df['Value Percentile'] <= 0.4) & 
        (valid_df['Score Percentile'] >= 0.7)
    ]
    
    if not gems_df.empty:
        st.markdown(f"Found **{len(gems_df)} hidden gems**. These players have high performance scores relative to their low market value.")
        
        # Get display columns
        show_percentiles = st.session_state.get('show_percentiles', False)
        metrics = POSITION_METRICS[st.session_state.get('current_position', 'CM')]
        display_cols = prepare_display_columns(gems_df, metrics, show_percentiles)
        
        if show_percentiles:
            gem_cols = [col for col in display_cols if 'Percentile' in col]
            styled_gems = gems_df[display_cols].head(10).style\
                .applymap(color_text_by_percentile, subset=gem_cols)\
                .format("{:.1f}", subset=gem_cols)
            st.dataframe(styled_gems)
        else:
            st.dataframe(gems_df[display_cols].head(10))
        
        # Create visualization
        fig = create_hidden_gems_plot(gems_df)
        st.plotly_chart(fig)
    else:
        st.info("No hidden gems matched the current filters.")

def display_age_projections(df):
    """Display age projection analysis"""
    st.subheader("📈 Age Projection (Peak at 25)")
    
    growth_df = df.copy()
    growth_df['Score Growth %'] = 100 * (
        growth_df['Projected Score (25)'] - growth_df['Moneyball Score']
    ) / growth_df['Moneyball Score']
    growth_df = growth_df.sort_values(by='Projected Score (25)', ascending=False)
    
    cols_to_show = ['Name', 'Age', 'Moneyball Score', 'Projected Score (25)', 'Score Growth %']
    st.dataframe(
        growth_df[cols_to_show].head(10).style.format({
            'Moneyball Score': '{:.2f}',
            'Projected Score (25)': '{:.2f}',
            'Score Growth %': '{:.1f}'
        })
    )

# ====================================================================================
# RUN APPLICATION
# ====================================================================================

if __name__ == "__main__":
    main()