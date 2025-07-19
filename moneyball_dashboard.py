import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Position-specific metrics mapping ---
POSITION_METRICS = {
    'GK': {'Sv %': 0.2, 'xSv %': 0.2, 'Shutouts': 0.15, 'Pens Saved': 0.1, 'Conc': 0.15, 'Av Rat': 0.2},
    'CB': {'Tck/90': 0.15, 'Hdr %': 0.1, 'Int/90': 0.1, 'Clear': 0.1, 'Conc': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Ps C/90': 0.1, 'Av Rat': 0.15},
    'RB/LB': {'Tck/90': 0.15, 'Drb/90': 0.1, 'Asts/90': 0.1, 'Cr A': 0.1, 'Cr C': 0.1, 'Pas %': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Av Rat': 0.15},
    'RWB/LWB': {'Tck/90': 0.1, 'Drb/90': 0.1, 'Cr A': 0.1, 'Cr C': 0.1, 'Pas %': 0.1, 'Asts/90': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Av Rat': 0.2},
    'CDM': {'Tck/90': 0.1, 'Int/90': 0.1, 'Pas %': 0.1, 'Tck R': 0.1, 'K Tck/90': 0.1, 'Poss Won/90': 0.1, 'Ps C/90': 0.1, 'Asts/90': 0.1, 'Av Rat': 0.2},
    'CM': {'Pas %': 0.1, 'Asts/90': 0.1, 'K Ps/90': 0.1, 'Drb/90': 0.1, 'Poss Won/90': 0.1, 'Tck/90': 0.1, 'Int/90': 0.1, 'Cr A': 0.1, 'Cr C': 0.1, 'Av Rat': 0.2},
    'CAM': {'Asts/90': 0.15, 'Gls/90': 0.1, 'xG': 0.1, 'Cr A': 0.1, 'K Ps/90': 0.1, 'Drb/90': 0.1, 'Pas %': 0.1, 'Cr C': 0.1, 'Poss Won/90': 0.05, 'Av Rat': 0.2},
    'LM/RM': {'Asts/90': 0.1, 'Drb/90': 0.1, 'Cr A': 0.1, 'Ps C/90': 0.1, 'xG': 0.1, 'Gls/90': 0.1, 'K Ps/90': 0.1, 'Pas %': 0.1, 'Cr C': 0.05, 'Poss Won/90': 0.05, 'Av Rat': 0.2},
    'RW/LW': {'Gls/90': 0.15, 'Cr A': 0.1, 'Drb/90': 0.1, 'xG': 0.1, 'Shot/90': 0.1, 'K Ps/90': 0.1, 'Pas %': 0.05, 'Cr C': 0.05, 'Asts/90': 0.1, 'Poss Won/90': 0.05, 'Av Rat': 0.1},
    'ST': {'Gls/90': 0.2, 'xG': 0.15, 'Shots': 0.1, 'ShT %': 0.1, 'PoM': 0.1, 'Asts/90': 0.1, 'K Ps/90': 0.05, 'Drb/90': 0.05, 'Pas %': 0.05, 'Cr C': 0.05, 'Poss Won/90': 0.05, 'Av Rat': 0.1}
}

# --- Helper function to parse transfer value ---
def extract_value(val):
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

# --- League coefficient mapping ---
LEAGUE_COEFFICIENTS = {
    'Premier League': 1.00,
    'La Liga': 0.975,
    'Serie A': 0.955,
    'Bundesliga': 0.950,
    'Ligue 1': 0.945,
    'Liga Portugal Bwin': 0.930,
    'Eredivisie': 0.920,
    'Premiership': 0.910,
    'Belgian Pro League': 0.900,
    'Super Lig': 0.890,
    'Russian Premier League': 0.880,
    'Championship': 0.870,
    'Serie B': 0.860,
    '2. Bundesliga': 0.855,
    'Ligue 2': 0.850,
    'Segunda Divisi\u00f3n': 0.845,
    'MLS': 0.840,
    'Brazilian Serie A': 0.835,
    'Argentine Primera Divisi\u00f3n': 0.830,
    'Swiss Super League': 0.825,
    'Austrian Bundesliga': 0.820,
    'Greek Super League': 0.815,
    'Czech First League': 0.810,
    '3F Superliga': 0.805,
    'Polish Ekstraklasa': 0.800,
    'Ukrainian Premier League': 0.795,
    'Croatian HNL': 0.790,
    'Serbian SuperLiga': 0.785,
    'Romanian Liga I': 0.780,
    'Slovak Super Liga': 0.775,
    'Hungarian NB I': 0.770,
    'Cypriot First Division': 0.765,
    'Bulgarian First League': 0.760,
    'Scottish Championship': 0.755,
    'J1 League': 0.750,
    'K League 1': 0.745,
    'Allsvenskan': 0.740,
    'Eliteserien': 0.735,
    'A-League': 0.730,
    'Other Leagues': 0.700
}

# --- Moneyball score computation ---
def compute_moneyball_score(row, metric_weights):
    try:
        value = extract_value(row['Transfer Value'])
        if not value:
            return 0
    except:
        return 0

    perf_score = 0
    total_weight = sum(metric_weights.values())
    for metric, weight in metric_weights.items():
        try:
            raw_val = str(row[metric]).strip()
            if raw_val == '-' or raw_val == '' or raw_val.lower() == 'nan':
                continue
            if '%' in raw_val:
                val = float(raw_val.replace('%', '').replace(',', '')) / 100  # normalize percentage
            else:
                val = float(raw_val.replace(',', ''))
            perf_score += weight * val
        except:
            continue

    coeff = 1.0
    if 'Division' in row and isinstance(row['Division'], str):
        for league, factor in LEAGUE_COEFFICIENTS.items():
            if league.lower() in row['Division'].lower():
                coeff = factor
                break

    age_penalty = 1 + (row['Age'] - 18) * 0.02 if 'Age' in row else 1

    mins_raw = str(row['Mins']).strip() if 'Mins' in row else '0'
    try:
        minutes_played = float(mins_raw.replace(',', '')) if mins_raw != '-' else 0
    except:
        minutes_played = 0

    reliability_bonus = min(1.0, minutes_played / 2000)

    value_factor = 1 + (value / 1_000_000) * 0.1  # reduced impact of value
    score = ((perf_score / total_weight) * coeff * reliability_bonus) / (value_factor * age_penalty)
    return round(score, 3)

# --- Streamlit UI ---
st.set_page_config(page_title="Moneyball Football Dashboard", layout="wide")
st.title(":soccer: Football Manager Moneyball Dashboard")

uploaded_file = st.file_uploader("Upload HTML file with player data", type=["html"])

if uploaded_file:
    try:
        raw_html = uploaded_file.read()
        decoded_html = raw_html.decode(errors='replace')
        tables = pd.read_html(decoded_html)
        df = tables[0]
    except Exception as e:
        st.error(f"Failed to parse HTML file: {e}")
        st.stop()

    df = df[df['Age'] <= 22]  # Initial hard age limit
    df['Numeric Value'] = df['Transfer Value'].apply(extract_value)

    # --- Position Detection by frequency ---
    fm_position_map = {
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
    def detect_dominant_position_from_filename(filename):
        first_word = filename.split()[0].upper()
        for position, aliases in fm_position_map.items():
            if first_word in aliases:
                return position
        return None
    
    detected_position = detect_dominant_position_from_filename(uploaded_file.name) if uploaded_file.name else None


    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    age_lower = st.sidebar.number_input("Min Age", min_value=15, max_value=25, value=age_min)
    age_upper = st.sidebar.number_input("Max Age", min_value=15, max_value=25, value=age_max)

    value_min = float(df['Numeric Value'].min() / 1_000_000) if df['Numeric Value'].notna().any() else 0
    value_max = float(df['Numeric Value'].max() / 1_000_000) if df['Numeric Value'].notna().any() else 100
    val_lower = st.sidebar.number_input("Min Transfer Value (Millions)", min_value=0.0, max_value=500.0, value=1.5, step=0.5)
    val_upper = st.sidebar.number_input("Max Transfer Value (Millions)", min_value=0.0, max_value=500.0, value=value_max, step=0.5)

    # Apply filters (convert inputs back to base unit)
    df = df[(df['Age'] >= age_lower) & (df['Age'] <= age_upper)]
    df = df[df['Numeric Value'].between(val_lower * 1_000_000, val_upper * 1_000_000, inclusive='both')]

    position = st.selectbox("Select Position", list(POSITION_METRICS.keys()), index=list(POSITION_METRICS.keys()).index(detected_position) if detected_position else 0)
    metrics = POSITION_METRICS[position]
    st.markdown(f"**Evaluating {position}** using: {', '.join(metrics)}")

    df['Moneyball Score'] = df.apply(lambda row: compute_moneyball_score(row, metrics), axis=1)
    df = df.sort_values(by='Moneyball Score', ascending=False).reset_index(drop=True)

    top_n = st.slider("Number of top players to display", 5, 50, 10)

    available_cols = df.columns.tolist()
    base_cols = [col for col in ['Name', 'Club', 'Division', 'Age', 'Salary', 'Transfer Value', 'Apps', 'Moneyball Score'] if col in available_cols]
    metric_cols = [col for col in metrics if col in available_cols]
    display_cols = base_cols + metric_cols

    st.subheader(f"Top {position} Players")
    st.dataframe(df[display_cols].head(top_n))

    st.subheader("Score Distribution")
    fig = px.bar(df.head(top_n), x='Name', y='Moneyball Score', color='Club', title=f'Top {position} by Moneyball Score',
             labels={'Moneyball Score': 'Performance-to-Value Ratio'}, height=500)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    st.subheader("Value for Money")
    scatter_df = df.dropna(subset=['Moneyball Score', 'Numeric Value'])
    fig2 = px.scatter(scatter_df.head(50), x='Numeric Value', y='Moneyball Score', color='Age', hover_data=['Name', 'Club', 'Division'],
                  title='Transfer Value vs. Moneyball Score', labels={'Numeric Value': 'Transfer Value (M)', 'Moneyball Score': 'Performance-to-Value Ratio'})
    fig2.update_traces(marker=dict(size=12, opacity=0.7))
    fig2.update_layout(height=600)
    st.plotly_chart(fig2)

    st.subheader("Compare Players via Radar Chart")
    selected_players = st.multiselect("Select up to 3 players for radar comparison", df['Name'].head(top_n).tolist())

    if selected_players:
        radar_df = df[df['Name'].isin(selected_players)][['Name'] + metrics].dropna()
        fig3 = go.Figure()
        for _, row in radar_df.iterrows():
            values = [float(str(row[m]).replace('%','').replace(',','').strip()) if m in row else 0 for m in metrics]
            fig3.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=row['Name']
            ))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig3)

    st.download_button("Download Ranked Players CSV", df[display_cols].to_csv(index=False), file_name="ranked_players.csv")

else:
    st.info("Upload an HTML file containing a table of players with stats.")
