import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Position-specific metrics mapping ---
POSITION_METRICS = {
    'GK': ['Sv %', 'xSv %', 'Shutouts', 'Pens Saved', 'Conc'],
    'CB': ['Tck/90', 'Hdr %', 'Int/90', 'Clear', 'Conc'],
    'RB/LB': ['Tck/90', 'Drb/90', 'Asts/90', 'Cr A', 'Cr C'],
    'RWB/LWB': ['Tck/90', 'Drb/90', 'Cr A', 'Cr C', 'Pas %'],
    'CDM': ['Tck/90', 'Int/90', 'Pas %', 'Tck R', 'K Tck/90', 'Poss Won/90', 'Ps C/90'],
    'CM': ['Pas %', 'Asts/90', 'K Ps/90', 'Drb/90', 'Poss Won/90'],
    'CAM': ['Asts/90', 'Gls/90', 'xG', 'Cr A', 'K Ps/90'],
    'LM/RM': ['Asts/90', 'Drb/90', 'Cr A', 'Ps C/90', 'xG'],
    'RW/LW': ['Gls/90', 'Cr A', 'Drb/90', 'xG', 'Shot/90'],
    'ST': ['Gls/90', 'xG', 'Shots', 'ShT %', 'PoM']
}

# --- Helper function to parse transfer value ---
def extract_value(val):
    if isinstance(val, str) and "$" in val:
        parts = val.replace("$", "").replace("M", "").replace("K", "").replace(",", "").split(" - ")
        try:
            return (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 else float(parts[0])
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
    'Segunda División': 0.845,
    'MLS': 0.840,
    'Brazilian Serie A': 0.835,
    'Argentine Primera División': 0.830,
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
def compute_moneyball_score(row, metrics):
    perf_score = 0
    weights = [1 / len(metrics)] * len(metrics)
    for metric, w in zip(metrics, weights):
        try:
            val = float(str(row[metric]).replace('%', '').replace(',', '').strip())
            perf_score += w * val
        except:
            continue

    try:
        value = extract_value(row['Transfer Value'])
        if not value:
            return 0
    except:
        return 0

    coeff = 1.0

    if 'Division' in row and isinstance(row['Division'], str):
        for league, factor in LEAGUE_COEFFICIENTS.items():
            if league.lower() in row['Division'].lower():
                coeff = factor
                break

    return (perf_score * coeff) / (value + 1)

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

    df = df[df['Age'] <= 22]  # Limit to max 22 years old players

    detected_position = None
    fm_position_map = {
        'GK': ['GK'],
        'CB': ['D(C)'],
        'RB/LB': ['D(R)', 'D(L)'],
        'RWB/LWB': ['WB(R)', 'WB(L)'],
        'CDM': ['DM'],
        'CM': ['M(C)'],
        'CAM': ['AM(C)'],
        'LM/RM': ['M(R)', 'M(L)'],
        'RW/LW': ['AM(R)', 'AM(L)'],
        'ST': ['ST(C)']
    }
    if 'Position' in df.columns:
        for key, fm_labels in fm_position_map.items():
            if df['Position'].astype(str).str.contains('|'.join(fm_labels), case=False).any():
                detected_position = key
                break
    position = st.selectbox("Select Position", list(POSITION_METRICS.keys()), index=list(POSITION_METRICS.keys()).index(detected_position) if detected_position else 0)
    metrics = POSITION_METRICS[position]
    st.markdown(f"**Evaluating {position}** using: {', '.join(metrics)}")

    df['Moneyball Score'] = df.apply(lambda row: compute_moneyball_score(row, metrics), axis=1)
    df = df.sort_values(by='Moneyball Score', ascending=False).reset_index(drop=True)

    top_n = st.slider("Number of top players to display", min_value=5, max_value=50, value=10)

    available_cols = df.columns.tolist()
    base_cols = [col for col in ['Name', 'Club', 'Division', 'Age', 'Salary', 'Transfer Value', 'Moneyball Score'] if col in available_cols]
    metric_cols = [col for col in metrics if col in available_cols]
    display_cols = base_cols + metric_cols
    st.subheader(f"Top {position} Players")
    st.dataframe(df[display_cols].head(top_n))

    st.subheader("Score Distribution")
    fig = px.bar(df.head(top_n), x='Name', y='Moneyball Score', color='Club', title=f'Top {position} by Moneyball Score')
    st.plotly_chart(fig)

    st.subheader("Value for Money")
    df['Numeric Value'] = df['Transfer Value'].apply(extract_value)
    scatter_df = df.dropna(subset=['Moneyball Score', 'Numeric Value'])
    fig2 = px.scatter(scatter_df.head(50), x='Numeric Value', y='Moneyball Score', color='Age', hover_data=['Name'],
                      title='Transfer Value vs. Performance')
    st.plotly_chart(fig2)

    # --- Radar Chart Comparison ---
    st.subheader("Compare Players via Radar Chart")
    selected_players = st.multiselect("Select up to 3 players for radar comparison", df['Name'].head(top_n).tolist())

    if selected_players:
        radar_df = df[df['Name'].isin(selected_players)][['Name'] + metrics].dropna()
        fig3 = go.Figure()
        for _, row in radar_df.iterrows():
            values = []
            for metric in metrics:
                try:
                    val = float(str(row[metric]).replace('%', '').replace(',', '').strip())
                except:
                    val = 0
                values.append(val)
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
