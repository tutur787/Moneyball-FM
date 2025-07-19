import re
from matplotlib import cm
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

# --- Moneyball score computation ---
def compute_moneyball_score(row, metric_weights, value_weight=0.3, league_weight=0.3):
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
        matched = False
        for league, factor in LEAGUE_COEFFICIENTS.items():
            if league.lower() in row['Division'].lower():
                coeff = factor
                matched = True
                break
        if not matched:
            coeff = LEAGUE_COEFFICIENTS.get('Other Leagues', 0.6225)  # default fallback
    else:
        coeff = LEAGUE_COEFFICIENTS.get('Other Leagues', 0.6225)

    coeff = (1 - league_weight) * 1.0 + league_weight * coeff

    age_penalty = 1 + (row['Age'] - 18) * 0.02 if 'Age' in row else 1

    mins_raw = str(row['Mins']).strip() if 'Mins' in row else '0'
    try:
        minutes_played = float(mins_raw.replace(',', '')) if mins_raw != '-' else 0
    except:
        minutes_played = 0

    reliability_bonus = min(1.0, minutes_played / 2000)

    value_factor = 1.0 if value_weight == 0 else 1 + ((value / 1_000_000) * value_weight)
    score = ((perf_score / total_weight) * coeff * reliability_bonus) / (value_factor * age_penalty)
    return round(score, 3)

def project_moneyball_score_at_age(row, target_age=25, metric_weights=None):
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
    tags = []
    
    def val(metric):
        try:
            raw = str(row[metric]).replace('%', '').replace(',', '').strip()
            return float(raw) if raw != '-' and raw else 0
        except:
            return 0

    if position in ['CM', 'CAM', 'LM/RM', 'RW/LW']:
        if val('Asts/90') > 0.3 or val('Cr A') > 1.0:
            tags.append("Creator")
        if val('Drb/90') > 2.0:
            tags.append("Dribbler")
        if val('Gls/90') > 0.3 or val('xG') > 3:
            tags.append("Finisher")

    if position in ['CDM', 'CB', 'RB/LB', 'RWB/LWB']:
        if val('Tck/90') > 2.5 or val('K Tck/90') > 1.0:
            tags.append("Ball-Winner")
        if val('Int/90') > 2.0:
            tags.append("Interceptor")
        if val('Clear') > 50:
            tags.append("Defender")

    if position == 'ST':
        if val('Gls/90') > 0.5:
            tags.append("Poacher")
        if val('ShT %') > 50:
            tags.append("Clinical")
        if val('Drb/90') > 1.5:
            tags.append("Mobile Striker")
    
    if 'Av Rat' in metrics and val('Av Rat') >= 7.5:
        tags.append("Performer")

    return ", ".join(tags) if tags else "Balanced"

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

    # --- Nationality Filter ---
    if 'Nat' in df.columns:
        nat_options = sorted(df['Nat'].dropna().unique())
        selected_nations = st.sidebar.multiselect("Filter by Nationality", options=nat_options)
        if selected_nations:
            df = df[df['Nat'].isin(selected_nations)]

    # --- League/Division Filter ---
    if 'Division' in df.columns:
        league_options = sorted(df['Division'].dropna().unique())
        selected_leagues = st.sidebar.multiselect("Filter by League/Division", options=league_options)
        if selected_leagues:
            df = df[df['Division'].isin(selected_leagues)]

    st.sidebar.markdown("---")
    show_percentiles = st.sidebar.checkbox("Show Percentile Rankings", value=False)

    # Apply filters (convert inputs back to base unit)
    df = df[(df['Age'] >= age_lower) & (df['Age'] <= age_upper)]
    df = df[df['Numeric Value'].between(val_lower * 1_000_000, val_upper * 1_000_000, inclusive='both')]

    # --- Displaying Data ---
    position = st.selectbox("Select Position", list(POSITION_METRICS.keys()), index=list(POSITION_METRICS.keys()).index(detected_position) if detected_position else 0)
    metrics = POSITION_METRICS[position]
    # --- Compute positional benchmarks ---
    positional_df = df.copy()
    benchmarks = {}
    for metric in metrics:
        try:
            positional_values = pd.to_numeric(positional_df[metric].replace('-', None), errors='coerce')
            benchmarks[metric] = {
                'mean': positional_values.mean(),
                'median': positional_values.median(),
                '80th': positional_values.quantile(0.80)
            }
        except:
            benchmarks[metric] = {'mean': None, 'median': None, '80th': None}

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
    value_impact = st.sidebar.slider(
        "Transfer Value Impact", 0.0, 1.0, value=0.3, step=0.01
    )
    league_impact = st.sidebar.slider(
        "League Coefficient Impact", 0.0, 1.0, value=0.3, step=0.01
    )

    # Normalize the custom weights
    total = sum(weight_inputs.values()) or 1  # avoid div by zero
    normalized_weights = {k: v / total for k, v in weight_inputs.items()}

    # Compute Percentile Columns
    for metric in metrics:
        col_name = f"{metric} Percentile"
        try:
            df[col_name] = pd.to_numeric(df[metric].astype(str).str.replace('%', '').str.replace(',', ''), errors='coerce')
            df[col_name] = df[col_name].rank(pct=True) * 100
            df[col_name] = df[col_name].round(1)
        except:
            df[col_name] = None

    st.markdown(f"**Evaluating {position}** using: {', '.join(metrics)}")

    df['Moneyball Score'] = df.apply(
        lambda row: compute_moneyball_score(row, normalized_weights, value_impact, league_impact),
        axis=1
    )
    df = df.sort_values(by='Moneyball Score', ascending=False).reset_index(drop=True)
    df['Projected Score (25)'] = df.apply(
        lambda row: project_moneyball_score_at_age(row, target_age=25, metric_weights=metrics), axis=1
    )
    df['Style Tags'] = df.apply(lambda row: assign_player_tags(row, position, metrics), axis=1)
    selected_tags = st.sidebar.multiselect("Filter by Style Tags", options=df['Style Tags'].unique())
    if selected_tags:
        df = df[df['Style Tags'].isin(selected_tags)]

    top_n = 10

    available_cols = df.columns.tolist()
    base_cols = [col for col in ['Name', 'Club', 'Division', 'Age', 'Salary', 'Transfer Value', 'Apps', 'Moneyball Score', 'Style Tags'] if col in available_cols]
    metric_cols = [col for col in metrics if col in available_cols]

    if show_percentiles:
        percentile_cols = [f"{m} Percentile" for m in metrics if f"{m} Percentile" in df.columns]
        display_cols = base_cols + percentile_cols
    else:
        metric_cols = [col for col in metrics if col in df.columns]
        display_cols = base_cols + metric_cols

    st.subheader(f"Top {position} Players")

    if show_percentiles:
        percentile_cols = [f"{m} Percentile" for m in metrics if f"{m} Percentile" in df.columns]
        display_cols = base_cols + percentile_cols

        def color_text_by_percentile(val):
            try:
                norm_val = float(val) / 100
                rgba = cm.RdYlGn(norm_val)
                r, g, b = [int(255 * x) for x in rgba[:3]]
                return f'color: rgb({r},{g},{b})'
            except:
                return ''

        styled_df = df[display_cols].head(top_n).style\
            .applymap(color_text_by_percentile, subset=percentile_cols)\
            .format("{:.1f}", subset=percentile_cols)

        st.dataframe(styled_df)
    else:
        metric_cols = [col for col in metrics if col in df.columns]
        display_cols = base_cols + metric_cols
        st.dataframe(df[display_cols].head(top_n))

    st.subheader(f"{position} Benchmarks")
    bm_df = pd.DataFrame(benchmarks).T.rename(columns={
        'mean': 'Mean',
        'median': 'Median',
        '80th': '80th Percentile'
    }).dropna()

    st.dataframe(bm_df.style.format("{:.2f}"))

    st.subheader("Value for Money")
    scatter_df = df.dropna(subset=['Moneyball Score', 'Numeric Value'])
    fig2 = px.scatter(scatter_df.head(50), x='Numeric Value', y='Moneyball Score', color='Age', hover_data=['Name', 'Club', 'Division'],
                  title='Transfer Value vs. Moneyball Score', labels={'Numeric Value': 'Transfer Value (M)', 'Moneyball Score': 'Performance-to-Value Ratio'})
    fig2.update_traces(marker=dict(size=12, opacity=0.7))
    fig2.update_layout(height=600)
    st.plotly_chart(fig2)

    st.download_button("Download Ranked Players CSV", df[display_cols].to_csv(index=False), file_name="ranked_players.csv")

    # --- Hidden Gems Detection (Refined) ---
    st.subheader("🔍 Hidden Gems: High Performers, Low Cost")

    # Exclude rows with invalid scores or missing values
    valid_df = df[(df['Moneyball Score'] > 0) & (df['Numeric Value'].notnull())]

    # Compute value and score percentiles on valid subset
    valid_df['Value Percentile'] = valid_df['Numeric Value'].rank(pct=True)
    valid_df['Score Percentile'] = valid_df['Moneyball Score'].rank(pct=True)

    # Hidden gems = low market value (bottom 40%) & high score (top 30%)
    gems_df = valid_df[(valid_df['Value Percentile'] <= 0.4) & (valid_df['Score Percentile'] >= 0.7)]

    if not gems_df.empty:
        st.markdown(f"Found **{len(gems_df)} hidden gems**. These players have high performance scores relative to their low market value.")

        if show_percentiles:
            gem_cols = [col for col in display_cols if 'Percentile' in col]
            styled_gems = gems_df[display_cols].head(10).style\
                .applymap(color_text_by_percentile, subset=gem_cols)\
                .format("{:.1f}", subset=gem_cols)
            st.dataframe(styled_gems)
        else:
            st.dataframe(gems_df[display_cols].head(10))

        fig3 = px.scatter(gems_df, x='Numeric Value', y='Moneyball Score', hover_data=['Name', 'Club'],
                        color='Score Percentile', color_continuous_scale=px.colors.sequential.Viridis,
                        title='Hidden Gems Highlighted in Market',
                        labels={'Numeric Value': 'Transfer Value (M)', 'Moneyball Score': 'Performance-to-Value Ratio', 'Score Percentile': 'Performance Score Percentile'})
        fig3.update_traces(marker=dict(size=10, opacity=0.6))
        fig3.update_layout(legend_title_text='Hidden Gem', height=500)
        st.plotly_chart(fig3)
    else:
        st.info("No hidden gems matched the current filters.")

    st.subheader("📈 Age Projection (Peak at 25)")

    growth_df = df.copy()
    growth_df['Score Growth %'] = 100 * (growth_df['Projected Score (25)'] - growth_df['Moneyball Score']) / growth_df['Moneyball Score']
    growth_df = growth_df.sort_values(by='Projected Score (25)', ascending=False)

    cols_to_show = ['Name', 'Age', 'Moneyball Score', 'Projected Score (25)', 'Score Growth %']
    st.dataframe(
        growth_df[cols_to_show].head(10).style.format({
            'Moneyball Score': '{:.2f}',
            'Projected Score (25)': '{:.2f}',
            'Score Growth %': '{:.1f}'
        })
    )

else:
    st.info("Upload an HTML file containing a table of players with stats.")
