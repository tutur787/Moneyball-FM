# ⚽ Moneyball Football Dashboard

A data-driven scouting tool inspired by the *Moneyball* philosophy, designed to identify undervalued football talent across global leagues using real match statistics. Built with **Python**, **Streamlit**, and **Plotly**, this dashboard parses Football Manager export files to rank players based on performance vs market value.

---

## 🚀 Features

* 📅 Upload Football Manager HTML player exports
* 🔍 Auto-detect and manually select player positions (e.g., CB, CDM, ST, etc.)
* 📊 Position-specific performance metrics used for ranking
* 🌍 League-strength adjustment via coefficient scaling (350+ leagues)
* 🧠 Calculates "Moneyball Score" = performance ÷ market value
* 📈 Visualizations:

  * Bar charts of top players
  * Scatter plots of value vs. performance
  * Radar chart comparisons for up to 3 players
* 📄 Export ranked players as CSV

---

## 📂 Input Format

Upload an **HTML file** exported from **Football Manager** that includes a table of player stats. The file should contain at minimum the following columns:

* `Name`
* `Position`
* `Club`
* `Division`
* `Transfer Value`
* `Age`
* And various performance metrics depending on position

All uploaded players are assumed to be **22 years old or younger**.

---

## 🧶 How the Moneyball Score Works

```
Moneyball Score = (Weighted Performance Metrics × League Coefficient) / (Transfer Value + 1)
```

* Each position uses its most relevant KPIs:

  * Example: `Gls/90`, `xG`, `Shots` for strikers
  * `Tck/90`, `Hdr %`, `Clear` for center backs
* League coefficients adjust scores based on difficulty level:

  * Example: Premier League = 1.00, Serie B = 0.86, MLS = 0.84, etc.

---

## 📊 Visualizations

* **Top Players by Moneyball Score** – sortable data table
* **Score Distribution** – bar chart of top N players
* **Value for Money** – scatter plot of Score vs. Transfer Value
* **Radar Comparison** – choose up to 3 players to compare across key metrics

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/moneyball-dashboard.git
cd moneyball-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run moneyball_dashboard.py
```

---

## 📋 Requirements

* Python 3.8 or newer
* Streamlit
* Pandas
* Plotly

---

## 📝 Notes

* League coefficients are automatically applied using a comprehensive FM-based tier system (\~350+ leagues).
* Radar charts auto-scale metrics and support mixed stat types.
* If a division is not matched to a known league, a fallback coefficient is used.

---

## 🖼️ Screenshots

To be added:

* Top Players Table
* Scatter Plot
* Radar Comparison

---

## 📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and share with attribution.

---

## 👤 Author

Developed by **Arthur Acker**
Inspired by *Moneyball* principles and optimized for Football Manager scouting data.
