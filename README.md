# Moneyball Football Dashboard

An interactive Streamlit dashboard that applies a Moneyball-style approach to football scouting using real player data. Upload an HTML export from Football Manager and analyze player performance, market value, role suitability, and more.

---

## 🏢 Features

* **Auto-Detect Position:** Automatically identifies the primary position from the uploaded filename.
* **Custom Weighting:** Fine-tune performance metrics and adjust transfer value/league strength impact.
* **Moneyball Score:** Quantitative score representing value-for-money, factoring performance, reliability, age, and league quality.
* **Hidden Gems Detector:** Find underrated high-performing players with low transfer values.
* **Role Assignment:** Intelligent best-fit role suggestions for each position using FM role profiles.
* **Percentile Rankings:** Benchmark players across core metrics relative to peers.
* **Age Projections:** Forecast player potential at age 25.
* **Interactive Visualizations:** Scatter plots for market analysis and value discovery.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/moneyball-fm-dashboard.git
cd moneyball-fm-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run moneyball_dashboard.py
```

---

## 🗃️ Data Input

* Upload an `.html` file exported from Football Manager (via "Print" -> "Web Page").
* Make sure the file contains the position in the filename at the beginning (e.g., `GK_2023-10-01.html` for Goalkeepers).

---

## 🌐 Supported Positions

The dashboard supports scoring and profiling for:

* Goalkeeper (GK)
* Center-Back (CB)
* Full-Back / Wing-Back (RB/LB, RWB/LWB)
* Defensive / Central / Attacking Midfielders (CDM, CM, CAM)
* Wingers (LM/RM, RW/LW)
* Strikers (ST)

---

## ⚖️ Scoring Logic

* **Performance Score:** Weighted sum of key stats.
* **League Adjustment:** Normalized using predefined league coefficients.
* **Age Penalty:** Younger players get higher future upside.
* **Reliability Bonus:** Players with higher minutes are scored more favorably.
* **Value Factor:** Transfer value penalizes high-cost players.

---

## 🪙 Role Detection

Each player is evaluated against FM-style role profiles (e.g., Ball-Winning Midfielder, Inverted Winger, False Nine) using thresholds on key stats.

---

## 🏦 League Coefficients

Strength of competition is factored using coefficients derived from major and minor leagues. Lower-tier leagues are adjusted down to normalize performance.

---

## 📚 File Structure

* `app.py`: Main Streamlit dashboard
* `requirements.txt`: Python dependencies
* `README.md`: Project documentation

---

## 🚫 Disclaimer

This tool is intended for analytical and scouting purposes only. It is not affiliated with Football Manager, any clubs, or players.

---

## 🙌 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.

---

## 🚀 Future Improvements

* CSV export of role matches
* Multiple file upload
* Role performance clustering
* Dynamic radar chart for individual players

---

## 🌟 Author

Developed by Arthur Acker. For questions or feedback, please open an issue on GitHub.

---

## 📁 License

[MIT](LICENSE)
