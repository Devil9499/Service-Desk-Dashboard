Markdown
# 📊 Service Desk Command Centre

An end-to-end, single-file Streamlit dashboard that unifies **ServiceNow** ITSM data and **Genesys** contact centre metrics into a real-time operational command centre. It features automated multi-file ingestion, comprehensive operational KPIs, self-service analytics, machine-learning demand forecasting, anomaly detection, and capacity planning.

---

## ✨ Features

* **Automated Data Processing & Classification**: Intelligently parses, normalises, and cleans raw CSV/Excel extracts (handling mixed date formats, title rows, and pivot tables automatically).
* **ServiceNow Analytics**: 
  * Incident lifecycle tracking (SLA achievement %, MTTR, FCR, aging queues, backlog trends, repeat incidents).
  * Request management (RITM catalogue item performance).
  * Change management success vs. failure rates.
* **Genesys Contact Centre Analytics**:
  * Voice metrics (Offered, Answered, Abandon %, ASA, AHT, Talk/Hold/ACW components, Transfers, Callbacks).
  * Chat metrics (IMS interactions, queue wait time, first response time, agent concurrency, time-outs, CSAT/Sentiment).
  * WFM Presence & Occupancy tracking across agents.
* **Self-Service & Deflection**: Analyzes portal adoption %, channel mix, top self-service categories, and identifies call/chat deflection opportunities.
* **Machine Learning Forecasting (scikit-learn)**:
  * Trains multiple regression models (*Gradient Boosting, Random Forest, Hist Gradient Boosting, Ridge*) using calendar, lag, and rolling features.
  * Evaluates performance on a hold-out test period (MAE, WAPE %, RMSE, R²).
  * Generates future demand predictions with 90% confidence intervals.
* **Anomaly Detection**: Flags unusual daily volume spikes using `IsolationForest`.
* **Capacity Planning Tool**: Calculates required FTE staffing per day based on forecast volume, target AHT, shrinkage, target occupancy, and productive agent hours.

---

## 🛠️ Installation & Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Required Dependencies
Install all required libraries using `pip`:

```bash
pip install streamlit pandas numpy scikit-learn plotly openpyxl xlrd
3. Run the Application
Launch the Streamlit app from your terminal:

Bash
streamlit run service_desk_dashboard.py
📄 File Ingestion & Auto-Detection
The app automatically recognizes and classifies files based on column headers and data signatures upon upload:

Data Type	Recognition Signature	Key Metrics Included
ServiceNow Incidents	Ticket number starting with INC	SLA %, FCR, MTTR, Backlog, Aging, Escalations
ServiceNow Requests	Ticket number starting with RITM or Catalogue name	Request volume, Catalogue distribution
ServiceNow Changes	Ticket number starting with CHG	Change outcome (Successful vs. Failed)
Genesys Interactions	Number starting with IMS	Chat volume, Wait time, Concurrency, Time-outs
Genesys Performance	Agent Name, Answered, Handle	ASA, AHT, Abandon %, Talk/Hold/ACW components
Genesys Presence	Agent, Logged In, On Queue	Agent occupancy %, On-queue %, Adherence
📌 Dashboard Navigation
Executive Dashboard: High-level combined view across tickets and calls/chats, queue load by assignment group, top ticket drivers, agent workload distribution, and demand forecast previews.

Self-Service Dashboard: Channel mix breakdown, weekly self-service adoption trends, portal MTTR vs. assisted channels, and ticket deflection analysis.

Call Data Dashboard: Detailed voice metrics, handle-time component breakdown, agent presence/occupancy analysis, and call volume heatmaps.

Chat Data Dashboard: Chat handle times, queue wait distributions, concurrency analysis per agent, and time-out tracking.

Forecast vs Actual Demand: Interactive ML model selection, feature importance ranking, anomaly detection, and automated FTE capacity planning.

📦 Requirements (requirements.txt)
If deploying to Streamlit Community Cloud or another hosting platform, create a requirements.txt file with the following:

Plaintext
streamlit
pandas
numpy
scikit-learn
plotly
openpyxl
xlrd
