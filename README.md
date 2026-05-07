# Paper Hoops

[中文说明](README.zh-CN.md)

<p align="center">
  <img src="assets/brand/paper-hoops-logo.png" alt="Paper Hoops" width="520">
</p>

Paper Hoops is an NBA roster-movement and trade-impact simulation website. It combines player and team season data from 1975-76 through 2025-26, uses the 2025-26 roster state as the current baseline, and projects how a trade may change a team's 2026-27 performance.

**Public site:** public website link coming soon

## Features

- **Trade simulation**: choose a target team, outgoing players, and incoming players, then generate a 2026-27 post-trade projection.
- **Team projection**: compare wins, win percentage, pace, offensive rating, defensive rating, and net rating before and after a simulated trade.
- **Player data**: browse 2025-26 player box-score data, shooting efficiency, salary fields, and impact metrics with sorting and pagination.
- **Team data**: review 2025-26 team records and efficiency metrics by conference.
- **Poster generation**: generate compact and full trade-result posters, preview them first, and download the PNG only if you choose to.
- **Mobile layout**: use a compact mobile trade workflow designed for smaller screens.
- **Chinese / English UI**: switch page text, dynamic labels, dialogs, and poster copy between Chinese and English.

![Trade simulator workflow](docs/readme/trade-simulator.svg)

![Player and team data tabs](docs/readme/data-tables.svg)

![Poster preview and download flow](docs/readme/poster-preview.svg)

## Tech Stack

- Python 3.11+
- Native `ThreadingHTTPServer` web service
- NumPy / pandas / scipy / scikit-learn
- Single-page HTML / CSS / JavaScript frontend
- Optional Docker runtime

Runtime inference uses exported NumPy bundles, so the web service does not need to import PyTorch. The public repository includes only the inference runtime, model parameter bundles, and minimal runtime data needed to run the website.

## Project Layout

```text
.
├── app/
│   ├── trade_simulator_server.py   # Web service and API
│   └── static/index.html           # Single-page frontend
├── assets/                         # Avatars, logo, salaries, rosters, impact metrics
├── code/                           # Inference runtime and NumPy inference artifacts
├── runtime_data/                   # Minimal public data snapshot used by the website
├── docs/                           # Deployment notes and README image assets
├── Dockerfile
├── requirements.txt
└── render.yaml
```

## Local Setup

### 1. Clone The Repository

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
```

### 2. Create A Python Environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start The Web Service

```bash
python app/trade_simulator_server.py
```

Default local URL:

```text
http://127.0.0.1:8765/
```

To use another port:

macOS / Linux:

```bash
PORT=8766 python app/trade_simulator_server.py
```

Windows PowerShell:

```powershell
$env:PORT="8766"
python app/trade_simulator_server.py
```

### 4. Verify The Service

```bash
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/api/teams
```

PowerShell:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8765/healthz'
Invoke-RestMethod 'http://127.0.0.1:8765/api/teams'
```

Expected behavior:

- `/healthz` returns service health information and the current season.
- `/api/teams` returns 30 teams.
- Opening `/` in a browser shows the homepage, trade simulator, player data, and team data.

## Docker

Build the image:

```bash
docker build -t paper-hoops .
```

Run the container:

```bash
docker run --rm -p 8765:8765 paper-hoops
```

Open:

```text
http://127.0.0.1:8765/
```

## API Overview

| Path | Method | Description |
| --- | --- | --- |
| `/` | GET | Frontend homepage |
| `/healthz` | GET | Health check |
| `/api/teams` | GET | Current-season team list |
| `/api/team_view?team=OKC` | GET | Team state, roster, and default outgoing players |
| `/api/players_by_team?team=LAL&exclude_team=OKC` | GET | Selectable incoming players by source team |
| `/api/player_stats` | GET | Player data table |
| `/api/team_stats` | GET | Team data table |
| `/api/simulate` | POST | Trade simulation |

Example simulation request:

```json
{
  "team": "OKC",
  "outgoing": ["Player A"],
  "incoming": ["Player B"]
}
```

## Data And Models

- The active application uses 2025-26 player and team data as the current state and projects 2026-27 outcomes.
- The public runtime data is stored under `runtime_data/`.
- Web inference depends on `code/outputs/**/weights.npz` and `preprocessing.pkl`.
- Training code, full historical datasets, experiment reports, and model-optimization details are not included in the public release.
- Sparse advanced metrics are hidden or rendered as blank values rather than filled with zero.

## Developer

- Developer: Yutian
- Hupu ID: 予田
- Xiaohongshu: 1670174606

## Disclaimer

This is a personal learning, research, and fan project. It is not affiliated with the NBA, NBA teams, or any official data provider. Team, player, salary, avatar, and statistical data may come from public sources or locally curated datasets. The project is intended for research and entertainment only and should not be treated as official records or decision-making advice.
