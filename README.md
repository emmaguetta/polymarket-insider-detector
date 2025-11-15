# Polymarket Insider Trading Detector

Simplified AI-powered detection system for identifying suspicious trading patterns on Polymarket using Claude Haiku 4.5.

## Features

- **Real-time Monitoring**: Polls Polymarket every 20 seconds for new trades
- **Historical Analysis**: Batch analysis with configurable lookback period
- **3 Detection Algorithms**: Timing, Volume, and Win Rate detectors
- **AI Analysis**: Claude Haiku 4.5 provides contextual reasoning for flagged trades
- **SQLite Database**: Simple file-based storage, no server setup required

## Installation

```bash
# Clone the repository
cd polymarket-insider-detector/clean

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Configuration

Edit `.env` file:

```bash
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional (defaults provided)
DATABASE_URL=sqlite:///polymarket_insider.db
TIMING_THRESHOLD_HOURS=24
VOLUME_ANOMALY_THRESHOLD=2.0
WIN_RATE_THRESHOLD=0.75
```

## Usage

### Real-time Monitoring

Monitor live trades as they happen:

```bash
python realtime_detector.py
```

The script will:
- Bootstrap with last 30 days of data (first run only)
- Poll for new trades every 20 seconds
- Display alerts when suspicious trades detected
- Run Claude AI analysis on flagged trades

Press `Ctrl+C` to stop.

### Historical Analysis

Analyze past trades in batch:

```bash
# Last 30 days (default)
python historical_pipeline.py

# Last 60 days with custom threshold
python historical_pipeline.py --days-back 60 --threshold 0.1

# Last 6 months without AI analysis (faster)
python historical_pipeline.py --days-back 180 --no-llm

# See all options
python historical_pipeline.py --help
```

**Arguments**:
- `--days-back`: Number of days to analyze (default: 30)
- `--threshold`: Detection threshold 0.0-1.0 (default: 0.05)
- `--no-llm`: Skip Claude analysis for faster processing

## Detection System

### Prefiltering (3 Detectors)

1. **Timing Detector (40% weight)**
   - Flags trades close to market resolution
   - Exponential scoring based on hours before resolution

2. **Volume Detector (30% weight)**
   - Detects abnormally large trades
   - Uses statistical analysis (Z-score)
   - Flags new wallets with large first trades

3. **Gains Detector (30% weight)**
   - Identifies suspiciously high win rates (>75%)
   - Applies binomial statistical test
   - Requires minimum 5 trades

**Combined Score**: Weighted average of all detectors (0.0-1.0 scale)

### AI Analysis

Trades above threshold are analyzed by Claude Haiku 4.5:
- Contextual reasoning with market/wallet history
- Suspicion score (0-100)
- Confidence level (low/medium/high)
- Recommendation (legitimate/suspicious/manual_review)

## Database

SQLite database (`polymarket_insider.db`) stores:
- Markets (Polymarket questions and outcomes)
- Trades (Individual transactions)
- Wallets (Trader addresses and statistics)
- Suspicious Transactions (Flagged trades with AI analysis)

Initialize/reset database:

```bash
python init_database.py
```

## Architecture

```
Data API (20s polling) → Database → Detectors → Enrichment → Claude AI → Alerts
```

**Key Modules**:
- `database/`: SQLite models
- `data_ingestion/`: Polymarket API clients
- `prefiltering/`: Detection algorithms
- `enrichment/`: Context enrichment (wallet profiling, market analysis)
- `llm_analysis/`: Claude Haiku 4.5 integration

## Requirements

- Python 3.11+
- Anthropic API key ([get one here](https://console.anthropic.com/))
- Internet connection (for Polymarket API access)

## Notes

- No Polymarket authentication required (uses public endpoints)
- First run bootstraps with 30 days of historical data
- Detection thresholds can be adjusted in `.env`
- All data stored locally in SQLite database

## License

MIT
