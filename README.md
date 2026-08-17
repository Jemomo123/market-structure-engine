# Market State Detector — Phase 1

Classifies each coin/timeframe into BULLISH / BEARISH / RANGING / TRANSITION
using pure price-structure analysis (swing highs/lows -> HH/HL/LH/LL).
No indicators, no scoring, no signals — structure only.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Pipeline

```
OHLCV (OKX, fallback MEXC Futures)
   -> Swing Detection (confirmed swings only, N=2 default)
   -> Market Structure (HH / HL / LH / LL, chronological)
   -> Market State (BULLISH / BEARISH / RANGING / TRANSITION)
   -> Report (coin | timeframe | state | source)
```

## Config (`config/settings.py`)

- `WATCHLIST` — the 10 tracked coins
- `TIMEFRAMES` — 5m, 15m, 1h (evaluated independently, no merged verdict)
- `SWING_N` — candles required on each side to confirm a swing (default 2)
- `STRUCTURE_WINDOW` — number of recent confirmed structure events considered
  when classifying state (default 4)
- `EXCHANGE_PRIORITY` — OKX tried first, MEXC Futures as fallback

## Data integrity

- Only OKX and MEXC Futures are used. If both fail or return invalid data,
  that coin/timeframe is reported as `NO DATA` — never fake or filled in.
- Every result is tagged with its data source, e.g. `Source: OKX` or
  `Source: MEXC (fallback)`.

## Explicitly out of scope for Phase 1

Range boundaries, breakout detection, retest detection, trading signals,
CVD, open interest, funding rates, whale tracking, indicators, scoring,
predictions, and any multi-timeframe merged verdict.
