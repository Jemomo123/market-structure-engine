"""
Phase 1 configuration.

Nothing in this file talks to an exchange or does any analysis — it's pure
configuration so every other module can be reconfigured from one place.
"""

# --- Watchlist -------------------------------------------------------------
# Base assets only. Providers are responsible for turning these into the
# correct exchange-specific symbol (e.g. "BTC" -> "BTC/USDT").
WATCHLIST = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "TON",
    "AVAX",
    "LINK",
    "SUI",
]

# --- Timeframes --------------------------------------------------------
# Each timeframe is processed completely independently. No cross-timeframe
# merging happens in Phase 1.
TIMEFRAMES = ["5m", "15m", "1h"]

# --- Data providers ----------------------------------------------------
# Order matters: first entry is tried first, remaining entries are
# fallbacks, tried in order, only if all prior providers fail validation.
EXCHANGE_PRIORITY = ["OKX", "MEXC"]

# Quote currency used to build trading pairs from the watchlist.
QUOTE_CURRENCY = "USDT"

# How many candles to request per fetch. Needs to be comfortably larger
# than what swing detection + a 4-swing structure window will consume.
CANDLE_FETCH_LIMIT = 300

# Minimum number of valid candles required to accept a provider's response.
MIN_VALID_CANDLES = 100

# --- Swing detection -----------------------------------------------------
# N = number of candles on each side of a pivot required to confirm it as
# a swing high/low. Keep this configurable; Phase 1 default is 2.
SWING_N = 2

# --- Market state classification -----------------------------------------
# Number of most recent CONFIRMED swing events considered when classifying
# the current market state. This is an evaluation window, not a rule that
# by itself forces a particular state.
STRUCTURE_WINDOW = 4
