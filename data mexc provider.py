"""
MEXC Futures OHLCV provider. Used only when OKX fails validation.
"""

from typing import List

import ccxt

from data.base_provider import BaseProvider, ProviderError
from models.types import Candle


class MEXCProvider(BaseProvider):
    name = "MEXC"

    def __init__(self):
        # 'options': {'defaultType': 'swap'} routes ccxt to MEXC's futures
        # (perpetual swap) market rather than spot.
        self._client = ccxt.mexc({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        # MEXC perpetual futures symbol format in ccxt: "BTC/USDT:USDT"
        return f"{base_asset}/{quote}:{quote}"

    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        symbol = self.to_exchange_symbol(base_asset, "USDT")
        try:
            raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise ProviderError(f"MEXC fetch failed for {symbol} {timeframe}: {exc}") from exc

        if not raw:
            raise ProviderError(f"MEXC returned empty OHLCV for {symbol} {timeframe}")

        return [
            Candle(
                timestamp=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5],
            )
            for row in raw
        ]
