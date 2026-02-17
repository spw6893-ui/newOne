"""
批量下载 Binance Vision 归档数据（按 symbol 逐个处理，减少峰值空间占用）。

默认行为：仅下载 U 本位（um）永续的 `metrics`。

说明：
- `liquidationSnapshot` 经实测主要只在币本位（cm）存在，且符号格式通常为 `BTCUSD_PERP`；
  因此这里不默认批量下载强平流，避免大面积 404 误报“失败”。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from download_binance_efficient import download_symbol_range


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_symbols_file() -> str:
    # 让脚本在任意工作目录下都能找到默认 symbols 文件
    return str(Path(__file__).resolve().parent / "top100_perp_symbols.txt")


def load_symbols(path: str, market: str) -> list[str]:
    """
    - market=um：允许 CCXT 格式 `BTC/USDT:USDT`，会转成 Binance Vision 归档所需的 `BTCUSDT`
    - market=cm：建议直接提供 `BTCUSD_PERP` 这类币本位合约符号，不做转换
    """
    symbols: list[str] = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if market == "um":
                s = s.replace("/USDT:USDT", "USDT")
                s = s.replace("/USDT", "USDT")
                s = s.replace("_", "")
            symbols.append(s)
    return symbols


def _default_output_dir(market: str, data_type: str, interval: str | None) -> str:
    if market == "um" and data_type == "metrics":
        return "AlphaQCM_data/binance_metrics"
    if market == "um" and data_type == "fundingRate":
        return "AlphaQCM_data/binance_fundingRate"
    if market == "um" and data_type == "aggTrades":
        return "AlphaQCM_data/binance_aggTrades"
    if market == "um" and data_type in {"klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"}:
        return f"AlphaQCM_data/binance_{data_type}_{interval or '1h'}"
    if market == "cm" and data_type == "liquidationSnapshot":
        return "AlphaQCM_data/binance_liquidations_cm"
    return f"AlphaQCM_data/binance_{market}_{data_type}"


def parse_data_type_token(token: str) -> tuple[str, str | None]:
    """
    支持两种写法：
    - metrics
    - klines:1h
    """
    if ":" not in token:
        return token, None
    name, interval = token.split(":", 1)
    name = name.strip()
    interval = interval.strip()
    return name, (interval or None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量下载 Binance Vision 归档数据并按小时聚合")
    parser.add_argument("--symbols-file", default=_default_symbols_file())
    parser.add_argument("--symbols-offset", type=int, default=0, help="从第几个 symbol 开始（用于分批跑）")
    parser.add_argument("--symbols-limit", type=int, default=0, help="最多跑多少个 symbol（0=不限制）")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--market", default="um", choices=["um", "cm"])
    parser.add_argument(
        "--data-types",
        default="metrics",
        help="逗号分隔，例如 metrics 或 metrics,aggTrades 或 liquidationSnapshot",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="可选：统一输出根目录（会在其下按 data_type 建子目录）。不传则用默认目录。",
    )
    parser.add_argument("--temp-dir", default="AlphaQCM_data/_tmp/binance_vision")
    parser.add_argument("--skip-existing", action="store_true", help="若输出文件已存在且覆盖到结束日期则跳过")
    parser.add_argument(
        "--aggtrades-1m-dir",
        default="",
        help="可选：aggTrades 的 1m 中间层 Parquet 输出目录（仅 data-type=aggTrades 时使用）",
    )
    parser.add_argument(
        "--aggtrades-keep-zip",
        action="store_true",
        help="仅 data-type=aggTrades：保留下载的月度 zip（默认处理后删除以节省空间）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    data_type_tokens = [x.strip() for x in args.data_types.split(",") if x.strip()]
    parsed_types = [parse_data_type_token(t) for t in data_type_tokens]

    symbols = load_symbols(args.symbols_file, market=args.market)
    if args.symbols_offset < 0:
        raise ValueError("--symbols-offset 不能为负数")
    if args.symbols_limit < 0:
        raise ValueError("--symbols-limit 不能为负数")
    if args.symbols_offset:
        symbols = symbols[args.symbols_offset :]
    if args.symbols_limit:
        symbols = symbols[: args.symbols_limit]

    print(f"Downloading {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    print(f"Market: {args.market}")
    print(f"Data types: {', '.join(data_type_tokens)}")
    print()

    if args.dry_run:
        for dt, interval in parsed_types:
            label = f"{dt}:{interval}" if interval else dt
            out_dir = (
                f"{args.output_root}/{label}".rstrip("/")
                if args.output_root
                else _default_output_dir(args.market, dt, interval)
            )
            print(f"- {label} -> {out_dir}")
        return 0

    completed: dict[str, list[str]] = {}
    for dt, interval in parsed_types:
        label = f"{dt}:{interval}" if interval else dt
        completed[label] = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        for dt, interval in parsed_types:
            label = f"{dt}:{interval}" if interval else dt
            out_dir = (
                f"{args.output_root}/{label}".rstrip("/")
                if args.output_root
                else _default_output_dir(args.market, dt, interval)
            )
            ok = download_symbol_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_type=dt,
                output_dir=out_dir,
                temp_dir=args.temp_dir,
                market=args.market,
                interval=interval,
                skip_existing=args.skip_existing,
                aggtrades_1m_dir=(args.aggtrades_1m_dir or None),
                aggtrades_keep_zip=args.aggtrades_keep_zip,
            )
            if ok:
                completed[label].append(symbol)
        print()

    print("\nCompleted:")
    for label in completed:
        print(f"  {label}: {len(completed[label])}/{len(symbols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
