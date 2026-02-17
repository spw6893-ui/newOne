"""
高效下载并处理 Binance Vision 历史归档数据（按天下载、按小时聚合、删除原始数据节省空间）。

核心策略：
1) 下载单日 ZIP
2) 解压得到 CSV
3) 立刻聚合到小时级
4) 删除 CSV（以及 ZIP）
5) 继续下一天

说明与坑位：
- `metrics` 在 `futures/um`（U 本位永续）可用（目前仅日度归档）。
- `aggTrades`、`klines/markPriceKlines/indexPriceKlines/premiumIndexKlines`、`fundingRate` 在 `futures/um` 提供月度归档（强烈建议用月度，避免按天下载的请求量爆炸）。
- `liquidationSnapshot` 经实测仅在 `futures/cm`（币本位合约）可用；U 本位很多日期/币种会 404。
  因此如果你想下历史强平流，请显式传 `--market cm` 并使用币本位合约符号（例如 `BTCUSD_PERP`）。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Iterable, Optional, Union

import pandas as pd
import requests


_BINANCE_VISION_BASE = "https://data.binance.vision/data/futures"


@dataclass(frozen=True)
class DownloadConfig:
    market: str  # um / cm
    timeout_sec: int = 60
    retries: int = 3
    backoff_sec: float = 1.5
    chunk_bytes: int = 1024 * 1024


def _alphaqcm_root() -> Path:
    # AlphaQCM/data_collection/download_binance_efficient.py -> AlphaQCM/
    return Path(__file__).resolve().parents[1]


def _maybe_rewrite_alphaqcm_data_path(path: str) -> str:
    """
    兼容两种运行方式：
    - 在 AlphaQCM/ 目录运行：相对路径 `AlphaQCM_data/...` 正常
    - 在仓库根目录运行：将 `AlphaQCM_data/...` 自动映射到 `AlphaQCM/AlphaQCM_data/...`
    """
    p = Path(path)
    if p.is_absolute():
        return path

    if str(p).startswith("AlphaQCM_data") and not Path("AlphaQCM_data").exists():
        candidate = _alphaqcm_root() / p
        return str(candidate)

    return path


def _build_daily_zip_url(symbol: str, date: datetime, data_type: str, market: str) -> str:
    date_str = date.strftime("%Y-%m-%d")
    return f"{_BINANCE_VISION_BASE}/{market}/daily/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip"


def _build_daily_kline_zip_url(symbol: str, date: datetime, kline_type: str, interval: str, market: str) -> str:
    date_str = date.strftime("%Y-%m-%d")
    # 例如：
    # https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-02-01.zip
    return f"{_BINANCE_VISION_BASE}/{market}/daily/{kline_type}/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"


def _build_monthly_zip_url(symbol: str, month: datetime, data_type: str, market: str) -> str:
    month_str = month.strftime("%Y-%m")
    # 例如：
    # https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-02.zip
    return f"{_BINANCE_VISION_BASE}/{market}/monthly/{data_type}/{symbol}/{symbol}-{data_type}-{month_str}.zip"


def _build_monthly_kline_zip_url(symbol: str, month: datetime, kline_type: str, interval: str, market: str) -> str:
    month_str = month.strftime("%Y-%m")
    # 例如：
    # https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-02.zip
    return f"{_BINANCE_VISION_BASE}/{market}/monthly/{kline_type}/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"


def _download_stream_to_file(
    session: requests.Session,
    url: str,
    dest_path: Path,
    cfg: DownloadConfig,
) -> bool:
    """
    返回 True 表示下载成功；False 表示明确不存在（404）或最终重试失败。
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点/重启友好：
    # - 若 dest_path 已存在且可用：直接复用，避免重复下载
    # - 若临时 .part 已存在且其实已是完整 ZIP：原子搬运成 dest_path 继续后续处理
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if dest_path.exists() and dest_path.stat().st_size > 0:
        if dest_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(dest_path, "r") as zf:
                    if zf.namelist():
                        return True
            except Exception:
                try:
                    dest_path.unlink(missing_ok=True)
                except TypeError:
                    if dest_path.exists():
                        dest_path.unlink()
        else:
            return True

    if tmp_path.exists() and tmp_path.stat().st_size > 0 and dest_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                if zf.namelist():
                    os.replace(tmp_path, dest_path)
                    return True
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except TypeError:
                if tmp_path.exists():
                    tmp_path.unlink()

    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try:
            with session.get(url, stream=True, timeout=cfg.timeout_sec) as resp:
                if resp.status_code == 404:
                    return False
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=cfg.chunk_bytes):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, dest_path)
                return True
        except Exception as e:
            last_error = e
            if attempt < cfg.retries:
                time.sleep(cfg.backoff_sec * attempt)

    if last_error is not None:
        print(f"  下载失败: {url} ({last_error})")
    return False


def _head_url_exists(session: requests.Session, url: str, cfg: DownloadConfig) -> Optional[bool]:
    """
    轻量探测：仅判断 URL 是否存在。
    - 返回 True：存在（200）
    - 返回 False：明确不存在（404）
    - 返回 None：其它错误/超时（此时调用方应降级为“不跳过”，避免误判）
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try:
            resp = session.head(url, timeout=cfg.timeout_sec, allow_redirects=True)
            try:
                if resp.status_code == 404:
                    return False
                if resp.status_code == 200:
                    return True
                # 其它状态码不要当成“确实不存在”
                return None
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
        except Exception as e:
            last_error = e
            if attempt < cfg.retries:
                time.sleep(cfg.backoff_sec * attempt)
    _ = last_error
    return None


def _build_monthly_zip_url_any(
    symbol: str, month: datetime, data_type: str, interval: Optional[str], market: str
) -> str:
    if interval:
        return _build_monthly_kline_zip_url(
            symbol=symbol, month=month, kline_type=data_type, interval=interval, market=market
        )
    return _build_monthly_zip_url(symbol=symbol, month=month, data_type=data_type, market=market)


def _build_daily_zip_url_any(
    symbol: str, date: datetime, data_type: str, interval: Optional[str], market: str
) -> str:
    if interval:
        return _build_daily_kline_zip_url(
            symbol=symbol, date=date, kline_type=data_type, interval=interval, market=market
        )
    return _build_daily_zip_url(symbol=symbol, date=date, data_type=data_type, market=market)


def _find_first_available_day(
    session: requests.Session,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_type: str,
    interval: Optional[str],
    cfg: DownloadConfig,
) -> Optional[datetime]:
    """
    针对“从很早开始一路 404（新币种/数据类型晚开放）”的情况，快速定位首个存在的日期，减少无效请求。

    只在明确 404 时跳过；遇到其它异常会降级为“返回起始日期”，确保不会误跳过真实数据。
    """
    if start_date > end_date:
        return None

    def exists(day_dt: datetime) -> Optional[bool]:
        url = _build_daily_zip_url_any(
            symbol=symbol, date=day_dt, data_type=data_type, interval=interval, market=cfg.market
        )
        return _head_url_exists(session, url, cfg)

    first = exists(start_date)
    if first is None:
        return start_date
    if first is True:
        return start_date

    # 指数探测找到第一个“存在”的上界（按月步进，避免日级爆炸）
    last_missing = start_date
    step_days = 30
    probe = start_date
    upper: Optional[datetime] = None
    while True:
        probe = probe + timedelta(days=step_days)
        if probe > end_date:
            break
        r = exists(probe)
        if r is None:
            return start_date
        if r is True:
            upper = probe
            break
        last_missing = probe
        step_days *= 2

    if upper is None:
        return None

    # 二分定位最早存在日期（在 last_missing..upper 之间）
    lo = last_missing
    hi = upper
    ans = upper
    while lo <= hi:
        mid = lo + timedelta(days=((hi - lo).days // 2))
        r = exists(mid)
        if r is None:
            return start_date
        if r is True:
            ans = mid
            hi = mid - timedelta(days=1)
        else:
            lo = mid + timedelta(days=1)
    return ans


def _read_csv_first_timestamp(csv_path: Path) -> Optional[pd.Timestamp]:
    """
    快速读取输出 CSV 的第一条有效时间戳（不全量加载）。
    约定：第一列为时间戳（由 DataFrame index 写出）。
    """
    try:
        with open(csv_path, "rb") as f:
            head = f.read(16384).splitlines()
        for raw in head:
            if not raw.strip():
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            ts_text = (line.split(",", 1)[0] or "").strip().strip('"')
            ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
            if ts is not None and pd.notna(ts):
                return ts
    except Exception:
        return None
    return None


def _read_csv_last_timestamp(csv_path: Path) -> Optional[pd.Timestamp]:
    """
    快速读取输出 CSV 的最后一条有效时间戳（不全量加载）。
    约定：第一列为时间戳（由 DataFrame index 写出）。
    """
    try:
        last_ts = None
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - 16384, 0), os.SEEK_SET)
            for raw in reversed(f.read().splitlines()):
                if not raw.strip():
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                ts_text = (line.split(",", 1)[0] or "").strip().strip('"')
                last_ts = pd.to_datetime(ts_text, utc=True, errors="coerce")
                break
        if last_ts is not None and pd.notna(last_ts):
            return last_ts
    except Exception:
        return None
    return None


def _find_first_available_month(
    session: requests.Session,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_type: str,
    interval: Optional[str],
    cfg: DownloadConfig,
) -> Optional[datetime]:
    """
    针对“新币种从 2020 开始一路 404”的情况，快速定位首个存在的月份，减少无效请求。

    只在明确 404 时跳过；遇到其它异常会降级为“返回起始月份”，确保不会误跳过真实数据。
    """
    start_month = datetime(start_date.year, start_date.month, 1)
    months: list[datetime] = []
    m = start_month
    last = datetime(end_date.year, end_date.month, 1)
    while m <= last:
        months.append(m)
        m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
    if not months:
        return start_month

    def exists(month_dt: datetime) -> Optional[bool]:
        url = _build_monthly_zip_url_any(
            symbol=symbol, month=month_dt, data_type=data_type, interval=interval, market=cfg.market
        )
        return _head_url_exists(session, url, cfg)

    first = exists(months[0])
    if first is None:
        return start_month
    if first is True:
        return months[0]

    # 指数探测找到第一个“存在”的上界
    last_missing = 0
    step = 1
    idx = 0
    while True:
        idx = min(last_missing + step, len(months) - 1)
        r = exists(months[idx])
        if r is None:
            return start_month
        if r is True:
            break
        last_missing = idx
        if idx == len(months) - 1:
            return None
        step *= 2

    # 二分定位最早存在月份
    lo = last_missing + 1
    hi = idx
    ans = months[hi]
    while lo <= hi:
        mid = (lo + hi) // 2
        r = exists(months[mid])
        if r is None:
            return start_month
        if r is True:
            ans = months[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def _safe_extract_single_csv(zip_path: Path, dest_dir: Path) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                return None
            if len(members) > 1:
                # 目前 Binance 归档文件通常只有一个 CSV；多文件先按第一个处理并提示。
                print(f"  警告: ZIP 内 CSV 文件数 > 1，将仅处理第一个: {zip_path.name}")

            member = members[0]
            # 防 ZipSlip：拒绝绝对路径或包含 .. 的路径
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"ZIP 文件路径不安全: {member}")

            out_path = dest_dir / member_path.name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            return out_path
    except Exception as e:
        print(f"  解压失败: {zip_path} ({e})")
        return None


def download_day_zip_to_csv(
    symbol: str,
    date: datetime,
    data_type: str,
    interval: Optional[str],
    temp_dir: str,
    cfg: DownloadConfig,
    session: Optional[requests.Session] = None,
) -> Optional[Path]:
    """
    下载并解压单日数据，返回 CSV 路径；失败返回 None。
    """
    temp_dir = _maybe_rewrite_alphaqcm_data_path(temp_dir)
    tmp_dir = Path(temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if interval:
        url = _build_daily_kline_zip_url(
            symbol=symbol,
            date=date,
            kline_type=data_type,
            interval=interval,
            market=cfg.market,
        )
    else:
        url = _build_daily_zip_url(symbol=symbol, date=date, data_type=data_type, market=cfg.market)
    date_str = date.strftime("%Y-%m-%d")
    interval_part = f"_{interval}" if interval else ""
    zip_path = tmp_dir / f"{symbol}_{data_type}{interval_part}_{date_str}.zip"

    sess = session or requests.Session()
    ok = _download_stream_to_file(sess, url, zip_path, cfg)
    if not ok:
        return None

    csv_path = _safe_extract_single_csv(zip_path, tmp_dir)
    try:
        zip_path.unlink(missing_ok=True)
    except TypeError:
        if zip_path.exists():
            zip_path.unlink()

    return csv_path


def download_month_zip_to_csv(
    symbol: str,
    month: datetime,
    data_type: str,
    interval: Optional[str],
    temp_dir: str,
    cfg: DownloadConfig,
    session: Optional[requests.Session] = None,
) -> Optional[Path]:
    """
    下载并解压单月数据，返回 CSV 路径；失败返回 None。

    适用：
    - fundingRate（um）
    - aggTrades（um）
    - K线类（klines/markPriceKlines/indexPriceKlines/premiumIndexKlines）
    """
    temp_dir = _maybe_rewrite_alphaqcm_data_path(temp_dir)
    tmp_dir = Path(temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if interval:
        url = _build_monthly_kline_zip_url(
            symbol=symbol,
            month=month,
            kline_type=data_type,
            interval=interval,
            market=cfg.market,
        )
    else:
        url = _build_monthly_zip_url(symbol=symbol, month=month, data_type=data_type, market=cfg.market)

    month_str = month.strftime("%Y-%m")
    interval_part = f"_{interval}" if interval else ""
    zip_path = tmp_dir / f"{symbol}_{data_type}{interval_part}_{month_str}.zip"

    sess = session or requests.Session()
    ok = _download_stream_to_file(sess, url, zip_path, cfg)
    if not ok:
        return None

    csv_path = _safe_extract_single_csv(zip_path, tmp_dir)
    try:
        zip_path.unlink(missing_ok=True)
    except TypeError:
        if zip_path.exists():
            zip_path.unlink()
    return csv_path


def download_month_zip(
    symbol: str,
    month: datetime,
    data_type: str,
    interval: Optional[str],
    temp_dir: str,
    cfg: DownloadConfig,
    session: Optional[requests.Session] = None,
) -> Optional[Path]:
    """
    仅下载 ZIP，不解压，返回 ZIP 路径；失败返回 None。

    典型用途：超大文件（如 aggTrades）使用 zip 内流式读取，避免落地巨型 CSV。
    """
    temp_dir = _maybe_rewrite_alphaqcm_data_path(temp_dir)
    tmp_dir = Path(temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if interval:
        url = _build_monthly_kline_zip_url(
            symbol=symbol,
            month=month,
            kline_type=data_type,
            interval=interval,
            market=cfg.market,
        )
    else:
        url = _build_monthly_zip_url(symbol=symbol, month=month, data_type=data_type, market=cfg.market)

    month_str = month.strftime("%Y-%m")
    interval_part = f"_{interval}" if interval else ""
    zip_path = tmp_dir / f"{symbol}_{data_type}{interval_part}_{month_str}.zip"

    sess = session or requests.Session()
    ok = _download_stream_to_file(sess, url, zip_path, cfg)
    if not ok:
        return None
    return zip_path


def open_zip_single_csv(zip_path: Path) -> IO[bytes]:
    """
    打开 ZIP 内唯一 CSV 的流（bytes）。调用方负责关闭返回的文件对象以及 ZipFile。
    """
    zf = zipfile.ZipFile(zip_path, "r")
    members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
    if not members:
        zf.close()
        raise ValueError(f"ZIP 内没有 CSV: {zip_path}")
    if len(members) > 1:
        print(f"  警告: ZIP 内 CSV 文件数 > 1，将仅处理第一个: {zip_path.name}")
    member = members[0]
    member_path = Path(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        zf.close()
        raise ValueError(f"ZIP 文件路径不安全: {member}")

    stream = zf.open(member)  # ZipExtFile
    # 让调用方同时拿到 zf 的生命周期：把 zf 挂在 stream 上，便于最终关闭
    setattr(stream, "_zipfile_handle", zf)
    return stream


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """
    Binance 归档里时间字段可能是：
    - 形如 1706745600028 的毫秒时间戳
    - 或已经是 ISO 字符串
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True)
    # 容错：字符串数字也可能出现
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.98:
            return pd.to_datetime(numeric, unit="ms", utc=True)
    except Exception:
        pass
    return pd.to_datetime(series, utc=True)


def process_metrics_to_hourly(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "create_time" not in df.columns:
        raise ValueError(f"metrics 缺少 create_time 字段: {csv_path}")

    df["create_time"] = _to_utc_datetime(df["create_time"])
    df = df.set_index("create_time").sort_index()

    hourly = df.resample("1h").last()

    # 不同年份/市场的 metrics 字段可能不完全一致：以可用字段为准，保持最小可用集合
    preferred_cols = [
        "sum_open_interest",
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        # 下面这些字段在部分归档里不存在（例如已聚合后的输出），存在则保留
        "count_toptrader_long_short_ratio",
        "count_long_short_ratio",
    ]
    available = [c for c in preferred_cols if c in hourly.columns]
    required = ["sum_open_interest", "sum_open_interest_value"]
    missing_required = [c for c in required if c not in hourly.columns]
    if missing_required:
        raise ValueError(f"metrics 缺少必要字段 {missing_required}: {csv_path}")

    hourly = hourly[available]
    csv_path.unlink(missing_ok=True)
    return hourly


def process_kline_like_to_hourly(csv_path: Path, prefix: str) -> pd.DataFrame:
    """
    处理 Binance Vision 的 1h K线类归档（klines/markPriceKlines/indexPriceKlines/premiumIndexKlines）。

    输入列示例：
    open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
    """
    # 兼容两种归档格式：
    # - 日度 ZIP：带表头 open_time,open,high,...
    # - 月度 ZIP：不带表头，直接是数值行
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = (f.readline() or "").strip()

    if first_line.lower().startswith("open_time"):
        df = pd.read_csv(csv_path)
    else:
        names = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.read_csv(csv_path, header=None, names=names)

    df["open_time"] = _to_utc_datetime(df["open_time"])
    df = df.set_index("open_time").sort_index()

    keep: dict[str, pd.Series] = {}
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            keep[f"{prefix}_{c}"] = pd.to_numeric(df[c], errors="coerce")

    # 只有真实成交 K线（klines）才保留成交相关字段；mark/index/premium 一般为 0
    if prefix == "last":
        for c, out in [
            ("volume", "volume"),
            ("quote_volume", "quote_volume"),
            ("count", "trade_count"),
            ("taker_buy_volume", "taker_buy_volume"),
            ("taker_buy_quote_volume", "taker_buy_quote_volume"),
        ]:
            if c in df.columns:
                keep[out] = pd.to_numeric(df[c], errors="coerce")

    out_df = pd.DataFrame(keep, index=df.index)
    out_df = out_df.groupby(out_df.index.floor("h")).last().sort_index()
    csv_path.unlink(missing_ok=True)
    return out_df


def process_fundingrate_to_hourly(csv_path: Path) -> pd.DataFrame:
    """
    fundingRate（月度归档）转小时：8 小时结算一次，按小时 forward-fill。

    CSV 示例：
    calc_time,funding_interval_hours,last_funding_rate
    1706745600000,8,0.00010000
    """
    df = pd.read_csv(csv_path)
    if "calc_time" not in df.columns or "last_funding_rate" not in df.columns:
        raise ValueError(f"fundingRate 字段不符合预期: {csv_path}")

    df["calc_time"] = _to_utc_datetime(df["calc_time"])
    df = df.set_index("calc_time").sort_index()
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")

    hourly = df[["last_funding_rate"]].resample("1h").ffill()

    # 将月内最后一个 funding 点 forward-fill 到月底 23:00，避免每月末尾缺最后 7 小时（8h 结算导致）。
    first_ts = hourly.index.min()
    if first_ts is not pd.NaT:
        month_start = pd.Timestamp(year=first_ts.year, month=first_ts.month, day=1, tz="UTC")
        next_month = month_start + pd.offsets.MonthBegin(1)
        month_end_hour = next_month - pd.Timedelta(hours=1)
        full_index = pd.date_range(month_start, month_end_hour, freq="h", tz="UTC")
        hourly = hourly.reindex(full_index).ffill()
    csv_path.unlink(missing_ok=True)
    return hourly


AggTradesCsvSource = Union[Path, IO[bytes], IO[str]]


def process_aggtrades_to_1m(
    csv_source: AggTradesCsvSource,
    chunksize: int = 200_000,
    whale_trade_usd: float = 50_000.0,
    retail_trade_usd: float = 1_000.0,
) -> pd.DataFrame:
    """
    将逐笔 `aggTrades` 压缩成 1 分钟中间表（建议落盘 Parquet，供回测/特征复用）。

    输出（minute 级，索引为 UTC minute）：
    - 汇总：trade_count、base/quote_volume、taker_buy/sell（qty/quote）
    - 大小单：whale_* / retail_*（按单笔成交额 quote=price*qty）
    - HHI 所需：taker_buy_quote_sq / taker_sell_quote_sq（sum(amount^2)）
    - 价格：price_first/last/high/low + first_ts/last_ts（便于构造小时 OHLC）
    """
    minute_sum: Optional[pd.DataFrame] = None
    first_map: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
    last_map: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
    high_map: dict[pd.Timestamp, float] = {}
    low_map: dict[pd.Timestamp, float] = {}

    # aggTrades 归档 CSV 在早期月份通常“无表头”，后期月份带表头；两者列顺序一致：
    # 0 agg_trade_id, 1 price, 2 quantity, 3 first_trade_id, 4 last_trade_id, 5 transact_time, 6 is_buyer_maker
    for chunk in pd.read_csv(
        csv_source,
        header=None,
        usecols=[1, 2, 5, 6],
        chunksize=chunksize,
        dtype={1: "string", 2: "string", 5: "string", 6: "string"},
    ):
        chunk.columns = ["price", "quantity", "transact_time", "is_buyer_maker"]
        chunk["transact_time"] = _to_utc_datetime(chunk["transact_time"])
        minute = chunk["transact_time"].dt.floor("min")

        price = pd.to_numeric(chunk["price"], errors="coerce")
        qty = pd.to_numeric(chunk["quantity"], errors="coerce")
        valid = chunk["transact_time"].notna() & price.notna() & qty.notna()
        if not bool(valid.any()):
            continue
        minute = minute[valid]
        price = price[valid]
        qty = qty[valid]
        chunk = chunk.loc[valid]
        is_buyer_maker = chunk["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        quote_qty = price * qty

        taker_buy_mask = ~is_buyer_maker
        taker_sell_mask = is_buyer_maker
        whale_mask = quote_qty >= whale_trade_usd
        retail_mask = quote_qty <= retail_trade_usd

        tmp = pd.DataFrame(
            {
                "minute": minute,
                "trade_count": 1,
                "base_volume": qty,
                "quote_volume": quote_qty,
                "taker_buy_qty": qty.where(taker_buy_mask, 0.0),
                "taker_sell_qty": qty.where(taker_sell_mask, 0.0),
                "taker_buy_quote": quote_qty.where(taker_buy_mask, 0.0),
                "taker_sell_quote": quote_qty.where(taker_sell_mask, 0.0),
                "taker_buy_quote_sq": (quote_qty * quote_qty).where(taker_buy_mask, 0.0),
                "taker_sell_quote_sq": (quote_qty * quote_qty).where(taker_sell_mask, 0.0),
                "taker_buy_trade_count": taker_buy_mask.astype(int),
                "taker_sell_trade_count": taker_sell_mask.astype(int),
                "whale_buy_quote": quote_qty.where(taker_buy_mask & whale_mask, 0.0),
                "whale_sell_quote": quote_qty.where(taker_sell_mask & whale_mask, 0.0),
                "retail_buy_quote": quote_qty.where(taker_buy_mask & retail_mask, 0.0),
                "retail_sell_quote": quote_qty.where(taker_sell_mask & retail_mask, 0.0),
            }
        )
        g = tmp.groupby("minute", as_index=True).sum(numeric_only=True)
        minute_sum = g if minute_sum is None else minute_sum.add(g, fill_value=0.0)

        price_df = pd.DataFrame({"minute": minute, "ts": chunk["transact_time"], "price": price})
        hi = price_df.groupby("minute", as_index=True)["price"].max()
        lo = price_df.groupby("minute", as_index=True)["price"].min()
        idx_first = price_df.groupby("minute")["ts"].idxmin()
        idx_last = price_df.groupby("minute")["ts"].idxmax()
        first_df = price_df.loc[idx_first, ["minute", "ts", "price"]].set_index("minute")
        last_df = price_df.loc[idx_last, ["minute", "ts", "price"]].set_index("minute")

        for m, v in hi.items():
            if pd.isna(v):
                continue
            prev = high_map.get(m)
            high_map[m] = float(v) if prev is None else max(prev, float(v))
        for m, v in lo.items():
            if pd.isna(v):
                continue
            prev = low_map.get(m)
            low_map[m] = float(v) if prev is None else min(prev, float(v))

        for m, row in first_df.iterrows():
            ts = row["ts"]
            p = row["price"]
            if pd.isna(ts) or pd.isna(p):
                continue
            prev = first_map.get(m)
            if prev is None or ts < prev[0]:
                first_map[m] = (pd.Timestamp(ts), float(p))

        for m, row in last_df.iterrows():
            ts = row["ts"]
            p = row["price"]
            if pd.isna(ts) or pd.isna(p):
                continue
            prev = last_map.get(m)
            if prev is None or ts > prev[0]:
                last_map[m] = (pd.Timestamp(ts), float(p))

    if minute_sum is None:
        return pd.DataFrame()

    out = minute_sum.sort_index()
    extra = pd.DataFrame(index=out.index)
    if first_map:
        extra["first_ts"] = pd.Series({k: v[0] for k, v in first_map.items()})
        extra["price_first"] = pd.Series({k: v[1] for k, v in first_map.items()})
    if last_map:
        extra["last_ts"] = pd.Series({k: v[0] for k, v in last_map.items()})
        extra["price_last"] = pd.Series({k: v[1] for k, v in last_map.items()})
    if high_map:
        extra["price_high"] = pd.Series(high_map)
    if low_map:
        extra["price_low"] = pd.Series(low_map)

    out = out.join(extra, how="left")
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "minute"
    return out


def process_aggtrades_1m_to_hourly(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    从 1m 中间表聚合出 1h 订单流特征（供小时线策略使用）。
    """
    if minute_df.empty:
        return pd.DataFrame()

    df = minute_df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("aggTrades 1m 数据 index 需要是 DatetimeIndex（UTC minute）")

    hour = df.index.floor("h")
    sum_cols = [
        "trade_count",
        "base_volume",
        "quote_volume",
        "taker_buy_qty",
        "taker_sell_qty",
        "taker_buy_quote",
        "taker_sell_quote",
        "taker_buy_quote_sq",
        "taker_sell_quote_sq",
        "taker_buy_trade_count",
        "taker_sell_trade_count",
        "whale_buy_quote",
        "whale_sell_quote",
        "retail_buy_quote",
        "retail_sell_quote",
    ]
    available_sum_cols = [c for c in sum_cols if c in df.columns]
    agg = df[available_sum_cols].groupby(hour, as_index=True).sum(numeric_only=True).sort_index()

    # 价格路径（由 minute 的 first/last/high/low 构造小时 OHLC）
    if "price_first" in df.columns:
        agg["price_open"] = df["price_first"].groupby(hour, as_index=True).first()
    if "price_last" in df.columns:
        agg["price_close"] = df["price_last"].groupby(hour, as_index=True).last()
    if "price_high" in df.columns:
        agg["price_high"] = df["price_high"].groupby(hour, as_index=True).max()
    if "price_low" in df.columns:
        agg["price_low"] = df["price_low"].groupby(hour, as_index=True).min()

    # 衍生指标（尽量与原 hourly 口径一致）
    agg["cvd_qty"] = agg.get("taker_buy_qty", 0.0) - agg.get("taker_sell_qty", 0.0)
    agg["cvd_quote"] = agg.get("taker_buy_quote", 0.0) - agg.get("taker_sell_quote", 0.0)
    agg["vwap"] = (agg.get("quote_volume", 0.0) / agg.get("base_volume", 0.0)).replace([float("inf"), -float("inf")], pd.NA)

    agg["whale_cvd_quote"] = agg.get("whale_buy_quote", 0.0) - agg.get("whale_sell_quote", 0.0)
    agg["retail_cvd_quote"] = agg.get("retail_buy_quote", 0.0) - agg.get("retail_sell_quote", 0.0)

    trade_count = agg.get("trade_count", pd.Series(index=agg.index, dtype="float64")).replace(0.0, pd.NA)
    agg["avg_trade_quote"] = (agg.get("quote_volume", 0.0) / trade_count).replace([float("inf"), -float("inf")], pd.NA)
    agg["avg_trade_base"] = (agg.get("base_volume", 0.0) / trade_count).replace([float("inf"), -float("inf")], pd.NA)

    buy_cnt = agg.get("taker_buy_trade_count", pd.Series(index=agg.index, dtype="float64")).replace(0.0, pd.NA)
    sell_cnt = agg.get("taker_sell_trade_count", pd.Series(index=agg.index, dtype="float64")).replace(0.0, pd.NA)
    agg["taker_buy_avg_trade_quote"] = (agg.get("taker_buy_quote", 0.0) / buy_cnt).replace([float("inf"), -float("inf")], pd.NA)
    agg["taker_sell_avg_trade_quote"] = (agg.get("taker_sell_quote", 0.0) / sell_cnt).replace([float("inf"), -float("inf")], pd.NA)

    buy_denom = (agg.get("taker_buy_quote", 0.0) * agg.get("taker_buy_quote", 0.0)).replace(0.0, pd.NA)
    sell_denom = (agg.get("taker_sell_quote", 0.0) * agg.get("taker_sell_quote", 0.0)).replace(0.0, pd.NA)
    agg["taker_buy_hhi_quote"] = (agg.get("taker_buy_quote_sq", 0.0) / buy_denom).clip(lower=0)
    agg["taker_sell_hhi_quote"] = (agg.get("taker_sell_quote_sq", 0.0) / sell_denom).clip(lower=0)

    total_quote = (agg.get("taker_buy_quote", 0.0) + agg.get("taker_sell_quote", 0.0)).replace(0.0, pd.NA)
    agg["imbalance_ratio_quote"] = ((agg["cvd_quote"].abs()) / total_quote).clip(lower=0)

    # 时间加权攻击性：按分钟加权（越接近小时末权重越大），并归一化到同一权重尺度
    if "taker_buy_quote" in df.columns and "taker_sell_quote" in df.columns:
        cvd_min = (df["taker_buy_quote"] - df["taker_sell_quote"]).fillna(0.0)
        w = (df.index.minute.astype(int) + 1).astype("float64")  # 1..60
        tw_num = (cvd_min.to_numpy() * w).astype("float64")
        tw_num_s = pd.Series(tw_num, index=df.index).groupby(hour, as_index=True).sum()
        agg["tw_cvd_quote_norm"] = (tw_num_s / 1830.0).reindex(agg.index)  # sum(1..60)=1830
    else:
        agg["tw_cvd_quote_norm"] = pd.NA

    if "price_open" in agg.columns and "price_close" in agg.columns:
        agg["price_change"] = agg["price_close"] - agg["price_open"]
        abs_price_change = agg["price_change"].abs().replace(0.0, pd.NA)
        agg["cvd_per_abs_price_change"] = (agg["cvd_quote"] / abs_price_change).replace([float("inf"), -float("inf")], pd.NA)
    else:
        agg["price_change"] = pd.NA
        agg["cvd_per_abs_price_change"] = pd.NA

    return agg.sort_index()


def process_aggtrades_to_hourly(
    csv_source: AggTradesCsvSource,
    chunksize: int = 1_000_000,
    whale_trade_usd: float = 50_000.0,
    retail_trade_usd: float = 1_000.0,
    return_1m: bool = False,
) -> pd.DataFrame:
    """
    从 aggTrades 聚合得到小时级订单流特征（CVD 等）。

    字段约定：
    - `is_buyer_maker=True` 表示 Maker 是买方，因此 Taker 是卖方
    - 所以：taker_buy = ~is_buyer_maker；taker_sell = is_buyer_maker

    输出特征（小时级）：
    - 基础：trade_count、base/quote_volume、taker_buy/sell（qty/quote）、cvd（qty/quote）、vwap
    - 大小单拆分（按单笔成交额 quote=price*qty）：
      whale_*（>= whale_trade_usd）、retail_*（<= retail_trade_usd）
    - 成交强度：avg_trade_quote、avg_trade_base、avg_trade_quote_per_trade 等
    - 时间加权攻击性：tw_cvd_quote_norm（越接近小时末权重越大）
    - 冲击/吸收：price_open/close/high/low、price_change、cvd_per_abs_price_change
    - 主动性集中度（HHI proxy）：taker_buy_hhi_quote、taker_sell_hhi_quote（按单笔成交额集中度）
    """
    agg: Optional[pd.DataFrame] = None
    open_price: dict[pd.Timestamp, float] = {}
    close_price: dict[pd.Timestamp, float] = {}
    high_price: dict[pd.Timestamp, float] = {}
    low_price: dict[pd.Timestamp, float] = {}
    tw_cvd_num: dict[pd.Timestamp, float] = {}
    tw_cvd_den: dict[pd.Timestamp, float] = {}

    minute_parts: list[pd.DataFrame] = []
    minute_first_parts: list[pd.DataFrame] = []
    minute_last_parts: list[pd.DataFrame] = []
    minute_hilo_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        csv_source,
        header=None,
        usecols=[1, 2, 5, 6],
        chunksize=chunksize,
        dtype={1: "string", 2: "string", 5: "string", 6: "string"},
    ):
        chunk.columns = ["price", "quantity", "transact_time", "is_buyer_maker"]
        chunk["transact_time"] = _to_utc_datetime(chunk["transact_time"])
        hour = chunk["transact_time"].dt.floor("h")
        minute_idx = chunk["transact_time"].dt.minute
        minute = chunk["transact_time"].dt.floor("min")

        price = pd.to_numeric(chunk["price"], errors="coerce")
        qty = pd.to_numeric(chunk["quantity"], errors="coerce")
        valid = chunk["transact_time"].notna() & price.notna() & qty.notna()
        if not bool(valid.any()):
            continue
        hour = hour[valid]
        minute_idx = minute_idx[valid]
        minute = minute[valid]
        price = price[valid]
        qty = qty[valid]
        chunk = chunk.loc[valid]
        is_buyer_maker = chunk["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        quote_qty = price * qty

        taker_buy_mask = ~is_buyer_maker
        taker_sell_mask = is_buyer_maker

        whale_mask = quote_qty >= whale_trade_usd
        retail_mask = quote_qty <= retail_trade_usd

        tmp = pd.DataFrame(
            {
                "hour": hour,
                "trade_count": 1,
                "base_volume": qty,
                "quote_volume": quote_qty,
                "taker_buy_qty": qty.where(taker_buy_mask, 0.0),
                "taker_sell_qty": qty.where(taker_sell_mask, 0.0),
                "taker_buy_quote": quote_qty.where(taker_buy_mask, 0.0),
                "taker_sell_quote": quote_qty.where(taker_sell_mask, 0.0),
                "taker_buy_quote_sq": (quote_qty * quote_qty).where(taker_buy_mask, 0.0),
                "taker_sell_quote_sq": (quote_qty * quote_qty).where(taker_sell_mask, 0.0),
                "taker_buy_trade_count": taker_buy_mask.astype(int),
                "taker_sell_trade_count": taker_sell_mask.astype(int),
                "whale_buy_quote": quote_qty.where(taker_buy_mask & whale_mask, 0.0),
                "whale_sell_quote": quote_qty.where(taker_sell_mask & whale_mask, 0.0),
                "retail_buy_quote": quote_qty.where(taker_buy_mask & retail_mask, 0.0),
                "retail_sell_quote": quote_qty.where(taker_sell_mask & retail_mask, 0.0),
            }
        )

        g = tmp.groupby("hour", as_index=True).sum(numeric_only=True)
        if agg is None:
            agg = g
        else:
            agg = agg.add(g, fill_value=0.0)

        if return_1m:
            mtmp = tmp.copy()
            mtmp["minute"] = minute
            mg = mtmp.groupby("minute", as_index=True).sum(numeric_only=True)
            minute_parts.append(mg)

            price_df = pd.DataFrame({"minute": minute, "transact_time": chunk["transact_time"], "price": price})
            # first/last：用 transact_time 做 tie-breaker
            first_idx = price_df.groupby("minute")["transact_time"].idxmin()
            last_idx = price_df.groupby("minute")["transact_time"].idxmax()

            first = price_df.loc[first_idx, ["minute", "transact_time", "price"]].rename(
                columns={"transact_time": "first_ts", "price": "price_first"}
            )
            last = price_df.loc[last_idx, ["minute", "transact_time", "price"]].rename(
                columns={"transact_time": "last_ts", "price": "price_last"}
            )
            hilo = price_df.groupby("minute", as_index=False)["price"].agg(price_high="max", price_low="min")

            minute_first_parts.append(first.set_index("minute"))
            minute_last_parts.append(last.set_index("minute"))
            minute_hilo_parts.append(hilo.set_index("minute"))

        # 价格路径（用于冲击/吸收类指标）
        price_df = pd.DataFrame({"hour": hour, "price": price})
        price_stats = price_df.groupby("hour", as_index=True)["price"].agg(["first", "last", "max", "min"])
        for ts, row in price_stats.iterrows():
            if ts not in open_price and pd.notna(row["first"]):
                open_price[ts] = float(row["first"])
            if pd.notna(row["last"]):
                close_price[ts] = float(row["last"])
            if pd.notna(row["max"]):
                prev = high_price.get(ts)
                high_price[ts] = float(row["max"]) if prev is None else max(prev, float(row["max"]))
            if pd.notna(row["min"]):
                prev = low_price.get(ts)
                low_price[ts] = float(row["min"]) if prev is None else min(prev, float(row["min"]))

        # 时间加权攻击性：按分钟聚合 CVD，并给越接近小时末的分钟更高权重（1..60）
        minute_tmp = pd.DataFrame(
            {
                "hour": hour,
                "minute": minute_idx,
                "taker_buy_quote": quote_qty.where(taker_buy_mask, 0.0),
                "taker_sell_quote": quote_qty.where(taker_sell_mask, 0.0),
            }
        )
        m = minute_tmp.groupby(["hour", "minute"], as_index=True).sum(numeric_only=True)
        cvd_min = m["taker_buy_quote"] - m["taker_sell_quote"]
        w = cvd_min.index.get_level_values("minute").astype(int) + 1
        cvd_w = cvd_min.to_numpy() * w.to_numpy()
        w_s = pd.Series(w.to_numpy(), index=cvd_min.index).groupby(level=0).sum()
        cvd_w_s = pd.Series(cvd_w, index=cvd_min.index).groupby(level=0).sum()
        for ts in cvd_w_s.index:
            tw_cvd_num[ts] = float(tw_cvd_num.get(ts, 0.0) + cvd_w_s.loc[ts])
            tw_cvd_den[ts] = float(tw_cvd_den.get(ts, 0.0) + w_s.loc[ts])

    if agg is None:
        if isinstance(csv_source, Path):
            csv_source.unlink(missing_ok=True)
        return pd.DataFrame()

    # 衍生指标
    agg["cvd_qty"] = agg["taker_buy_qty"] - agg["taker_sell_qty"]
    agg["cvd_quote"] = agg["taker_buy_quote"] - agg["taker_sell_quote"]
    agg["vwap"] = (agg["quote_volume"] / agg["base_volume"]).replace([float("inf"), -float("inf")], pd.NA)

    # 大小单拆分 CVD（按 quote USD）
    agg["whale_cvd_quote"] = agg["whale_buy_quote"] - agg["whale_sell_quote"]
    agg["retail_cvd_quote"] = agg["retail_buy_quote"] - agg["retail_sell_quote"]

    # 成交强度
    agg["avg_trade_quote"] = (agg["quote_volume"] / agg["trade_count"]).replace([float("inf"), -float("inf")], pd.NA)
    agg["avg_trade_base"] = (agg["base_volume"] / agg["trade_count"]).replace([float("inf"), -float("inf")], pd.NA)
    agg["taker_buy_avg_trade_quote"] = (agg["taker_buy_quote"] / agg["taker_buy_trade_count"]).replace(
        [float("inf"), -float("inf")], pd.NA
    )
    agg["taker_sell_avg_trade_quote"] = (agg["taker_sell_quote"] / agg["taker_sell_trade_count"]).replace(
        [float("inf"), -float("inf")], pd.NA
    )

    # 主动性集中度（HHI proxy）：sum(amount^2) / sum(amount)^2
    buy_denom = (agg["taker_buy_quote"] * agg["taker_buy_quote"]).replace(0.0, pd.NA)
    sell_denom = (agg["taker_sell_quote"] * agg["taker_sell_quote"]).replace(0.0, pd.NA)
    agg["taker_buy_hhi_quote"] = (agg["taker_buy_quote_sq"] / buy_denom).clip(lower=0)
    agg["taker_sell_hhi_quote"] = (agg["taker_sell_quote_sq"] / sell_denom).clip(lower=0)

    # 订单流失衡（VPIN 的简化代理）
    total_quote = (agg["taker_buy_quote"] + agg["taker_sell_quote"]).replace(0.0, pd.NA)
    agg["imbalance_ratio_quote"] = ((agg["cvd_quote"].abs()) / total_quote).clip(lower=0)

    # 时间加权攻击性（越接近小时末权重越大）：归一化到同一权重尺度
    tw_num_s = pd.Series(tw_cvd_num, dtype="float64")
    tw_den_s = pd.Series(tw_cvd_den, dtype="float64").replace(0.0, pd.NA)
    agg["tw_cvd_quote_norm"] = (tw_num_s / tw_den_s).reindex(agg.index)

    # 价格路径与冲击/吸收
    price_path = pd.DataFrame(
        {
            "price_open": pd.Series(open_price, dtype="float64"),
            "price_close": pd.Series(close_price, dtype="float64"),
            "price_high": pd.Series(high_price, dtype="float64"),
            "price_low": pd.Series(low_price, dtype="float64"),
        }
    )
    agg = agg.join(price_path, how="left")
    agg["price_change"] = agg["price_close"] - agg["price_open"]
    abs_price_change = agg["price_change"].abs().replace(0.0, pd.NA)
    agg["cvd_per_abs_price_change"] = (agg["cvd_quote"] / abs_price_change).replace([float("inf"), -float("inf")], pd.NA)

    if return_1m:
        minute_sum = pd.concat(minute_parts).groupby(level=0).sum(numeric_only=True) if minute_parts else pd.DataFrame()
        minute_first = (
            pd.concat(minute_first_parts).sort_values("first_ts").loc[lambda d: ~d.index.duplicated(keep="first")]
            if minute_first_parts
            else pd.DataFrame()
        )
        minute_last = (
            pd.concat(minute_last_parts).sort_values("last_ts").loc[lambda d: ~d.index.duplicated(keep="last")]
            if minute_last_parts
            else pd.DataFrame()
        )
        minute_hilo = (
            pd.concat(minute_hilo_parts).groupby(level=0).agg(price_high=("price_high", "max"), price_low=("price_low", "min"))
            if minute_hilo_parts
            else pd.DataFrame()
        )

        minute_df = minute_sum.join(minute_first, how="left").join(minute_last, how="left").join(minute_hilo, how="left")
        minute_df = minute_df.sort_index()

        if isinstance(csv_source, Path):
            csv_source.unlink(missing_ok=True)

        # 返回一个两层结构：小时 + 1m（通过附加属性传递，避免改现有调用方太多）
        hourly = agg.sort_index()
        setattr(hourly, "_aggtrades_1m", minute_df)
        return hourly

    if isinstance(csv_source, Path):
        csv_source.unlink(missing_ok=True)
    return agg.sort_index()


def process_liquidations_to_hourly(csv_path: Path) -> pd.DataFrame:
    """
    处理 liquidationSnapshot（通常仅 cm 市场可用）。
    输出：
    - liq_long_qty / liq_short_qty：按小时汇总的强平数量（以合约基础计量单位）
    - liq_long_notional / liq_short_notional：按小时汇总的强平名义金额（qty * average_price）
    """
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "side" not in df.columns:
        raise ValueError(f"liquidationSnapshot 字段不符合预期: {csv_path}")

    df = df.drop_duplicates()
    df["time"] = _to_utc_datetime(df["time"])
    df["hour"] = df["time"].dt.floor("h")

    qty_col = "accumulated_fill_quantity" if "accumulated_fill_quantity" in df.columns else "original_quantity"
    price_col = "average_price" if "average_price" in df.columns else "price"

    qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    price = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
    notional = qty * price

    side = df["side"].astype(str).str.upper()
    is_long = side.eq("SELL")  # SELL 通常对应多头被强平
    is_short = side.eq("BUY")

    tmp = pd.DataFrame(
        {
            "hour": df["hour"],
            "liq_count": 1,
            "liq_long_qty": qty.where(is_long, 0.0),
            "liq_short_qty": qty.where(is_short, 0.0),
            "liq_long_notional": notional.where(is_long, 0.0),
            "liq_short_notional": notional.where(is_short, 0.0),
        }
    )
    hourly = tmp.groupby("hour", as_index=True).sum(numeric_only=True).sort_index()
    csv_path.unlink(missing_ok=True)
    return hourly


def download_symbol_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_type: str,
    output_dir: str,
    temp_dir: str = "AlphaQCM_data/_tmp/binance_vision",
    market: str = "um",
    cfg: Optional[DownloadConfig] = None,
    interval: Optional[str] = None,
    skip_existing: bool = False,
    aggtrades_1m_dir: Optional[str] = None,
    aggtrades_keep_zip: bool = False,
) -> bool:
    """
    下载某个 symbol 的日期区间，并输出小时级 CSV。

    返回 True 表示成功写出了结果文件（至少有 1 天数据被处理）；否则 False。
    """
    cfg = cfg or DownloadConfig(market=market)

    output_dir = _maybe_rewrite_alphaqcm_data_path(output_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"{symbol}_{data_type}.csv"
    meta_file = output_file.with_suffix(output_file.suffix + ".meta.json")
    existing_df: Optional[pd.DataFrame] = None
    existing_min_ts: Optional[pd.Timestamp] = None
    existing_max_ts: Optional[pd.Timestamp] = None
    if skip_existing and output_file.exists():
        expected_start = pd.Timestamp(start_date, tz="UTC")
        # metrics 是 daily 归档：不少合约的文件只给到当天某个固定时间点（例如 02:00），
        # 因此“覆盖到结束日期”用日期粒度判断更稳妥，避免每次重跑都重复请求尾部。
        expected_end = pd.Timestamp(end_date, tz="UTC")
        if data_type != "metrics":
            expected_end = expected_end + pd.Timedelta(hours=23)

        existing_min_ts = _read_csv_first_timestamp(output_file)
        existing_max_ts = _read_csv_last_timestamp(output_file)

        # 1) 优先使用 meta 判断（最快）：仅当范围也覆盖时才可跳过
        if meta_file.exists() and existing_min_ts is not None and existing_max_ts is not None:
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if (
                    meta.get("status") == "ok"
                    and meta.get("symbol") == symbol
                    and meta.get("data_type") == data_type
                    and meta.get("market") == market
                    and meta.get("interval") == interval
                    and meta.get("requested_start") == start_date.strftime("%Y-%m-%d")
                    and meta.get("requested_end") == end_date.strftime("%Y-%m-%d")
                    and existing_min_ts <= expected_start
                    and existing_max_ts >= expected_end
                ):
                    print("  已存在且已标记完成，跳过")
                    return True
            except Exception:
                pass

        # 2) meta 不存在/不匹配时，用 min/max 兜底判断是否已覆盖
        if existing_min_ts is not None and existing_max_ts is not None:
            if existing_min_ts <= expected_start and existing_max_ts >= expected_end:
                print(f"  已存在且覆盖到 {existing_max_ts}，跳过")
                # 若 meta 缺失，补写一份，便于下次快速跳过
                if not meta_file.exists():
                    try:
                        meta = {
                            "status": "ok",
                            "symbol": symbol,
                            "data_type": data_type,
                            "market": market,
                            "interval": interval,
                            "requested_start": start_date.strftime("%Y-%m-%d"),
                            "requested_end": end_date.strftime("%Y-%m-%d"),
                            "min_ts": existing_min_ts.isoformat(),
                            "max_ts": existing_max_ts.isoformat(),
                            "rows": None,
                            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        }
                        meta_file.write_text(
                            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
                    except Exception:
                        pass
                return True

        # 3) metrics 允许“增量补齐”：只下载缺口部分再合并，避免重复请求
        if data_type == "metrics":
            try:
                existing_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
                existing_df.index = pd.to_datetime(existing_df.index, utc=True, errors="coerce")
                existing_df = existing_df.loc[existing_df.index.notna()].sort_index()
                if len(existing_df) > 0:
                    existing_min_ts = existing_df.index.min()
                    existing_max_ts = existing_df.index.max()
            except Exception:
                existing_df = None

    kline_prefix = {
        "klines": "last",
        "markPriceKlines": "mark",
        "indexPriceKlines": "index",
        "premiumIndexKlines": "premium",
    }

    if data_type in kline_prefix and not interval:
        interval = "1h"

    processor = {
        "metrics": lambda p: process_metrics_to_hourly(p),
        "fundingRate": lambda p: process_fundingrate_to_hourly(p),
        "aggTrades": lambda p: process_aggtrades_to_hourly(p),
        "liquidationSnapshot": lambda p: process_liquidations_to_hourly(p),
        "klines": lambda p: process_kline_like_to_hourly(p, prefix=kline_prefix["klines"]),
        "markPriceKlines": lambda p: process_kline_like_to_hourly(p, prefix=kline_prefix["markPriceKlines"]),
        "indexPriceKlines": lambda p: process_kline_like_to_hourly(p, prefix=kline_prefix["indexPriceKlines"]),
        "premiumIndexKlines": lambda p: process_kline_like_to_hourly(p, prefix=kline_prefix["premiumIndexKlines"]),
    }.get(data_type)

    if processor is None:
        raise ValueError(f"暂不支持的数据类型: {data_type}")

    all_data: list[pd.DataFrame] = []

    def _iter_month_starts(start: datetime, end: datetime) -> Iterable[datetime]:
        month = datetime(start.year, start.month, 1)
        last = datetime(end.year, end.month, 1)
        while month <= last:
            yield month
            month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)

    with requests.Session() as session:
        if data_type == "fundingRate" and cfg.market != "um":
            raise ValueError("fundingRate 仅支持 um 市场")

        use_monthly = data_type in {"fundingRate", "aggTrades"} or data_type in kline_prefix
        if use_monthly:
            month_interval = interval if data_type in kline_prefix else None
            # 优化：自动跳过起始处连续 404 的月份（常见于新上线合约）
            first_month = _find_first_available_month(
                session=session,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type,
                interval=month_interval,
                cfg=cfg,
            )
            if first_month is None:
                print("  无可用月份数据，跳过")
                return False

            if first_month > datetime(start_date.year, start_date.month, 1):
                print(f"  前置月份无数据，跳到 {first_month.strftime('%Y-%m')}")

            for month in _iter_month_starts(first_month, end_date):
                month_str = month.strftime("%Y-%m")
                print(f"  {month_str}...", end=" ")

                try:
                    if data_type == "aggTrades":
                        # aggTrades 体量巨大：优先用 zip 内流式聚合成 1m Parquet，再由 1m 聚合出 1h
                        if aggtrades_1m_dir is None:
                            aggtrades_1m_dir = "AlphaQCM_data/binance_aggTrades_1m_parquet"
                        aggtrades_1m_dir = _maybe_rewrite_alphaqcm_data_path(aggtrades_1m_dir)

                        month_parquet = Path(aggtrades_1m_dir) / symbol / f"{month_str}.parquet"
                        month_parquet.parent.mkdir(parents=True, exist_ok=True)

                        if month_parquet.exists():
                            minute_df = pd.read_parquet(month_parquet)
                            if "minute" in minute_df.columns:
                                minute_df["minute"] = pd.to_datetime(minute_df["minute"], utc=True, errors="coerce")
                                minute_df = minute_df.set_index("minute").sort_index()
                            if not isinstance(minute_df.index, pd.DatetimeIndex):
                                minute_df.index = pd.to_datetime(minute_df.index, utc=True, errors="coerce")
                                minute_df = minute_df.sort_index()
                        else:
                            zip_path = download_month_zip(
                                symbol=symbol,
                                month=month,
                                data_type=data_type,
                                interval=None,
                                temp_dir=temp_dir,
                                cfg=cfg,
                                session=session,
                            )
                            if zip_path is None:
                                print("✗")
                                continue

                            raw_stream = open_zip_single_csv(zip_path)
                            text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8", errors="ignore")
                            setattr(text_stream, "_zipfile_handle", getattr(raw_stream, "_zipfile_handle", None))
                            try:
                                minute_df = process_aggtrades_to_1m(text_stream)
                            finally:
                                try:
                                    text_stream.close()
                                finally:
                                    zf = getattr(text_stream, "_zipfile_handle", None)
                                    if zf is not None:
                                        try:
                                            zf.close()
                                        except Exception:
                                            pass

                            minute_df.to_parquet(month_parquet)
                            if not aggtrades_keep_zip:
                                try:
                                    zip_path.unlink(missing_ok=True)
                                except TypeError:
                                    if zip_path.exists():
                                        zip_path.unlink()

                        hourly = process_aggtrades_1m_to_hourly(minute_df)
                        if len(hourly) > 0:
                            all_data.append(hourly)
                        print(f"✓ {len(hourly)} hours (1m={len(minute_df)})")
                    else:
                        csv_path = download_month_zip_to_csv(
                            symbol=symbol,
                            month=month,
                            data_type=data_type,
                            interval=month_interval,
                            temp_dir=temp_dir,
                            cfg=cfg,
                            session=session,
                        )
                        if csv_path is None:
                            print("✗")
                            continue

                        hourly = processor(csv_path)
                        if len(hourly) > 0:
                            all_data.append(hourly)
                        print(f"✓ {len(hourly)} hours")
                except Exception as e:
                    print(f"✗ 处理失败: {e}")
                    try:
                        if data_type != "aggTrades":
                            csv_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        else:
            ranges: list[tuple[datetime, datetime]] = [(start_date, end_date)]
            if existing_df is not None and existing_min_ts is not None and existing_max_ts is not None:
                expected_start = pd.Timestamp(start_date, tz="UTC")
                expected_end = pd.Timestamp(end_date, tz="UTC")

                ranges = []
                # 头部缺口
                if expected_start < existing_min_ts:
                    head_end = (existing_min_ts - pd.Timedelta(hours=1)).to_pydatetime().date()
                    head_end_dt = datetime(head_end.year, head_end.month, head_end.day)
                    ranges.append((start_date, min(head_end_dt, end_date)))
                # 尾部缺口
                if expected_end > existing_max_ts:
                    tail_start = (existing_max_ts + pd.Timedelta(hours=1)).to_pydatetime().date()
                    tail_start_dt = datetime(tail_start.year, tail_start.month, tail_start.day)
                    ranges.append((max(start_date, tail_start_dt), end_date))

                # 若完全被 existing 覆盖（但范围不含 start/end），ranges 可能为空，直接走合并过滤即可

            for r_start, r_end in ranges:
                if r_start > r_end:
                    continue

                first_day = _find_first_available_day(
                    session=session,
                    symbol=symbol,
                    start_date=r_start,
                    end_date=r_end,
                    data_type=data_type,
                    interval=interval,
                    cfg=cfg,
                )
                if first_day is None:
                    continue

                current = first_day
                while current <= r_end:
                    date_str = current.strftime("%Y-%m-%d")
                    print(f"  {date_str}...", end=" ")

                    csv_path = download_day_zip_to_csv(
                        symbol=symbol,
                        date=current,
                        data_type=data_type,
                        interval=interval,
                        temp_dir=temp_dir,
                        cfg=cfg,
                        session=session,
                    )
                    if csv_path is None:
                        print("✗")
                        current += timedelta(days=1)
                        continue

                    try:
                        hourly = processor(csv_path)
                        if len(hourly) > 0:
                            all_data.append(hourly)
                        print(f"✓ {len(hourly)} hours")
                    except Exception as e:
                        print(f"✗ 处理失败: {e}")
                        try:
                            csv_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                    current += timedelta(days=1)

    if not all_data:
        # 若没有新数据，但 existing_df 覆盖了部分区间，仍视为“成功”（保留并裁剪）
        if existing_df is not None and len(existing_df) > 0:
            df_final = existing_df
        else:
            return False
    else:
        df_final = pd.concat(all_data).sort_index()

    # metrics 增量补齐：合并 existing
    if existing_df is not None and len(existing_df) > 0:
        df_final = pd.concat([existing_df, df_final]).sort_index()
        df_final = df_final[~df_final.index.duplicated(keep="last")]

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
    df_final = df_final.loc[(df_final.index >= start_ts) & (df_final.index <= end_ts)]
    df_final.to_csv(output_file)
    try:
        meta = {
            "status": "ok",
            "symbol": symbol,
            "data_type": data_type,
            "market": market,
            "interval": interval,
            "requested_start": start_date.strftime("%Y-%m-%d"),
            "requested_end": end_date.strftime("%Y-%m-%d"),
            "min_ts": (df_final.index.min().isoformat() if len(df_final.index) else None),
            "max_ts": (df_final.index.max().isoformat() if len(df_final.index) else None),
            "rows": int(df_final.shape[0]),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    print(f"  Saved: {output_file}")
    return True


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="下载 Binance Vision 归档数据并按小时聚合输出 CSV")
    parser.add_argument("--symbol", required=True, help="合约符号，例如 BTCUSDT 或 BTCUSD_PERP")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD（包含）")
    parser.add_argument(
        "--data-type",
        default="metrics",
        choices=[
            "metrics",
            "fundingRate",
            "klines",
            "markPriceKlines",
            "indexPriceKlines",
            "premiumIndexKlines",
            "aggTrades",
            "liquidationSnapshot",
        ],
    )
    parser.add_argument("--interval", default="", help="K线类数据周期，例如 1h；非 K线类可忽略")
    parser.add_argument("--market", default="um", choices=["um", "cm"], help="um=U本位，cm=币本位")
    parser.add_argument("--output-dir", default="AlphaQCM_data/binance_archive_processed")
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    print(f"Downloading {args.symbol} {args.data_type} ({args.market}) from {start.date()} to {end.date()}...")
    ok = download_symbol_range(
        symbol=args.symbol,
        start_date=start,
        end_date=end,
        data_type=args.data_type,
        interval=(args.interval or None),
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        market=args.market,
        skip_existing=args.skip_existing,
        aggtrades_1m_dir=(args.aggtrades_1m_dir or None),
        aggtrades_keep_zip=args.aggtrades_keep_zip,
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
