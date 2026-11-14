"""MT5 bar-export importer: parsing variants, timezone rules and detection, cleaning, spread units."""
import io

import numpy as np
import pandas as pd
import pytest

from goldsys.data import mt5csv, store

TAB = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\r\n"


def tab_csv(rows):
    return io.StringIO(TAB + "".join("\t".join(map(str, r)) + "\r\n" for r in rows))


def test_parse_tab_export_with_brackets_and_crlf():
    df = mt5csv.parse_mt5_csv(tab_csv([("2024.03.01", "00:00:00", 2000.1, 2000.5, 1999.9, 2000.3, 75, 0, 18),
                                       ("2024.03.01", "00:01:00", 2000.3, 2000.4, 2000.0, 2000.2, 55, 0, 0)]))
    assert list(df.index) == [pd.Timestamp("2024-03-01 00:00"), pd.Timestamp("2024-03-01 00:01")]
    assert df["spread"].tolist() == [18, 0] and df["tickvol"].tolist() == [75, 55]


def test_parse_comma_python_style_with_epoch_time():
    txt = "time,open,high,low,close,tick_volume,spread,real_volume\n1709251200,2000,2001,1999,2000.5,10,20,0\n1709251260,2000.5,2001,2000,2000.7,12,20,0\n"
    df = mt5csv.parse_mt5_csv(io.StringIO(txt))
    assert df.index[0] == pd.Timestamp("2024-03-01 00:00") and df["tickvol"].tolist() == [10, 12]


def test_parse_rejects_non_mt5_text():
    with pytest.raises(mt5csv.ImportErrorMT5):
        mt5csv.parse_mt5_csv(io.StringIO("hello world\nnot a csv\n"))
    with pytest.raises(mt5csv.ImportErrorMT5):
        mt5csv.parse_mt5_csv(io.StringIO("a,b,c\n1,2,3\n"))


def test_convert_units_and_cleaning():
    rows = [("2024.03.01", "00:00:00", 2000.0, 2000.5, 1999.5, 2000.2, 10, 0, 25),
            ("2024.03.01", "00:01:00", 2000.2, 2000.1, 2000.0, 2000.6, 12, 0, 0),         # high below close -> envelope widened; spread 0 -> unknown
            ("2024.03.01", "00:02:00", 2000.6, 2001.0, 2000.4, 2000.9, 8, 0, 12),
            ("2024.03.01", "00:02:00", 2000.7, 2001.1, 2000.5, 2001.0, 9, 0, 14),         # duplicate timestamp: last wins
            ("2024.03.01", "00:03:00", -1, 2000, 1999, 2000, 1, 0, 10)]                   # non-positive price: dropped
    out, rep = mt5csv.convert(mt5csv.parse_mt5_csv(tab_csv(rows)), tz="utc", point=0.01)
    assert list(out.columns) == store.BAR_COLS and str(out.index.tz) == "UTC" and out.index.name == "ts"
    assert len(out) == 3
    assert out["spread"].iloc[0] == pytest.approx(0.25) and np.isnan(out["spread"].iloc[1]) and out["spread"].iloc[2] == pytest.approx(0.14)
    assert out["high"].iloc[1] == 2000.6 and rep.fixed["ohlc_envelope_widened"] == 1
    assert rep.dropped["duplicate_timestamps"] == 1 and rep.dropped["missing_or_nonpositive_price"] == 1
    assert out["close"].iloc[2] == 2001.0
    assert (out["volume"] == out["ticks"]).all()                       # vol column is 0 in MT5 exports -> tick volume stands in


@pytest.mark.parametrize("rule,naive,utc", [
    ("utc", "2024-07-01 12:00", "2024-07-01 12:00"),
    ("+2", "2024-01-10 12:00", "2024-01-10 10:00"),
    ("-5", "2024-01-10 12:00", "2024-01-10 17:00"),
    ("ny+7", "2024-01-10 12:00", "2024-01-10 10:00"),                  # NY winter = UTC-5 -> server = UTC+2
    ("ny+7", "2024-07-10 12:00", "2024-07-10 09:00"),                  # NY summer = UTC-4 -> server = UTC+3
    ("Europe/Athens", "2024-07-10 12:00", "2024-07-10 09:00"),
])
def test_to_utc_rules(rule, naive, utc):
    got, bad = mt5csv.to_utc(pd.DatetimeIndex([naive]), rule)
    assert got[0] == pd.Timestamp(utc, tz="UTC") and bad == 0


def test_unknown_timezone_rule_is_an_error():
    with pytest.raises(mt5csv.ImportErrorMT5):
        mt5csv.to_utc(pd.DatetimeIndex(["2024-01-10"]), "Mars/Olympus")


def synthetic_weeks(first_sunday="2024-01-07", weeks=70):
    """Half-hourly bars from Sunday 18:00 New York to Friday 16:59 New York, as UTC."""
    parts = []
    for k in range(weeks):
        sun = pd.Timestamp(first_sunday) + pd.Timedelta(weeks=k)
        o = (sun + pd.Timedelta(hours=18)).tz_localize(mt5csv.NY).tz_convert("UTC")
        c = (sun + pd.Timedelta(days=5, hours=17)).tz_localize(mt5csv.NY).tz_convert("UTC")
        parts.append(pd.date_range(o, c, freq="30min", inclusive="left").append(pd.DatetimeIndex([c - pd.Timedelta(minutes=1)])))
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def to_server_naive(utc_idx, rule):
    if rule == "utc":
        return utc_idx.tz_localize(None)
    if rule.startswith("+"):
        return (utc_idx + pd.Timedelta(hours=float(rule))).tz_localize(None)
    if rule == "ny+7":
        return (utc_idx.tz_convert(mt5csv.NY) + pd.Timedelta(hours=7)).tz_localize(None)
    return utc_idx.tz_convert(rule).tz_localize(None)


@pytest.mark.parametrize("truth", ["utc", "+2", "+3", "ny+7", "Europe/Athens"])
def test_detect_timezone_recovers_the_true_clock(truth):
    naive = to_server_naive(synthetic_weeks(), truth)
    rule, rank = mt5csv.choose_timezone(naive)
    assert rule == truth, rank


def test_detection_refuses_when_unsure_or_too_short():
    with pytest.raises(mt5csv.ImportErrorMT5):
        mt5csv.choose_timezone(to_server_naive(synthetic_weeks(weeks=5), "utc"))
    rng = pd.date_range("2024-01-01", periods=70 * 24 * 7, freq="1h")          # continuous 24x7 data: no weekly structure at all
    with pytest.raises(mt5csv.ImportErrorMT5):
        mt5csv.choose_timezone(rng)


def test_end_to_end_round_trip_to_lake(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "BAR_DIR", tmp_path)
    idx = to_server_naive(synthetic_weeks(weeks=20), "ny+7")
    rng = np.random.default_rng(0)
    px = 2000 + np.cumsum(rng.normal(0, .3, len(idx)))
    raw = pd.DataFrame({"open": px, "high": px + .5, "low": px - .5, "close": px + .1, "tickvol": 5.0, "vol": 0.0, "spread": 12.0}, index=pd.DatetimeIndex(idx, name="ts"))
    out, rep = mt5csv.convert(raw, tz="auto")
    assert rep.timezone_rule == "ny+7"
    assert not mt5csv.has_bars("T")
    store.write_bars(out, "T", "M1")
    back = store.read_bars("T", "M1")
    assert len(back) == len(out) and back.index[0] == out.index[0] and back["spread"].iloc[0] == pytest.approx(0.12)
    assert mt5csv.has_bars("T") and mt5csv.read_source("T") is None
