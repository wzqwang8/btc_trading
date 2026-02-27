"""
UK Crypto Capital Gains Tax Calculator — Koinly Replacement
============================================================
Fetches fills from Coinbase Advanced Trade API and produces:
  - Per-asset CSV gain reports
  - Comprehensive multi-asset PDF (like Koinly's tax report)
  - SA108-formatted PDFs per tax year (ready for HMRC Self Assessment)

HMRC share-pooling rules implemented correctly:
  Rule 1 — Same-day match: sells matched against buys on the same calendar day
  Rule 2 — 30-day bed & breakfast: sells matched against buys in the 30 days AFTER the sell
  Rule 3 — Section 104 pool: remaining sells matched against the running average-cost pool

Usage:
  python tax_return.py                        # fetch live, all years
  python tax_return.py --tax-year 2024/25     # report for 2024/25 only
  python tax_return.py --use-cache            # use existing all_fills.csv
  python tax_return.py --use-cache --tax-year 2024/25
"""

import sys
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, date as date_type
from coinbase.rest import RESTClient
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY_FILE = "/Users/williamwang/Documents/python_scripts/btc_bot/cdp_api_key.json"

# ── Historical FX rate cache ──────────────────────────────────────────────────

_fx_cache: dict = {}  # (from_currency, date_str) → float


def get_historical_gbp_rate(from_currency: str, trade_time) -> float:
    """
    Return the GBP rate for *from_currency* on the calendar day of *trade_time*,
    fetched from Coinbase Exchange's public candle endpoint (no auth required).
    Falls back to the live Coinbase Advanced Trade price if the candle API fails,
    then to 1.0 if both fail.
    """
    if from_currency == "GBP":
        return 1.0

    dt = trade_time
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if hasattr(dt, "date"):
        day = dt.date()
    else:
        day = dt

    date_str = day.isoformat()
    cache_key = (from_currency, date_str)
    if cache_key in _fx_cache:
        return _fx_cache[cache_key]

    product_id = f"{from_currency}-GBP"
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)

    # Try Coinbase Exchange REST (public, no auth)
    try:
        resp = requests.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "granularity": 86400,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            candles = resp.json()
            if candles:
                # candle format: [time, low, high, open, close, volume]
                rate = float(candles[0][4])  # use close price
                _fx_cache[cache_key] = rate
                return rate
    except Exception as e:
        print(f"   ⚠️  Candle API failed for {product_id} on {date_str}: {e}")

    # Fallback: USDC/USDT → try via USD-GBP
    if from_currency in ("USDC", "USDT"):
        rate = get_historical_gbp_rate("USD", trade_time)
        _fx_cache[cache_key] = rate
        return rate

    # Fallback: live price from Coinbase Advanced Trade
    try:
        client = RESTClient(key_file=API_KEY_FILE)
        product = client.get_product(product_id)
        rate = float(product["price"])
        print(f"   ⚠️  Using live rate for {product_id}: {rate:.6f} (historical unavailable)")
        _fx_cache[cache_key] = rate
        return rate
    except Exception:
        pass

    print(f"   ⚠️  Could not get GBP rate for {from_currency} on {date_str}. Defaulting to 1.0")
    _fx_cache[cache_key] = 1.0
    return 1.0


# ── Fetch fills from Coinbase ─────────────────────────────────────────────────

def get_all_fills() -> pd.DataFrame:
    """
    Fetch all trade fills from Coinbase Advanced Trade API.
    Converts every trade's value and fee to GBP using the historical exchange
    rate at the time of each trade.
    """
    client = RESTClient(key_file=API_KEY_FILE)
    print("📥 Fetching fills from Coinbase Advanced Trade...")
    fills = client.get_fills()
    fills_data = fills.fills

    if not fills_data:
        print("⚠️  No fills returned.")
        return pd.DataFrame()

    records = []
    for f in fills_data:
        price = float(f.price)
        size = float(f.size)
        fee = float(f.commission) if hasattr(f, "commission") else 0.0
        side = f.side.upper()
        product_id = f.product_id
        base_currency, quote_currency = product_id.split("-")
        fee_currency = getattr(f, "fee_currency", quote_currency)
        trade_time = pd.to_datetime(f.trade_time, utc=True)

        # Historical GBP conversion rates at trade time
        quote_gbp = get_historical_gbp_rate(quote_currency, trade_time)
        fee_gbp_rate = (
            get_historical_gbp_rate(fee_currency, trade_time)
            if fee_currency != quote_currency
            else quote_gbp
        )

        trade_value_gbp = price * size * quote_gbp
        fee_value_gbp = fee * fee_gbp_rate

        # For buys, cost includes the fee; for sells, proceeds are net of fee
        if side == "BUY":
            total_gbp = trade_value_gbp + fee_value_gbp
        else:
            total_gbp = trade_value_gbp - fee_value_gbp

        records.append(
            {
                "trade_time": trade_time,
                "side": side,
                "product_id": product_id,
                "base_currency": base_currency,
                "quote_currency": quote_currency,
                "price": price,
                "size": size,
                "fee": fee,
                "fee_currency": fee_currency,
                "quote_to_gbp": quote_gbp,
                "trade_value_gbp": trade_value_gbp,
                "fee_value_gbp": fee_value_gbp,
                "total_gbp": total_gbp,
            }
        )

    df = pd.DataFrame(records).sort_values("trade_time").reset_index(drop=True)
    df.to_csv("all_fills.csv", index=False)
    print(f"✅ Fetched {len(df)} fills across {df['product_id'].nunique()} products.")
    print("💾 Saved to all_fills.csv")
    return df


# ── UK CGT three-rule share pooling ──────────────────────────────────────────

def compute_uk_cgt(df: pd.DataFrame, asset_label: str):
    """
    Calculate UK CGT for a single asset using HMRC's three matching rules.

    Two-pass algorithm:
      Pass 1 — identify same-day and 30-day matches (consuming buy units)
      Pass 2 — build Section 104 pool from remaining buy units chronologically,
               then match remaining sell units against the pool

    Returns:
      gains_df        — DataFrame of disposal events with gain/loss
      pool_units      — units remaining in Section 104 pool after all disposals
      pool_cost       — total GBP cost of units remaining in pool
    """
    df = df.sort_values("trade_time").reset_index(drop=True)

    # Build buy/sell lists with tracking fields
    buys = []
    for _, row in df[df["side"] == "BUY"].iterrows():
        units = float(row["size"])
        total_gbp = float(row["total_gbp"])
        cost_per_unit = total_gbp / units if units > 0 else 0.0
        buys.append(
            {
                "date": row["trade_time"].date(),
                "units": units,
                "cost_per_unit": cost_per_unit,
                "remaining": units,       # units not yet matched or pooled
                "in_pool": False,         # whether already added to Section 104 pool
            }
        )

    sells = []
    for _, row in df[df["side"] == "SELL"].iterrows():
        sells.append(
            {
                "datetime": row["trade_time"],
                "date": row["trade_time"].date(),
                "units": float(row["size"]),
                "proceeds": float(row["total_gbp"]),
                "matches": [],            # list of (rule_label, qty, cost_per_unit)
                "pool_qty": 0.0,          # quantity to match from pool (set in pass 1)
            }
        )

    if not sells:
        # No sells — accumulate pool and return empty gains
        pool_units = sum(b["units"] for b in buys)
        pool_cost = sum(b["units"] * b["cost_per_unit"] for b in buys)
        return pd.DataFrame(), pool_units, pool_cost

    # ── Pass 1: same-day and 30-day matching ─────────────────────────────────
    for sell in sorted(sells, key=lambda s: s["datetime"]):
        sell_date = sell["date"]
        remaining = sell["units"]

        # Rule 1 — same-day
        for buy in buys:
            if remaining < 1e-10:
                break
            if buy["date"] == sell_date and buy["remaining"] > 1e-10:
                match = min(remaining, buy["remaining"])
                sell["matches"].append(("Same-Day", match, buy["cost_per_unit"]))
                buy["remaining"] -= match
                remaining -= match

        # Rule 2 — 30-day bed & breakfast (buys AFTER the sell, within 30 days)
        bb_cutoff = sell_date + timedelta(days=30)
        future_buys = sorted(
            [b for b in buys if sell_date < b["date"] <= bb_cutoff and b["remaining"] > 1e-10],
            key=lambda b: b["date"],  # FIFO within the window
        )
        for buy in future_buys:
            if remaining < 1e-10:
                break
            match = min(remaining, buy["remaining"])
            sell["matches"].append(("30-Day B&B", match, buy["cost_per_unit"]))
            buy["remaining"] -= match
            remaining -= match

        sell["pool_qty"] = remaining  # what must be matched from pool

    # ── Pass 2: Section 104 pool matching ────────────────────────────────────
    pool_units = 0.0
    pool_cost = 0.0
    results = []

    for sell in sorted(sells, key=lambda s: s["datetime"]):
        sell_date = sell["date"]

        # Absorb all buys that occurred strictly before this sell into the pool
        # (only buys not already consumed by same-day / 30-day matching)
        for buy in buys:
            if buy["date"] < sell_date and not buy["in_pool"] and buy["remaining"] > 1e-10:
                pool_units += buy["remaining"]
                pool_cost += buy["remaining"] * buy["cost_per_unit"]
                buy["in_pool"] = True

        # Cost basis from rule 1 and rule 2 matches
        cost_basis = sum(m[1] * m[2] for m in sell["matches"])
        match_labels = list(dict.fromkeys(m[0] for m in sell["matches"]))  # preserve order, dedupe

        # Pool match for remaining quantity
        pool_qty = sell["pool_qty"]
        if pool_qty > 1e-10:
            if pool_units >= pool_qty - 1e-10:
                avg_pool_cost = pool_cost / pool_units
                pool_cost_used = pool_qty * avg_pool_cost
                cost_basis += pool_cost_used
                pool_units -= pool_qty
                pool_cost -= pool_cost_used
                match_labels.append("Pool")
            else:
                # Use whatever is in the pool (orphaned sell — shouldn't happen on Coinbase)
                if pool_units > 1e-10:
                    avg_pool_cost = pool_cost / pool_units
                    cost_basis += pool_units * avg_pool_cost
                    pool_cost = 0.0
                    pool_units = 0.0
                    match_labels.append("Pool (partial)")
                print(
                    f"   ⚠️  Insufficient pool for {asset_label} disposal on {sell_date}. "
                    f"Needed {pool_qty:.8f}, had {pool_units:.8f}"
                )

        gain = sell["proceeds"] - cost_basis
        results.append(
            {
                "Sell Date": sell["datetime"],
                "Asset": asset_label,
                "Units Sold": sell["units"],
                "Proceeds (GBP)": sell["proceeds"],
                "Cost Basis (GBP)": cost_basis,
                "Gain (GBP)": gain,
                "Match Type": "+".join(match_labels) if match_labels else "Unmatched",
            }
        )

    # Absorb any remaining buys that came after all sells
    for buy in buys:
        if not buy["in_pool"] and buy["remaining"] > 1e-10:
            pool_units += buy["remaining"]
            pool_cost += buy["remaining"] * buy["cost_per_unit"]
            buy["in_pool"] = True

    gains_df = pd.DataFrame(results) if results else pd.DataFrame()
    return gains_df, pool_units, pool_cost


# ── Tax year helpers ──────────────────────────────────────────────────────────

def get_uk_tax_year(dt) -> str:
    """Return tax year label like '2024/25' for a given date or datetime."""
    if hasattr(dt, "date"):
        d = dt.date()
    elif isinstance(dt, date_type):
        d = dt
    else:
        d = pd.Timestamp(dt).date()
    year = d.year
    start_year = year - 1 if (d.month < 4 or (d.month == 4 and d.day < 6)) else year
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def tax_year_date_range(tax_year_str: str):
    """Return (start_date, end_date) for a tax year string like '2024/25'."""
    start_year = int(tax_year_str.split("/")[0])
    return (
        date_type(start_year, 4, 6),
        date_type(start_year + 1, 4, 5),
    )


# ── Table styling ─────────────────────────────────────────────────────────────

def _table_style(header_bg="#3D5A80"):
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.white),
        ]
    )


# ── Main PDF report ───────────────────────────────────────────────────────────

def export_pdf_report(
    all_gains: list,
    portfolio_summary: dict,
    filename: str = "crypto_tax_report.pdf",
    tax_year_label: str = None,
):
    """
    Generate the comprehensive multi-asset PDF report.
    Mirrors Koinly's layout: cover → CGT summary → portfolio → per-asset detail.
    """
    print(f"\n📄 Generating PDF report: {filename}")
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    elements = []

    # ── Cover page ────────────────────────────────────────────────────────────
    year_display = tax_year_label or "All Tax Years"
    elements += [
        Paragraph(f"Crypto Tax Report — {year_display}", styles["Title"]),
        Spacer(1, 6),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", styles["Normal"]),
    ]
    if tax_year_label and "/" in tax_year_label:
        try:
            sy = int(tax_year_label.split("/")[0])
            elements.append(
                Paragraph(f"Period: 6 April {sy} – 5 April {sy + 1}", styles["Normal"])
            )
        except ValueError:
            pass

    elements += [
        Spacer(1, 12),
        Paragraph(
            "Capital gains calculated using HMRC Share Pooling rules: "
            "(1) Same-day match, (2) 30-day bed &amp; breakfast rule, (3) Section 104 pool.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "<i>This report is for informational purposes. Verify accuracy with a qualified "
            "tax adviser before submitting your Self Assessment.</i>",
            styles["Normal"],
        ),
        PageBreak(),
    ]

    # ── CGT summary by tax year ───────────────────────────────────────────────
    if all_gains:
        combined = pd.concat(all_gains, ignore_index=True)
        combined["Gain (GBP)"] = pd.to_numeric(combined["Gain (GBP)"], errors="coerce")
        combined["Proceeds (GBP)"] = pd.to_numeric(combined["Proceeds (GBP)"], errors="coerce")
        combined["Cost Basis (GBP)"] = pd.to_numeric(combined["Cost Basis (GBP)"], errors="coerce")
        combined["Sell Date"] = pd.to_datetime(combined["Sell Date"])
        combined["Tax Year"] = combined["Sell Date"].apply(get_uk_tax_year)

        elements.append(Paragraph("Capital Gains Summary by Tax Year", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        summary_rows = [
            ["Tax Year", "Disposals", "Proceeds (£)", "Allowable Cost (£)", "Gains (£)", "Losses (£)", "Net (£)"]
        ]
        for ty in sorted(combined["Tax Year"].unique()):
            grp = combined[combined["Tax Year"] == ty]
            gains = grp[grp["Gain (GBP)"] > 0]["Gain (GBP)"].sum()
            losses = -grp[grp["Gain (GBP)"] < 0]["Gain (GBP)"].sum()
            summary_rows.append(
                [
                    ty,
                    str(len(grp)),
                    f"{grp['Proceeds (GBP)'].sum():,.2f}",
                    f"{grp['Cost Basis (GBP)'].sum():,.2f}",
                    f"{gains:,.2f}",
                    f"{losses:,.2f}",
                    f"{gains - losses:,.2f}",
                ]
            )

        t = Table(summary_rows, repeatRows=1)
        t.setStyle(_table_style())
        elements += [t, Spacer(1, 24)]

    # ── Current portfolio ─────────────────────────────────────────────────────
    if portfolio_summary:
        elements.append(Paragraph("Current Portfolio (Section 104 Pool Balances)", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        port_rows = [["Asset", "Units Held", "Avg Cost/Unit (£)", "Current Price (£)", "Market Value (£)", "Unrealised P&amp;L (£)"]]
        for asset, info in portfolio_summary.items():
            pnl = (info["current_price"] - info["avg_cost"]) * info["units"]
            port_rows.append(
                [
                    asset,
                    f"{info['units']:.8f}",
                    f"{info['avg_cost']:,.4f}",
                    f"{info['current_price']:,.4f}",
                    f"{info['market_value']:,.2f}",
                    f"{pnl:+,.2f}",
                ]
            )

        t = Table(port_rows, repeatRows=1)
        t.setStyle(_table_style())
        elements += [t, Spacer(1, 24)]

    elements.append(PageBreak())

    # ── Per-asset disposal detail ─────────────────────────────────────────────
    if all_gains:
        elements.append(Paragraph("Detailed Disposals by Asset and Tax Year", styles["Heading1"]))
        elements.append(Spacer(1, 12))

        for gains_df in all_gains:
            if gains_df.empty:
                continue
            gains_df = gains_df.copy()
            gains_df["Sell Date"] = pd.to_datetime(gains_df["Sell Date"])
            gains_df["Tax Year"] = gains_df["Sell Date"].apply(get_uk_tax_year)
            asset = gains_df["Asset"].iloc[0]

            for ty in sorted(gains_df["Tax Year"].unique()):
                grp = gains_df[gains_df["Tax Year"] == ty].copy()
                net = grp["Gain (GBP)"].sum()
                gains_pos = grp[grp["Gain (GBP)"] > 0]["Gain (GBP)"].sum()
                losses_pos = -grp[grp["Gain (GBP)"] < 0]["Gain (GBP)"].sum()

                elements.append(Paragraph(f"{asset} — Tax Year {ty}", styles["Heading2"]))
                elements.append(Spacer(1, 4))
                elements.append(
                    Paragraph(
                        f"Disposals: <b>{len(grp)}</b> &nbsp;|&nbsp; "
                        f"Proceeds: <b>£{grp['Proceeds (GBP)'].sum():,.2f}</b> &nbsp;|&nbsp; "
                        f"Gains: <b>£{gains_pos:,.2f}</b> &nbsp;|&nbsp; "
                        f"Losses: <b>£{losses_pos:,.2f}</b> &nbsp;|&nbsp; "
                        f"Net: <b>£{net:,.2f}</b>",
                        styles["Normal"],
                    )
                )
                elements.append(Spacer(1, 8))

                # Format display columns
                disp = grp.copy()
                disp["Sell Date"] = disp["Sell Date"].dt.strftime("%Y-%m-%d")
                disp["Units Sold"] = disp["Units Sold"].map(lambda x: f"{float(x):.8f}")
                for col in ["Proceeds (GBP)", "Cost Basis (GBP)", "Gain (GBP)"]:
                    disp[col] = pd.to_numeric(disp[col]).map(lambda x: f"£{x:,.2f}")

                disp = disp.drop(columns=["Asset", "Tax Year"], errors="ignore")
                table_data = [list(disp.columns)] + disp.values.tolist()

                t = Table(table_data, repeatRows=1)
                t.setStyle(_table_style())
                elements += [t, Spacer(1, 24)]

    doc.build(elements)
    print(f"✅ PDF saved: {filename}")


# ── SA108 summary PDF ─────────────────────────────────────────────────────────

def export_sa108_pdf(summary: dict, tax_year: str, filename: str):
    """
    Generate a single-page SA108 summary for a given tax year.
    The user transfers these figures to their HMRC Self Assessment return.
    """
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    elements = []

    try:
        start_year = int(tax_year.split("/")[0])
        period = f"6 April {start_year} – 5 April {start_year + 1}"
    except (ValueError, IndexError):
        period = tax_year

    net = summary["gains"] - summary["losses"]

    def field(label, value):
        return [
            Paragraph(f"<b>{label}</b>", styles["Normal"]),
            Paragraph(str(value), styles["Normal"]),
            Spacer(1, 10),
        ]

    elements += [
        Paragraph(f"SA108 Capital Gains Summary — Tax Year {tax_year}", styles["Title"]),
        Spacer(1, 4),
        Paragraph(f"Period: {period}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y')}", styles["Normal"]),
        Spacer(1, 20),
        Paragraph("Other property, assets and gains (Cryptoassets)", styles["Heading2"]),
        Spacer(1, 12),
    ]

    for item in field("Box 14 — Number of disposals", summary["disposals"]):
        elements.append(item)
    for item in field("Box 15 — Disposal proceeds", f"£{summary['proceeds']:,.2f}"):
        elements.append(item)
    for item in field("Box 16 — Allowable costs (including purchase price)", f"£{summary['costs']:,.2f}"):
        elements.append(item)
    for item in field("Box 17 — Gains in the year, before losses", f"£{summary['gains']:,.2f}"):
        elements.append(item)
    for item in field("Box 18 — Losses used against this year's gains", "£0.00  (enter prior losses if applicable)"):
        elements.append(item)
    for item in field("Box 19 — Losses in the year", f"£{summary['losses']:,.2f}"):
        elements.append(item)

    elements += [
        Spacer(1, 20),
        Paragraph(f"<b>Net Capital Gain (or Loss) for {tax_year}: £{net:,.2f}</b>", styles["Heading2"]),
        Spacer(1, 20),
        Paragraph(
            "<i>How to use this report:</i>",
            styles["Heading3"] if "Heading3" in styles else styles["Normal"],
        ),
        Spacer(1, 6),
        Paragraph(
            "1. Log in to your HMRC Government Gateway account and open your Self Assessment return.",
            styles["Normal"],
        ),
        Spacer(1, 4),
        Paragraph(
            "2. Navigate to <b>SA108 – Capital Gains Summary</b> → "
            "<i>Other property, assets and gains</i>.",
            styles["Normal"],
        ),
        Spacer(1, 4),
        Paragraph(
            "3. Enter the figures above into boxes 14–19.",
            styles["Normal"],
        ),
        Spacer(1, 4),
        Paragraph(
            "4. The Annual Exempt Amount for 2024/25 is <b>£3,000</b>. "
            "If your net gain exceeds this, you will owe Capital Gains Tax.",
            styles["Normal"],
        ),
        Spacer(1, 20),
        Paragraph(
            "<i>This document is computer-generated. Verify all figures and consult a "
            "qualified accountant before submitting your tax return.</i>",
            styles["Normal"],
        ),
    ]

    doc.build(elements)
    print(f"✅ SA108 saved: {filename}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UK Crypto CGT Calculator — Koinly Replacement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tax-year",
        metavar="YYYY/YY",
        help="Filter output to a specific UK tax year, e.g. 2024/25",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load fills from all_fills.csv instead of hitting the Coinbase API",
    )
    args = parser.parse_args()

    # ── Load or fetch fills ──────────────────────────────────────────────────
    if args.use_cache:
        try:
            df_all = pd.read_csv("all_fills.csv")
            df_all["trade_time"] = pd.to_datetime(df_all["trade_time"], utc=True)
            print(f"📂 Loaded {len(df_all)} fills from all_fills.csv (cache).")
        except FileNotFoundError:
            print("⚠️  No cache found. Fetching from Coinbase API...")
            df_all = get_all_fills()
    else:
        df_all = get_all_fills()

    if df_all.empty:
        print("No data found. Exiting.")
        sys.exit(0)

    # ── Process each trading pair ────────────────────────────────────────────
    all_gains = []
    portfolio_summary = {}
    client_live = RESTClient(key_file=API_KEY_FILE)

    for pid in sorted(df_all["product_id"].unique()):
        base = pid.split("-")[0]
        df_asset = df_all[df_all["product_id"] == pid].copy()
        print(f"\n📊 Processing {pid} ({len(df_asset)} trades)...")

        gains_df, pool_units, pool_cost = compute_uk_cgt(df_asset, base)

        # Save per-asset CSV (all years, for reference)
        if not gains_df.empty:
            csv_name = f"{pid.lower().replace('-', '_')}_uk_tax_report.csv"
            gains_df.to_csv(csv_name, index=False)
            print(f"   💾 {csv_name}")

            # Filter gains to requested tax year for PDF/SA108
            filtered_df = gains_df.copy()
            if args.tax_year:
                filtered_df["Sell Date"] = pd.to_datetime(filtered_df["Sell Date"])
                filtered_df["_ty"] = filtered_df["Sell Date"].apply(get_uk_tax_year)
                filtered_df = filtered_df[filtered_df["_ty"] == args.tax_year].drop(
                    columns=["_ty"]
                )

            if not filtered_df.empty:
                all_gains.append(filtered_df)

        # Current portfolio position from Section 104 pool
        if pool_units > 1e-10:
            avg_cost = pool_cost / pool_units
            current_price = avg_cost  # fallback
            try:
                product = client_live.get_product(f"{base}-GBP")
                current_price = float(product["price"])
            except Exception:
                # Try via USDC or USD
                try:
                    p2 = client_live.get_product(f"{base}-USDC")
                    usdc_price = float(p2["price"])
                    current_price = usdc_price * get_historical_gbp_rate("USDC", datetime.now())
                except Exception:
                    pass

            portfolio_summary[base] = {
                "units": pool_units,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "market_value": pool_units * current_price,
            }
            print(
                f"   💰 Pool: {pool_units:.8f} {base} @ avg £{avg_cost:,.4f}/unit "
                f"| Current £{current_price:,.4f} | MV £{pool_units * current_price:,.2f}"
            )

    # ── Generate SA108 summaries ─────────────────────────────────────────────
    if all_gains:
        combined = pd.concat(all_gains, ignore_index=True)
        combined["Sell Date"] = pd.to_datetime(combined["Sell Date"])
        combined["Tax Year"] = combined["Sell Date"].apply(get_uk_tax_year)
        combined["Gain (GBP)"] = pd.to_numeric(combined["Gain (GBP)"], errors="coerce")
        combined["Proceeds (GBP)"] = pd.to_numeric(combined["Proceeds (GBP)"], errors="coerce")
        combined["Cost Basis (GBP)"] = pd.to_numeric(combined["Cost Basis (GBP)"], errors="coerce")

        for ty in sorted(combined["Tax Year"].unique()):
            grp = combined[combined["Tax Year"] == ty]
            gains = grp[grp["Gain (GBP)"] > 0]["Gain (GBP)"].sum()
            losses = -grp[grp["Gain (GBP)"] < 0]["Gain (GBP)"].sum()
            sa108_summary = {
                "disposals": len(grp),
                "proceeds": grp["Proceeds (GBP)"].sum(),
                "costs": grp["Cost Basis (GBP)"].sum(),
                "gains": gains,
                "losses": losses,
            }
            safe_ty = ty.replace("/", "-")
            export_sa108_pdf(sa108_summary, ty, filename=f"SA108_{safe_ty}.pdf")

        # ── Generate main tax report PDF ─────────────────────────────────────
        if args.tax_year:
            year_label = args.tax_year
        else:
            years = sorted(combined["Tax Year"].unique())
            year_label = f"{years[0]}–{years[-1]}" if len(years) > 1 else years[0]

        export_pdf_report(
            all_gains,
            portfolio_summary,
            filename="crypto_tax_report.pdf",
            tax_year_label=year_label,
        )

        print("\n" + "=" * 60)
        print("✅ All reports generated successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  📄 crypto_tax_report.pdf   — full multi-asset report")
        print("  📄 SA108_XXXX-XX.pdf       — one per tax year (for HMRC)")
        print("  📊 {pair}_uk_tax_report.csv — one per trading pair")

        # Print a quick console summary
        print("\n📋 Quick Summary:")
        for ty in sorted(combined["Tax Year"].unique()):
            grp = combined[combined["Tax Year"] == ty]
            net = grp["Gain (GBP)"].sum()
            print(
                f"  {ty}: {len(grp)} disposals | "
                f"Proceeds £{grp['Proceeds (GBP)'].sum():,.2f} | "
                f"Net {'gain' if net >= 0 else 'loss'} £{abs(net):,.2f}"
            )
    else:
        print("\n⚠️  No disposal events found for the selected tax year(s).")
        print("    (Buys-only positions are tracked in the portfolio summary above.)")


if __name__ == "__main__":
    main()
