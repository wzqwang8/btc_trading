"""
Coinbase Trader CLI
===================
A simple command-line interface for viewing balances, prices,
placing orders, and managing open orders on Coinbase Advanced Trade.

Usage:
  python3 cb_trader.py status                        # all balances
  python3 cb_trader.py price BTC-GBP                 # live price
  python3 cb_trader.py buy   BTC-GBP 100             # market buy £100
  python3 cb_trader.py sell  BTC-GBP 100             # market sell £100
  python3 cb_trader.py limit-buy  BTC-GBP 100 80000  # limit buy £100 at £80,000
  python3 cb_trader.py limit-sell BTC-GBP 100 90000  # limit sell £100 at £90,000
  python3 cb_trader.py orders                        # list open orders
  python3 cb_trader.py cancel <order_id>             # cancel an order
"""

import sys
import uuid
import argparse
from coinbase.rest import RESTClient

API_KEY_FILE = "/Users/williamwang/Documents/python_scripts/btc_bot/cdp_api_key.json"
client = RESTClient(key_file=API_KEY_FILE)


def cmd_status(_args):
    """Print all non-zero account balances."""
    response = client.get_accounts()
    accounts = response.accounts

    print(f"\n{'Asset':<10} {'Available':>16} {'Hold':>16}")
    print("-" * 44)
    for a in accounts:
        available = float(a.available_balance["value"])
        hold = float(a.hold["value"])
        if available > 0 or hold > 0:
            currency = a.currency
            print(f"{currency:<10} {available:>16.8f} {hold:>16.8f}")
    print()


def cmd_price(args):
    """Show live price and basic stats for a trading pair."""
    product_id = args.pair.upper()
    try:
        product = client.get_product(product_id)
        price = float(product["price"])
        base, quote = product_id.split("-")
        pct_24h = float(product.get("price_percentage_change_24h", 0))
        vol_24h = float(product.get("volume_24h", 0))
        high_24h = float(product.get("price_increment", 0))

        print(f"\n  {product_id}")
        print(f"  Price:       £{price:,.4f}")
        print(f"  24h Change:  {pct_24h:+.2f}%")
        print(f"  24h Volume:  {vol_24h:,.2f} {base}")
        print(f"  5% lower:    £{price * 0.95:,.4f}")
        print(f"  5% higher:   £{price * 1.05:,.4f}\n")
    except Exception as e:
        print(f"Error fetching price for {product_id}: {e}")
        sys.exit(1)


def cmd_buy(args):
    """Place a market buy order."""
    product_id = args.pair.upper()
    amount = str(args.amount)

    product = client.get_product(product_id)
    price = float(product["price"])
    print(f"\n  Market BUY on {product_id}")
    print(f"  Spend:       £{float(amount):,.2f}")
    print(f"  ~Est. units: {float(amount) / price:.8f}")

    confirm = input("\n  Confirm order? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order = client.market_order_buy(
        product_id=product_id,
        quote_size=amount,
        client_order_id=str(uuid.uuid4()),
    )
    _print_order_result(order)


def cmd_sell(args):
    """Place a market sell order."""
    product_id = args.pair.upper()
    amount = str(args.amount)

    product = client.get_product(product_id)
    price = float(product["price"])
    base = product_id.split("-")[0]
    print(f"\n  Market SELL on {product_id}")
    print(f"  Spend:       {float(amount):,.8f} {base}")
    print(f"  ~Est. value: £{float(amount) * price:,.2f}")

    confirm = input("\n  Confirm order? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order = client.market_order_sell(
        product_id=product_id,
        base_size=amount,
        client_order_id=str(uuid.uuid4()),
    )
    _print_order_result(order)


def cmd_limit_buy(args):
    """Place a limit buy order."""
    product_id = args.pair.upper()
    amount = str(args.amount)
    limit_price = str(args.price)

    print(f"\n  Limit BUY on {product_id}")
    print(f"  Spend:       £{float(amount):,.2f}")
    print(f"  At price:    £{float(limit_price):,.2f}")
    print(f"  ~Est. units: {float(amount) / float(limit_price):.8f}")

    confirm = input("\n  Confirm order? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order = client.limit_order_gtc_buy(
        product_id=product_id,
        quote_size=amount,
        limit_price=limit_price,
        client_order_id=str(uuid.uuid4()),
        post_only=True,
    )
    _print_order_result(order)


def cmd_limit_sell(args):
    """Place a limit sell order."""
    product_id = args.pair.upper()
    amount = str(args.amount)
    limit_price = str(args.price)
    base = product_id.split("-")[0]

    print(f"\n  Limit SELL on {product_id}")
    print(f"  Sell:        {float(amount):,.8f} {base}")
    print(f"  At price:    £{float(limit_price):,.2f}")
    print(f"  ~Est. value: £{float(amount) * float(limit_price):,.2f}")

    confirm = input("\n  Confirm order? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order = client.limit_order_gtc_sell(
        product_id=product_id,
        base_size=amount,
        limit_price=limit_price,
        client_order_id=str(uuid.uuid4()),
        post_only=True,
    )
    _print_order_result(order)


def cmd_orders(_args):
    """List all open orders."""
    response = client.list_orders(order_status="OPEN")
    orders = response.orders

    if not orders:
        print("\n  No open orders.\n")
        return

    print(f"\n  {'Order ID':<38} {'Pair':<12} {'Side':<6} {'Type':<8} {'Size/Funds':>14} {'Limit Price':>14}")
    print("  " + "-" * 96)
    for o in orders:
        config = o.order_configuration
        side = o.side
        pair = o.product_id

        # Pull size/price from the config dict
        if hasattr(config, "limit_limit_gtc"):
            cfg = config.limit_limit_gtc
            size = cfg.base_size
            lp = f"£{float(cfg.limit_price):,.2f}"
            otype = "LIMIT"
        elif hasattr(config, "market_market_ioc"):
            cfg = config.market_market_ioc
            size = getattr(cfg, "quote_size", None) or getattr(cfg, "base_size", "?")
            lp = "MARKET"
            otype = "MARKET"
        else:
            size = "?"
            lp = "?"
            otype = "?"

        print(f"  {o.order_id:<38} {pair:<12} {side:<6} {otype:<8} {str(size):>14} {lp:>14}")
    print()


def cmd_cancel(args):
    """Cancel an open order by ID."""
    order_id = args.order_id
    confirm = input(f"\n  Cancel order {order_id}? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    result = client.cancel_orders(order_ids=[order_id])
    results = result.results
    if results and results[0].success:
        print(f"  ✅ Order {order_id} cancelled.")
    else:
        print(f"  ❌ Failed to cancel: {results}")


def _print_order_result(order):
    """Pretty-print an order response."""
    if order.success:
        o = order.success_response
        print(f"\n  ✅ Order placed")
        print(f"  Order ID: {o.order_id}")
        print(f"  Product:  {o.product_id}")
        print(f"  Side:     {o.side}\n")
    else:
        err = order.error_response
        print(f"\n  ❌ Order failed: {err.message} — {err.preview_failure_reason}\n")


# ── CLI setup ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coinbase Trader CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    sub.add_parser("status", help="Show all non-zero account balances")

    p_price = sub.add_parser("price", help="Show live price for a pair")
    p_price.add_argument("pair", help="e.g. BTC-GBP")

    p_buy = sub.add_parser("buy", help="Market buy (spend GBP amount)")
    p_buy.add_argument("pair", help="e.g. BTC-GBP")
    p_buy.add_argument("amount", type=float, help="GBP amount to spend")

    p_sell = sub.add_parser("sell", help="Market sell (sell base amount)")
    p_sell.add_argument("pair", help="e.g. BTC-GBP")
    p_sell.add_argument("amount", type=float, help="Base currency amount to sell")

    p_lb = sub.add_parser("limit-buy", help="Limit buy order (post-only)")
    p_lb.add_argument("pair", help="e.g. BTC-GBP")
    p_lb.add_argument("amount", type=float, help="GBP amount to spend")
    p_lb.add_argument("price", type=float, help="Limit price in GBP")

    p_ls = sub.add_parser("limit-sell", help="Limit sell order (post-only)")
    p_ls.add_argument("pair", help="e.g. BTC-GBP")
    p_ls.add_argument("amount", type=float, help="Base currency amount to sell")
    p_ls.add_argument("price", type=float, help="Limit price in GBP")

    sub.add_parser("orders", help="List open orders")

    p_cancel = sub.add_parser("cancel", help="Cancel an order by ID")
    p_cancel.add_argument("order_id", help="Order ID to cancel")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "price": cmd_price,
        "buy": cmd_buy,
        "sell": cmd_sell,
        "limit-buy": cmd_limit_buy,
        "limit-sell": cmd_limit_sell,
        "orders": cmd_orders,
        "cancel": cmd_cancel,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
