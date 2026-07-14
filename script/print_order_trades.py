import yaml
from src.ops.clients.binance_futures_client import BinanceFuturesClient
from src.utils.calculate_helper import calc_avg_price, calc_commission
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True, help="Account name to fetch trades for")
    parser.add_argument("--order_id", required=True, help="Order ID to fetch trades for")
    args = parser.parse_args()

    account_name = args.account
    order_id = args.order_id

    # Load API credentials from accounts.yaml
    with open("src/config/accounts.yaml", "r") as file:
        accounts_data = yaml.safe_load(file)

    accounts = {account['name']: account for account in accounts_data['accounts']}

    account = accounts.get(account_name)
    if not account:
        print(f"Error: No account found for '{account_name}' in accounts.yaml")
        return

    api_key = account.get("api_key")
    api_secret = account.get("api_secret")

    if not api_key or not api_secret:
        print(f"Error: Missing API credentials for account '{account_name}'")
        return

    # Initialize Binance client directly
    client = BinanceFuturesClient(api_key=api_key, api_secret=api_secret, testnet=True)

    try:
        # Fetch trades for the given order ID
        trades = client.get_order_trades(order_id)

        if not trades:
            print(f"No trades found for order ID '{order_id}'")
            return

        # Print raw trades
        print(f"Raw trades from BINANCESDK, account='{account_name}':")
        for trade in trades:
            print(trade)

        # Calculate and print average price and commission
        avg_price = calc_avg_price(trades)
        commission = calc_commission(trades)

        print(f"\nCalculated average price: {avg_price}")
        print(f"Calculated commission: {commission}")

        print(f"\nTo update the EXIT PRICE in order.db with these values, run:")
        print(f"UPDATE orders SET exit_price={avg_price}, commission=commission+{commission} WHERE id=;")

        print(f"\nTo update the ENTRY PRICE in order.db with these values, run:")
        print(f"UPDATE orders SET entry_price={avg_price}, commission=commission+{commission} WHERE id=;")


    except Exception as e:
        print(f"Error while fetching trades or calculating values: {e}")

if __name__ == "__main__":
    main()