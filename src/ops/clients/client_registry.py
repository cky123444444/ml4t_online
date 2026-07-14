import yaml
from src.ops.clients.binance_futures_client import BinanceFuturesClient
from src.utils.logger import setup_logger

logger = setup_logger("client_registry")


class ClientRegistry:
    _clients = {}

    @classmethod
    def init_from_config(cls, config_path="config/accounts.yaml", testnet=True):
        """
        Initialize clients from a YAML configuration file.

        YAML format:
        accounts:
          - name: account1
            api_key: xxx
            api_secret: xxx
          - name: account2
            api_key: yyy
            api_secret: yyy
        """
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read config {config_path}: {e}")
            raise

        accounts = cfg.get("accounts", [])
        for acc in accounts:
            name = acc["name"]
            api_key = acc["api_key"]
            api_secret = acc["api_secret"]
            cls._clients[name] = BinanceFuturesClient(api_key, api_secret, testnet=testnet)
            logger.info(f"Initialized client for account: {name}")

    @classmethod
    def get_client(cls, account_name):
        """
        Retrieve the BinanceFuturesClient instance for the given account name.
        """
        logger.info(f"Retrieving client for account: {account_name}")
        return cls._clients.get(account_name)

    @classmethod
    def all_clients(cls):
        """
        Return the dict of all account_name -> client instances
        """
        return cls._clients
    
    @classmethod
    def get_total_asset_btc_sum(cls) -> float:
        """
        Sum total BTC-equivalent asset across all accounts
        """
        total_btc = 0.0
        for name, client in cls._clients.items():
            try:
                asset_btc = client.get_total_asset_btc()
                total_btc += asset_btc
            except Exception as e:
                logger.warning(f"Failed to fetch total BTC asset for {name}: {e}")
        return total_btc
