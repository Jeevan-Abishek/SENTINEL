try:
    from dashboard.services.api_client import SentinelAPIClient, api_client
except ImportError:
    from sentinel.dashboard.services.api_client import SentinelAPIClient, api_client

__all__ = ["SentinelAPIClient", "api_client"]
