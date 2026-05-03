"""MT5 connection manager with investor-account support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MT5ConnectionError(Exception):
    """Raised when the MT5 terminal cannot be reached or authenticated."""


class MT5Client:
    """Thin wrapper around the MetaTrader5 Python library.

    Manages connection lifecycle (connect / disconnect) and exposes
    helper properties for account info. Credentials are read from the
    .env file at project root.

    Example
    -------
    >>> client = MT5Client()
    >>> client.connect()
    >>> info = client.account_info()
    >>> client.disconnect()
    """

    def __init__(self, env_path: Optional[Path] = None) -> None:
        """Load credentials from .env and initialise connection state.

        Parameters
        ----------
        env_path:
            Explicit path to the .env file. Defaults to project root .env.
        """
        load_dotenv(dotenv_path=env_path)
        self._login: int = int(os.environ["MT5_LOGIN"])
        self._password: str = os.environ["MT5_PASSWORD"]
        self._server: str = os.environ["MT5_SERVER"]
        self._terminal_path: Optional[str] = os.getenv("MT5_PATH")
        self._connected: bool = False

    # ── public interface ───────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise and authenticate with the MT5 terminal.

        Returns
        -------
        bool
            True on success.

        Raises
        ------
        MT5ConnectionError
            If initialisation or login fails.
        """
        init_kwargs: dict = {}
        if self._terminal_path:
            init_kwargs["path"] = self._terminal_path

        if not mt5.initialize(**init_kwargs):
            code, msg = mt5.last_error()
            raise MT5ConnectionError(f"mt5.initialize() failed [{code}]: {msg}")

        if not mt5.login(
            login=self._login,
            password=self._password,
            server=self._server,
        ):
            code, msg = mt5.last_error()
            mt5.shutdown()
            raise MT5ConnectionError(
                f"mt5.login() failed for account {self._login} [{code}]: {msg}"
            )

        self._connected = True
        info = mt5.account_info()
        logger.info(
            "Connected to %s — account %s, balance %.2f %s",
            self._server,
            self._login,
            info.balance,
            info.currency,
        )
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 connection gracefully."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed.")

    def account_info(self) -> mt5.AccountInfo:
        """Return the raw MT5 AccountInfo named-tuple.

        Raises
        ------
        MT5ConnectionError
            If not connected.
        """
        self._require_connected()
        info = mt5.account_info()
        if info is None:
            code, msg = mt5.last_error()
            raise MT5ConnectionError(f"account_info() failed [{code}]: {msg}")
        return info

    @property
    def is_connected(self) -> bool:
        """True if the client is currently connected."""
        return self._connected

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ── private helpers ───────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            raise MT5ConnectionError("Not connected. Call connect() first.")
