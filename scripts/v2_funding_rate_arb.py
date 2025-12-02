import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, cast

import pandas as pd
from pydantic import Field, field_validator

from hummingbot.client.config.config_data_types import ClientConfigEnum
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.event.events import FundingPaymentCompletedEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class FundingRateArbitrageConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}
    leverage: int = Field(
        default=20, gt=0,
        json_schema_extra={"prompt": lambda mi: "Enter the leverage (e.g. 20): ", "prompt_on_new": True},
    )
    min_funding_rate_profitability: Decimal = Field(
        default=0.001,
        json_schema_extra={
            "prompt": lambda mi: "Enter the min funding rate profitability to enter in a position (e.g. 0.001): ",
            "prompt_on_new": True}
    )
    connectors: Set[str] = Field(
        default="hyperliquid_perpetual,binance_perpetual",
        json_schema_extra={
            "prompt": lambda mi: "Enter the connectors separated by commas (e.g. hyperliquid_perpetual,binance_perpetual): ",
            "prompt_on_new": True}
    )
    token_directions: Dict[str, Dict[str, TradeType]] = Field(
        default_factory=lambda: {
            "WIF": {
                "hyperliquid_perpetual": TradeType.BUY,
                "binance_perpetual": TradeType.SELL,
            },
            "FET": {
                "hyperliquid_perpetual": TradeType.BUY,
                "binance_perpetual": TradeType.SELL,
            },
        },
        json_schema_extra={
            "prompt": lambda mi: (
                "Enter token directions as token->connector->direction mappings (e.g."
                " WIF.hyperliquid_perpetual=buy,WIF.binance_perpetual=sell)"
            ),
            "prompt_on_new": True,
        },
    )
    position_size_quote: Decimal = Field(
        default=100,
        json_schema_extra={
            "prompt": lambda mi: "Enter the position size in quote asset (e.g. order amount 100 will open 100 long on hyperliquid and 100 short on binance): ",
            "prompt_on_new": True
        }
    )
    profitability_to_take_profit: Decimal = Field(
        default=0.01,
        json_schema_extra={
            "prompt": lambda mi: "Enter the profitability to take profit (including PNL of positions and fundings received): ",
            "prompt_on_new": True}
    )
    funding_rate_diff_stop_loss: Decimal = Field(
        default=-0.001,
        json_schema_extra={
            "prompt": lambda mi: "Enter the funding rate difference to stop the position (e.g. -0.001): ",
            "prompt_on_new": True}
    )
    trade_profitability_condition_to_enter: bool = Field(
        default=False,
        json_schema_extra={
            "prompt": lambda mi: "Do you want to check the trade profitability condition to enter? (True/False): ",
            "prompt_on_new": True}
    )

    @field_validator("connectors", mode="before")
    @classmethod
    def validate_connectors(cls, v):
        if isinstance(v, str):
            return {item.strip() for item in v.split(",") if item.strip()}
        return v

    @field_validator("token_directions", mode="before")
    @classmethod
    def validate_token_directions(cls, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            normalized: Dict[str, Dict[str, TradeType]] = {}
            for token, connector_map in value.items():
                if not isinstance(connector_map, dict):
                    raise ValueError(
                        "token_directions entries must map each token to a connector:direction dictionary"
                    )
                normalized[token] = {}
                for connector, direction in connector_map.items():
                    normalized[token][connector] = cls._normalize_direction(direction)
            return normalized
        raise ValueError("token_directions must be a mapping of token to connector directions")

    @staticmethod
    def _normalize_direction(direction) -> TradeType:
        if isinstance(direction, TradeType):
            return direction
        if isinstance(direction, str):
            direction_value = direction.strip().lower()
            if direction_value == "buy":
                return TradeType.BUY
            if direction_value == "sell":
                return TradeType.SELL
        raise ValueError(f"Unsupported direction value: {direction}")

    @property
    def token_symbols(self) -> Set[str]:
        if not self.token_directions:
            raise ValueError("token_directions must define at least one token with connector directions.")
        return set(self.token_directions.keys())

    def connectors_for_token(self, token: str) -> Set[str]:
        direction_map = self.token_directions.get(token)
        if direction_map:
            return set(direction_map.keys())
        raise ValueError(f"No connector directions configured for token {token}.")

    def connector_direction(self, token: str, connector: str) -> Optional[TradeType]:
        return self.token_directions.get(token, {}).get(connector)


@dataclass
class FundingArbitrageState:
    connector_1: str
    connector_2: str
    executors_ids: List[str]
    side: TradeType
    funding_payments: List[FundingPaymentCompletedEvent] = field(default_factory=list)
    stop_condition_start: Optional[float] = None


class FundingRateArbitrage(StrategyV2Base):
    quote_markets_map = {
        "hyperliquid_perpetual": "USD",
        "binance_perpetual": "USDT"
    }
    seconds_per_day = 60 * 60 * 24

    @classmethod
    def get_trading_pair_for_connector(cls, token, connector):
        return f"{token}-{cls.quote_markets_map.get(connector, 'USDT')}"

    @classmethod
    def init_markets(cls, config: FundingRateArbitrageConfig):
        markets = {}
        for connector in config.connectors:
            trading_pairs = {cls.get_trading_pair_for_connector(token, connector) for token in config.token_symbols}
            markets[connector] = trading_pairs
        cls.markets = markets

    def __init__(self, connectors: Dict[str, ConnectorBase], config: FundingRateArbitrageConfig):
        super().__init__(connectors, config)
        self.config = config
        self.active_funding_arbitrages: Dict[str, FundingArbitrageState] = {}
        self._initial_funding_rates: Dict[Tuple[str, str], Decimal] = {}
        self._funding_rate_updates: Dict[Tuple[str, str], bool] = {}
        self._stop_condition_duration = 60 * 60

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                position_mode = PositionMode.ONEWAY if connector_name == "hyperliquid_perpetual" else PositionMode.HEDGE
                connector.set_position_mode(position_mode)
                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_funding_info_by_token(self, token):
        """
        This method provides the funding rates across all the connectors
        """
        connector_rules = self.config.connectors_for_token(token)
        if not connector_rules:
            raise ValueError(f"No connector directions configured for token {token}.")
        funding_rates = {}
        for connector_name in connector_rules:
            connector = self.connectors.get(connector_name)
            if connector is None:
                raise ValueError(
                    f"Connector {connector_name} configured for token {token} is not initialized in this strategy."
                )
            trading_pair = self.get_trading_pair_for_connector(token, connector_name)
            funding_info = connector.get_funding_info(trading_pair)
            if funding_info is None:
                raise ValueError(
                    f"Funding info missing for token {token} on {connector_name}."
                )
            interval_raw = funding_info.funding_interval_hours
            if interval_raw is None:
                raise ValueError(
                    f"Funding interval missing for token {token} on {connector_name}."
                )
            try:
                interval_hours = int(interval_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Funding interval not castable to int for token {token} on {connector_name}: {interval_raw}"
                ) from exc
            if interval_hours <= 0:
                raise ValueError(
                    f"Funding interval must be positive for token {token} on {connector_name}: {interval_hours}"
                )
            funding_info.funding_interval_hours = interval_hours
            self._track_funding_rate_update(token, connector_name, funding_info.rate)
            funding_rates[connector_name] = funding_info
        return funding_rates

    def _track_funding_rate_update(self, token: str, connector_name: str, current_rate: Decimal):
        key = (token, connector_name)
        tracked_rate = self._initial_funding_rates.get(key)
        if tracked_rate is None:
            self._initial_funding_rates[key] = current_rate
            self._funding_rate_updates.setdefault(key, False)
        elif not self._funding_rate_updates.get(key, False) and current_rate != tracked_rate:
            self._funding_rate_updates[key] = True

    def _desired_side_for_connector(self, token: str, connector_name: str) -> Optional[TradeType]:
        return self.config.connector_direction(token, connector_name)

    def _is_direction_allowed(self, token: str, connector_name: str, desired_side: TradeType) -> bool:
        required_side = self._desired_side_for_connector(token, connector_name)
        if required_side is None:
            return True
        return required_side == desired_side

    def _token_direction_lines(self) -> List[str]:
        lines = ["Configured Token Directions:"]
        for token in sorted(self.config.token_directions.keys()):
            direction_map = self.config.token_directions[token]
            connector_entries = []
            for connector_name, direction in sorted(direction_map.items()):
                if isinstance(direction, TradeType):
                    direction_label = direction.name.lower()
                else:
                    direction_label = str(direction).lower()
                connector_entries.append(f"{connector_name} -> {direction_label}")
            connectors_text = ", ".join(connector_entries) if connector_entries else "None"
            lines.append(f"{token}: {connectors_text}")
        return lines

    def get_current_profitability_after_fees(self, token: str, connector_1: str, connector_2: str, side: TradeType):
        """
        This methods compares the profitability of buying at market in the two exchanges. If the side is TradeType.BUY
        means that the operation is long on connector 1 and short on connector 2.
        """
        trading_pair_1 = self.get_trading_pair_for_connector(token, connector_1)
        trading_pair_2 = self.get_trading_pair_for_connector(token, connector_2)

        connector_1_price = Decimal(self.market_data_provider.get_price_for_quote_volume(
            connector_name=connector_1,
            trading_pair=trading_pair_1,
            quote_volume=float(self.config.position_size_quote),
            is_buy=side == TradeType.BUY,
        ).result_price)
        connector_2_price = Decimal(self.market_data_provider.get_price_for_quote_volume(
            connector_name=connector_2,
            trading_pair=trading_pair_2,
            quote_volume=float(self.config.position_size_quote),
            is_buy=side != TradeType.BUY,
        ).result_price)
        estimated_fees_connector_1 = self.connectors[connector_1].get_fee(
            base_currency=trading_pair_1.split("-")[0],
            quote_currency=trading_pair_1.split("-")[1],
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=self.config.position_size_quote / connector_1_price,
            price=connector_1_price,
            is_maker=False,
            position_action=PositionAction.OPEN
        ).percent
        estimated_fees_connector_2 = self.connectors[connector_2].get_fee(
            base_currency=trading_pair_2.split("-")[0],
            quote_currency=trading_pair_2.split("-")[1],
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=self.config.position_size_quote / connector_2_price,
            price=connector_2_price,
            is_maker=False,
            position_action=PositionAction.OPEN
        ).percent

        if side == TradeType.BUY:
            estimated_trade_pnl_pct = (connector_2_price - connector_1_price) / connector_1_price
        else:
            estimated_trade_pnl_pct = (connector_1_price - connector_2_price) / connector_2_price
        return estimated_trade_pnl_pct - estimated_fees_connector_1 - estimated_fees_connector_2

    def _find_best_combination(self, token: str, funding_info_report: Dict, enforce_direction: bool):
        best_combination = None
        highest_profitability = Decimal("0")
        for connector_1 in funding_info_report:
            for connector_2 in funding_info_report:
                if connector_1 != connector_2:
                    rate_connector_1 = self.get_normalized_funding_rate_in_seconds(token, funding_info_report, connector_1)
                    rate_connector_2 = self.get_normalized_funding_rate_in_seconds(token, funding_info_report, connector_2)
                    funding_rate_diff = abs(rate_connector_1 - rate_connector_2) * self.seconds_per_day
                    if funding_rate_diff > highest_profitability:
                        trade_side = TradeType.BUY if rate_connector_1 < rate_connector_2 else TradeType.SELL
                        opposite_side = TradeType.SELL if trade_side == TradeType.BUY else TradeType.BUY
                        if enforce_direction:
                            if not self._is_direction_allowed(token, connector_1, trade_side):
                                continue
                            if not self._is_direction_allowed(token, connector_2, opposite_side):
                                continue
                        highest_profitability = funding_rate_diff
                        best_combination = (connector_1, connector_2, trade_side, funding_rate_diff)
        return best_combination

    def get_most_profitable_combination(self, token: str, funding_info_report: Dict):
        """Return the best funding combination without enforcing direction rules."""
        return self._find_best_combination(token, funding_info_report, enforce_direction=False)

    def get_directionally_allowed_combination(self, token: str, funding_info_report: Dict):
        """Return the best funding combination that complies with configured directions."""
        return self._find_best_combination(token, funding_info_report, enforce_direction=True)

    def get_normalized_funding_rate_in_seconds(self, token: str, funding_info_report, connector_name):
        funding_info = funding_info_report[connector_name]
        interval_hours = funding_info.funding_interval_hours
        interval_seconds = interval_hours * 60 * 60
        return funding_info.rate / interval_seconds

    @staticmethod
    def _format_time_to_funding(seconds: float) -> str:
        """Format seconds to a human readable HH:MM:SS string, defaulting to MM:SS when below an hour."""
        if seconds is None:
            return "N/A"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @classmethod
    def _format_funding_payments_for_connector(
        cls, payments: List[FundingPaymentCompletedEvent], connector_name: str
    ) -> str:
        connector_payments = [payment for payment in payments if payment.market == connector_name]
        total_amount = sum((payment.amount for payment in connector_payments), Decimal("0"))
        return f"{total_amount:.5f} USDT"

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        """
        In this method we are going to evaluate if a new set of positions has to be created for each of the tokens that
        don't have an active arbitrage.
        More filters can be applied to limit the creation of the positions, since the current logic is only checking for
        positive pnl between funding rate. Is logged and computed the trading profitability at the time for entering
        at market to open the possibilities for other people to create variations like sending limit position executors
        and if one gets filled buy market the other one to improve the entry prices.
        """
        create_actions = []
        for token in self.config.token_symbols:
            if token not in self.active_funding_arbitrages:
                funding_info_report = self.get_funding_info_by_token(token)
                best_combination = self.get_directionally_allowed_combination(token, funding_info_report)
                if best_combination is None:
                    continue

                connector_1, connector_2, trade_side, expected_profitability = best_combination
                if expected_profitability >= self.config.min_funding_rate_profitability:
                    current_profitability = self.get_current_profitability_after_fees(
                        token, connector_1, connector_2, trade_side
                    )
                    if self.config.trade_profitability_condition_to_enter:
                        if current_profitability < 0:
                            self.logger().debug(
                                f"Best Combination: {connector_1} | {connector_2} | {trade_side} "
                                f"Funding rate profitability: {expected_profitability:.5f} "
                                f"Trading profitability after fees: {current_profitability:.5f} "
                                f"Trade profitability is negative, skipping..."
                            )
                            continue
                    self.logger().info(
                        f"Found best combination for {token}: {connector_1} | {connector_2} | {trade_side} "
                        f"Funding rate profitability: {expected_profitability:.5f} "
                        f"Trading profitability after fees: {current_profitability:.5f}"
                    )
                    position_executor_config_1, position_executor_config_2 = self.get_position_executors_config(
                        token, connector_1, connector_2, trade_side
                    )
                    self.active_funding_arbitrages[token] = FundingArbitrageState(
                        connector_1=connector_1,
                        connector_2=connector_2,
                        executors_ids=[position_executor_config_1.id, position_executor_config_2.id],
                        side=trade_side,
                    )
                    return [
                        CreateExecutorAction(executor_config=position_executor_config_1),
                        CreateExecutorAction(executor_config=position_executor_config_2),
                    ]
        return create_actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        Once the funding rate arbitrage is created we are going to control the funding payments pnl and the current
        pnl of each of the executors at the cost of closing the open position at market.
        If that PNL is greater than the profitability_to_take_profit
        """
        stop_executor_actions = []
        tokens_to_remove = []
        for token, funding_arbitrage_info in self.active_funding_arbitrages.items():
            executors = self.filter_executors(
                executors=self.get_all_executors(),
                filter_func=lambda x: x.id in funding_arbitrage_info.executors_ids
            )
            funding_payments_pnl = sum(funding_payment.amount for funding_payment in funding_arbitrage_info.funding_payments)
            executors_pnl = sum(executor.net_pnl_quote for executor in executors)
            funding_info_report = self.get_funding_info_by_token(token)
            if funding_arbitrage_info.side == TradeType.BUY:
                funding_rate_diff = self.get_normalized_funding_rate_in_seconds(
                    token, funding_info_report, funding_arbitrage_info.connector_2
                ) - self.get_normalized_funding_rate_in_seconds(token, funding_info_report, funding_arbitrage_info.connector_1)
            else:
                funding_rate_diff = self.get_normalized_funding_rate_in_seconds(
                    token, funding_info_report, funding_arbitrage_info.connector_1
                ) - self.get_normalized_funding_rate_in_seconds(token, funding_info_report, funding_arbitrage_info.connector_2)
            below_threshold = funding_rate_diff * self.seconds_per_day < self.config.funding_rate_diff_stop_loss
            if below_threshold:
                if funding_arbitrage_info.stop_condition_start is None:
                    funding_arbitrage_info.stop_condition_start = self.current_timestamp
                condition_start = funding_arbitrage_info.stop_condition_start
                condition_duration = self.current_timestamp - condition_start
                if (condition_duration == 0):
                    self.logger().info(
                        f"Funding rate stop loss condition for {token} started: duration={condition_duration:.2f}s "
                        f"funding_rate_diff={funding_rate_diff:.5f}"
                    )
            else:
                condition_duration = 0
                funding_arbitrage_info.stop_condition_start = None

            if below_threshold and condition_duration >= self._stop_condition_duration:
                self.logger().info(
                    f"Funding rate stop loss met for {token}: executors={funding_arbitrage_info.executors_ids} "
                    f"funding_rate_diff={funding_rate_diff:.5f} for {condition_duration}s"
                    f"net_pnl={executors_pnl:.5f} funding_pnl={funding_payments_pnl:.5f} "
                )
                stop_executor_actions.extend([StopExecutorAction(executor_id=executor.id) for executor in executors])
                tokens_to_remove.append(token)
        for token in tokens_to_remove:
            self.active_funding_arbitrages.pop(token, None)
        return stop_executor_actions

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        """
        Based on the funding payment event received, check if one of the active arbitrages matches to add the event
        to the list.
        """
        try:
            token = funding_payment_completed_event.trading_pair.split("-")[0]

            if token in self.active_funding_arbitrages:
                self.active_funding_arbitrages[token].funding_payments.append(funding_payment_completed_event)
        except Exception as event_error:  # broad catch to prevent strategy crash on malformed events
            self.logger().exception("Failed to handle funding payment event: %s", event_error)

    def get_position_executors_config(self, token, connector_1, connector_2, trade_side):
        price = self.market_data_provider.get_price_by_type(
            connector_name=connector_1,
            trading_pair=self.get_trading_pair_for_connector(token, connector_1),
            price_type=PriceType.MidPrice
        )
        position_amount = self.config.position_size_quote / price

        position_executor_config_1 = PositionExecutorConfig(
            timestamp=self.current_timestamp,
            connector_name=connector_1,
            trading_pair=self.get_trading_pair_for_connector(token, connector_1),
            side=trade_side,
            amount=position_amount,
            leverage=self.config.leverage,
            triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
        )
        position_executor_config_2 = PositionExecutorConfig(
            timestamp=self.current_timestamp,
            connector_name=connector_2,
            trading_pair=self.get_trading_pair_for_connector(token, connector_2),
            side=TradeType.BUY if trade_side == TradeType.SELL else TradeType.SELL,
            amount=position_amount,
            leverage=self.config.leverage,
            triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
        )
        return position_executor_config_1, position_executor_config_2

    def format_status(self) -> str:
        original_status = super().format_status()
        funding_rate_status = []

        if self.ready_to_trade:
            all_funding_info = []
            all_best_paths = []
            for token in self.config.token_symbols:
                best_paths_info = {"token": token}
                funding_info_report = self.get_funding_info_by_token(token)
                if not funding_info_report:
                    continue

                token_info = {"token": token}

                for connector_name, info in funding_info_report.items():
                    interval_hours = info.funding_interval_hours
                    normalized_rate = self.get_normalized_funding_rate_in_seconds(
                        token, funding_info_report, connector_name
                    ) * self.seconds_per_day
                    direction_label = self._desired_side_for_connector(token, connector_name)
                    if direction_label == TradeType.BUY:
                        direction_text = "long"
                    elif direction_label == TradeType.SELL:
                        direction_text = "short"
                    else:
                        direction_text = "n/a"
                    token_info[f"{connector_name} Funding"] = (
                        f"{normalized_rate:.2%} ({info.rate:.4%} each {interval_hours}h, {direction_text})"
                    )

                best_combination = self.get_most_profitable_combination(token, funding_info_report)
                if best_combination is None:
                    continue

                connector_1, connector_2, side, funding_rate_diff = best_combination
                profitability_after_fees = self.get_current_profitability_after_fees(token, connector_1, connector_2, side)
                connector_1_label = connector_1.replace("_perpetual", "")
                connector_2_label = connector_2.replace("_perpetual", "")
                if side == TradeType.BUY:
                    connector_1_arrow, connector_2_arrow = "↑", "↓"
                else:
                    connector_1_arrow, connector_2_arrow = "↓", "↑"
                best_paths_info["Best Path"] = (
                    f"{connector_1_label} {connector_1_arrow} | {connector_2_label} {connector_2_arrow}"
                )
                best_paths_info["Best Rate Diff"] = f"{funding_rate_diff:.2%}"
                best_paths_info["Trade Profitability"] = f"{profitability_after_fees:.2%}"
                if funding_rate_diff != 0:
                    days_trade_prof = -profitability_after_fees / funding_rate_diff
                    best_paths_info["Days Trade Prof"] = f"{days_trade_prof:.2f}"
                    days_to_tp = (self.config.profitability_to_take_profit - profitability_after_fees) / funding_rate_diff
                    best_paths_info["Days to TP"] = "already" if days_to_tp <= 0 else f"{days_to_tp:.2f}"
                else:
                    best_paths_info["Days Trade Prof"] = "N/A"
                    best_paths_info["Days to TP"] = "N/A"

                time_to_next_funding_info_c1 = funding_info_report[connector_1].next_funding_utc_timestamp - self.current_timestamp
                time_to_next_funding_info_c2 = funding_info_report[connector_2].next_funding_utc_timestamp - self.current_timestamp
                best_paths_info["Time to Funding 1"] = self._format_time_to_funding(time_to_next_funding_info_c1)
                best_paths_info["Time to Funding 2"] = self._format_time_to_funding(time_to_next_funding_info_c2)

                all_funding_info.append(token_info)
                all_best_paths.append(best_paths_info)

            funding_rate_status.append(f"\n\n\nMin Funding Rate Profitability: {self.config.min_funding_rate_profitability:.2%}")
            funding_rate_status.append(f"Profitability to Take Profit: {self.config.profitability_to_take_profit:.2%}\n")
            funding_rate_status.extend(self._token_direction_lines())
            funding_rate_status.append("\nFunding Rate Info (Funding Profitability in Days): ")
            table_format = cast(ClientConfigEnum, "psql")
            funding_rate_status.append(format_df_for_printout(df=pd.DataFrame(all_funding_info), table_format=table_format,))

            funding_rate_status.append("\nBest Funding Rate Arbitrage Paths: ")
            funding_rate_status.append(format_df_for_printout(df=pd.DataFrame(all_best_paths), table_format=table_format,))
            stale_combinations = [
                f"{token}@{connector}"
                for (token, connector), was_updated in self._funding_rate_updates.items()
                if not was_updated
            ]
            if stale_combinations:
                funding_rate_status.append(
                    "WARNING: Funding rate never updated after initialization for: " + ", ".join(sorted(stale_combinations))
                )

            if self.active_funding_arbitrages:
                active_rows = []
                for token, funding_arbitrage_info in self.active_funding_arbitrages.items():
                    long_connector = (
                        funding_arbitrage_info.connector_1
                        if funding_arbitrage_info.side == TradeType.BUY
                        else funding_arbitrage_info.connector_2
                    )
                    short_connector = (
                        funding_arbitrage_info.connector_2
                        if funding_arbitrage_info.side == TradeType.BUY
                        else funding_arbitrage_info.connector_1
                    )
                    long_funding = self._format_funding_payments_for_connector(
                        funding_arbitrage_info.funding_payments, long_connector
                    )
                    short_funding = self._format_funding_payments_for_connector(
                        funding_arbitrage_info.funding_payments, short_connector
                    )
                    connector_1_label = funding_arbitrage_info.connector_1.replace("_perpetual", "")
                    connector_2_label = funding_arbitrage_info.connector_2.replace("_perpetual", "")
                    if funding_arbitrage_info.side == TradeType.BUY:
                        connector_1_arrow, connector_2_arrow = "↑", "↓"
                    else:
                        connector_1_arrow, connector_2_arrow = "↓", "↑"
                    path_display = (
                        f"{connector_1_label} {connector_1_arrow} | {connector_2_label} {connector_2_arrow}"
                    )
                    if funding_arbitrage_info.stop_condition_start is not None:
                        elapsed = self.current_timestamp - funding_arbitrage_info.stop_condition_start
                        remaining = max(0, self._stop_condition_duration - elapsed)
                        stop_eta = self._format_time_to_funding(remaining)
                    else:
                        stop_eta = "N/A"
                    active_rows.append(
                        {
                            "token": token,
                            "path": path_display,
                            "long funding": long_funding,
                            "short funding": short_funding,
                            "stop loss ETA": stop_eta,
                        }
                    )
                funding_rate_status.append("\nActive Funding Arbitrages:")
                funding_rate_status.append(
                    format_df_for_printout(
                        df=pd.DataFrame(active_rows),
                        table_format=table_format,
                    )
                )
            else:
                funding_rate_status.append("\nNo active funding arbitrages.")
        return original_status + "\n".join(funding_rate_status)
