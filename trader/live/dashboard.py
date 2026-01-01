# -*- coding: utf-8 -*-
"""
Live trading terminal dashboard.

Real-time dashboard displaying trading status, positions, signals, and statistics.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("rich library not available. Dashboard will be disabled. Install with: pip install rich")


logger = logging.getLogger(__name__)


class LiveTradingDashboard:
    """Real-time terminal dashboard for live trading."""
    
    def __init__(self, live_trader):
        """
        Initialize dashboard.
        
        Parameters
        ----------
        live_trader
            LiveTrader instance
        """
        if not RICH_AVAILABLE:
            raise ImportError("rich library is required for dashboard. Install with: pip install rich")
        
        self.live_trader = live_trader
        self.console = Console()
        self.running = False
        self._update_thread = None
        self._last_update = None
        
    def _get_connection_status(self) -> Dict[str, Any]:
        """Get websocket connection status."""
        market_data = self.live_trader.market_data
        status = {
            'connected': market_data.is_running,
            'last_message': market_data.last_message_time,
            'reconnect_attempts': market_data.reconnect_attempts,
            'fallback_polling': market_data._fallback_polling if hasattr(market_data, '_fallback_polling') else False
        }
        
        if status['last_message']:
            time_since = time.time() - status['last_message']
            status['time_since_last'] = f"{time_since:.0f}s"
        else:
            status['time_since_last'] = "N/A"
        
        return status
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        config = self.live_trader.config
        account_balance = self.live_trader.account_balance
        initial_balance = self.live_trader.initial_balance
        
        # Calculate realized PnL from trades
        realized_pnl = sum(trade.get('realized_pnl', 0) for trade in self.live_trader.trades_log)
        
        # Get unrealized PnL from positions
        unrealized_pnl = self.live_trader.position_tracker.get_total_unrealized_pnl()
        
        total_pnl = realized_pnl + unrealized_pnl
        total_pnl_pct = ((account_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
        
        return {
            'balance': account_balance,
            'initial_balance': initial_balance,
            'equity': account_balance + unrealized_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'symbol': config.symbol,
            'timeframe': config.timeframe
        }
    
    def _get_current_price(self) -> Dict[str, Any]:
        """Get current price information."""
        try:
            current_price = self.live_trader.binance_client.get_current_price(
                self.live_trader.config.symbol
            )
            
            # Get 24h ticker for change
            ticker = self.live_trader.binance_client.client.get_ticker(
                symbol=self.live_trader.config.symbol
            )
            
            price_change_24h = float(ticker.get('priceChangePercent', 0))
            
            return {
                'price': current_price,
                'change_24h': price_change_24h,
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'volume_24h': float(ticker.get('volume', 0))
            }
        except Exception as e:
            logger.debug(f"Error getting current price: {e}")
            return {
                'price': 0.0,
                'change_24h': 0.0,
                'high_24h': 0.0,
                'low_24h': 0.0,
                'volume_24h': 0.0
            }
    
    def _get_positions_table(self) -> Table:
        """Create positions table."""
        table = Table(title="Open Positions", box=box.ROUNDED)
        table.add_column("Trade ID", style="cyan")
        table.add_column("Direction", style="magenta")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("PnL %", justify="right")
        
        positions = self.live_trader.position_tracker.get_all_positions()
        
        if not positions:
            table.add_row("No open positions", "", "", "", "", "", "")
        else:
            for pos in positions:
                trade_id = pos['trade_id'][:12] + "..." if len(pos['trade_id']) > 12 else pos['trade_id']
                direction = pos['direction'].upper()
                entry_price = f"${pos['entry_price']:.2f}"
                current_price = f"${pos['current_price']:.2f}"
                size = f"{pos['position_size']:.6f}"
                
                pnl = pos['unrealized_pnl']
                pnl_pct = pos['unrealized_pnl_pct']
                
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]"
                pnl_pct_str = f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"
                
                table.add_row(
                    trade_id, direction, entry_price, current_price,
                    size, pnl_str, pnl_pct_str
                )
        
        return table
    
    def _get_signals_info(self) -> Dict[str, Any]:
        """Get current signal information."""
        trade_state = self.live_trader.trade_state
        waiting_state = self.live_trader.active_waiting_state
        
        signal_info = {
            'state': trade_state.get('current_state', 'IDLE'),
            'pending_setup': trade_state.get('pending_setup') is not None,
            'active_trade': trade_state.get('active_trade') is not None,
            'waiting_state': waiting_state is not None
        }
        
        if waiting_state:
            signal_info['waiting_type'] = waiting_state.get('type', 'Unknown')
            signal_info['waiting_candles'] = waiting_state.get('candles_waited', 0)
        
        return signal_info
    
    def _get_recent_trades_table(self) -> Table:
        """Create recent trades table."""
        table = Table(title="Recent Trades", box=box.ROUNDED)
        table.add_column("Trade ID", style="cyan")
        table.add_column("Direction", style="magenta")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("Reason", style="yellow")
        table.add_column("PnL", justify="right")
        table.add_column("PnL %", justify="right")
        
        trades = self.live_trader.trades_log[-10:]  # Last 10 trades
        
        if not trades:
            table.add_row("No trades yet", "", "", "", "", "", "")
        else:
            for trade in reversed(trades):  # Show most recent first
                trade_id = trade['trade_id'][:12] + "..." if len(trade['trade_id']) > 12 else trade['trade_id']
                direction = trade['direction'].upper()
                entry_price = f"${trade['entry_price']:.2f}"
                exit_price = f"${trade['exit_price']:.2f}"
                reason = trade.get('exit_reason', 'N/A')
                
                pnl = trade['realized_pnl']
                pnl_pct = trade['realized_pnl_pct']
                
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]"
                pnl_pct_str = f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"
                
                table.add_row(
                    trade_id, direction, entry_price, exit_price,
                    reason, pnl_str, pnl_pct_str
                )
        
        return table
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Calculate trading statistics."""
        trades = self.live_trader.trades_log
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        winning_trades = [t for t in trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in trades if t['realized_pnl'] < 0]
        
        total_pnl = sum(t['realized_pnl'] for t in trades)
        avg_win = sum(t['realized_pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t['realized_pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades) * 100) if trades else 0.0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        # Split into header and body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        
        # Split body into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Split left into top and bottom
        layout["left"].split_column(
            Layout(name="left_top", ratio=1),
            Layout(name="left_bottom", ratio=1)
        )
        
        # Split right into top and bottom
        layout["right"].split_column(
            Layout(name="right_top", ratio=1),
            Layout(name="right_bottom", ratio=1)
        )
        
        return layout
    
    def _render_dashboard(self) -> Layout:
        """Render the dashboard."""
        layout = self._create_layout()
        
        # Header
        config = self.live_trader.config
        header_text = Text()
        header_text.append("LIVE TRADING BOT", style="bold white on blue")
        header_text.append(f" | {config.symbol} {config.timeframe}", style="cyan")
        header_text.append(f" | {'TESTNET' if config.use_testnet else 'REALNET'}", 
                          style="yellow" if config.use_testnet else "red")
        header_text.append(f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # Connection Status
        conn_status = self._get_connection_status()
        conn_text = Text()
        conn_text.append("WebSocket: ", style="bold")
        if conn_status['connected']:
            conn_text.append("CONNECTED", style="green")
        else:
            conn_text.append("DISCONNECTED", style="red")
        
        if conn_status['fallback_polling']:
            conn_text.append(" | Fallback: ACTIVE", style="yellow")
        
        conn_text.append(f" | Last: {conn_status['time_since_last']}", style="dim")
        conn_text.append(f" | Reconnects: {conn_status['reconnect_attempts']}", style="dim")
        
        layout["left_top"].update(Panel(conn_text, title="Connection Status", box=box.ROUNDED))
        
        # Account Info
        account = self._get_account_info()
        account_table = Table.grid(padding=1)
        account_table.add_column(style="cyan")
        account_table.add_column(style="white", justify="right")
        
        account_table.add_row("Balance:", f"${account['balance']:.2f}")
        account_table.add_row("Equity:", f"${account['equity']:.2f}")
        account_table.add_row("Realized PnL:", 
                             f"[green]{account['realized_pnl']:+.2f}[/green]" if account['realized_pnl'] >= 0 
                             else f"[red]{account['realized_pnl']:+.2f}[/red]")
        account_table.add_row("Unrealized PnL:",
                             f"[green]{account['unrealized_pnl']:+.2f}[/green]" if account['unrealized_pnl'] >= 0
                             else f"[red]{account['unrealized_pnl']:+.2f}[/red]")
        account_table.add_row("Total PnL:",
                             f"[green]{account['total_pnl']:+.2f} ({account['total_pnl_pct']:+.2f}%)[/green]" 
                             if account['total_pnl'] >= 0
                             else f"[red]{account['total_pnl']:+.2f} ({account['total_pnl_pct']:+.2f}%)[/red]")
        
        # Positions Table
        positions_table = self._get_positions_table()
        layout["left_top"].update(Panel(positions_table, box=box.ROUNDED))
        
        # Recent Trades Table
        trades_table = self._get_recent_trades_table()
        layout["left_bottom"].update(Panel(trades_table, box=box.ROUNDED))
        
        # Combine Account Info and Statistics
        account = self._get_account_info()
        stats = self._get_statistics()
        
        right_top_table = Table.grid(padding=1)
        right_top_table.add_column(style="cyan", width=20)
        right_top_table.add_column(style="white", justify="right", width=15)
        
        right_top_table.add_row("[bold]Account[/bold]", "")
        right_top_table.add_row("Balance:", f"${account['balance']:.2f}")
        right_top_table.add_row("Equity:", f"${account['equity']:.2f}")
        right_top_table.add_row("Realized PnL:", 
                               f"[green]{account['realized_pnl']:+.2f}[/green]" if account['realized_pnl'] >= 0 
                               else f"[red]{account['realized_pnl']:+.2f}[/red]")
        right_top_table.add_row("Unrealized PnL:",
                               f"[green]{account['unrealized_pnl']:+.2f}[/green]" if account['unrealized_pnl'] >= 0
                               else f"[red]{account['unrealized_pnl']:+.2f}[/red]")
        right_top_table.add_row("Total PnL:",
                               f"[green]{account['total_pnl']:+.2f} ({account['total_pnl_pct']:+.2f}%)[/green]" 
                               if account['total_pnl'] >= 0
                               else f"[red]{account['total_pnl']:+.2f} ({account['total_pnl_pct']:+.2f}%)[/red]")
        right_top_table.add_row("", "")
        right_top_table.add_row("[bold]Statistics[/bold]", "")
        right_top_table.add_row("Total Trades:", str(stats['total_trades']))
        right_top_table.add_row("Wins:", f"[green]{stats['winning_trades']}[/green]")
        right_top_table.add_row("Losses:", f"[red]{stats['losing_trades']}[/red]")
        right_top_table.add_row("Win Rate:", f"{stats['win_rate']:.1f}%")
        right_top_table.add_row("Avg Win:", f"[green]${stats['avg_win']:.2f}[/green]")
        right_top_table.add_row("Avg Loss:", f"[red]${stats['avg_loss']:.2f}[/red]")
        
        layout["right_top"].update(Panel(right_top_table, title="Account & Statistics", box=box.ROUNDED))
        
        # Combine Current Price and Signals
        price_info = self._get_current_price()
        signals = self._get_signals_info()
        
        right_bottom_text = Text()
        right_bottom_text.append("[bold]Price:[/bold] ", style="bold")
        right_bottom_text.append(f"${price_info['price']:.2f}", style="bold white")
        price_change = price_info['change_24h']
        if price_change >= 0:
            right_bottom_text.append(f" (+{price_change:.2f}%)", style="green")
        else:
            right_bottom_text.append(f" ({price_change:.2f}%)", style="red")
        right_bottom_text.append(f"\n24h High: ${price_info['high_24h']:.2f}", style="dim")
        right_bottom_text.append(f" | Low: ${price_info['low_24h']:.2f}", style="dim")
        
        right_bottom_text.append("\n\n[bold]Signals:[/bold] ", style="bold")
        right_bottom_text.append(signals['state'], style="cyan")
        
        if signals['waiting_state']:
            right_bottom_text.append(f"\nWaiting: {signals.get('waiting_type', 'Unknown')}", style="yellow")
            right_bottom_text.append(f" ({signals.get('waiting_candles', 0)} candles)", style="dim")
        
        if signals['active_trade']:
            right_bottom_text.append("\nActive Trade: YES", style="green")
        
        layout["right_bottom"].update(Panel(right_bottom_text, title="Price & Signals", box=box.ROUNDED))
        
        return layout
    
    def update(self):
        """Update dashboard (called from main thread)."""
        self._last_update = time.time()
    
    def start(self):
        """Start dashboard update loop."""
        if self.running:
            return
        
        self.running = True
        
        def update_loop():
            """Dashboard update loop."""
            try:
                with Live(self._render_dashboard(), refresh_per_second=10, screen=True) as live:
                    while self.running:
                        live.update(self._render_dashboard())
                        time.sleep(self.live_trader.config.dashboard_refresh_rate)
            except Exception as e:
                logger.error(f"Dashboard error: {e}", exc_info=True)
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Dashboard started")
    
    def stop(self):
        """Stop dashboard."""
        self.running = False
        if self._update_thread:
            self._update_thread.join(timeout=2)
        logger.info("Dashboard stopped")

