# risk_manager.py (VERSÃO COM RL)
import pandas as pd
from datetime import datetime, timedelta
import logging
from profit_tracker import ProfitTracker
from rl_trader import RLTrader


class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()
        
        # Métricas de trading
        self.daily_trades = 0
        self.daily_pnl = 0
        self.total_trades = 0
        self.win_loss_ratio = 0
        self.last_reset = datetime.now()
        self.open_positions = []
        self.last_trade_time = {}
        self.cooldown_period = config.get('cooldown_period', 30)

        # Valores para controle de posição
        self.max_position_value = 50
        self.min_position_value = 20

        # Rastreador de lucros
        self.profit_tracker = ProfitTracker(config)
        
        # RL Trader - NOVO
        self.rl_trader = RLTrader(enable_training=True)
        self.portfolio_state = {}
        self.rl_enabled = config.get('rl_enabled', True)

    def _setup_logger(self):
        """Configura sistema de logging"""
        logger = logging.getLogger('RiskManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def execute_trade(self, market, signal, trader):
        """Executa trade com gerenciamento de risco E RL"""
        try:
            symbol = market['symbol']

            # Verifica se pode trade
            if not await self.can_trade():
                return None

            # Verifica cooldown
            if symbol in self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
                if time_since_last < self.cooldown_period:
                    self.logger.info(f"⏳ Cooldown ativo para {symbol}: {self.cooldown_period - time_since_last:.0f}s restantes")
                    return None

            # ATUALIZAÇÃO: Integração com RL
            final_signal = await self._get_enhanced_signal(market, signal, trader)
            
            if final_signal['action'] == 'hold':
                self.logger.info(f"⏭️ RL recomendou HOLD para {symbol}")
                return None

            # Calcula posição com validações
            position_size = await self._calculate_proper_position(market, final_signal, trader)
            if position_size <= 0:
                return None

            # Executa trade
            trade = await trader.place_market_order(
                symbol=symbol,
                side=final_signal['action'],
                amount=position_size
            )

            if trade:
                # Atualiza métricas
                self.daily_trades += 1
                self.total_trades += 1
                self.last_trade_time[symbol] = datetime.now()

                # Registra no profit tracker
                current_price = await trader.get_current_price(symbol)
                await self.profit_tracker.record_entry(trade, market, final_signal, current_price)

                # ATUALIZAÇÃO: Salva estado RL para aprendizado futuro
                if hasattr(final_signal, 'rl_state'):
                    self.last_rl_state = final_signal['rl_state']
                    self.last_rl_action = 0 if final_signal['action'] == 'buy' else 1 if final_signal['action'] == 'sell' else 2

                self.logger.info(f"✅ TRADE EXECUTADO: {symbol} {final_signal['action']} {position_size:.6f} unidades")
                self._log_trade_metrics()

            return trade

        except Exception as e:
            self.logger.error(f"❌ Erro executando trade: {e}")
            return None

    async def _get_enhanced_signal(self, market, original_signal, trader):
        """Combina sinal tradicional com RL"""
        try:
            if not self.rl_enabled:
                return original_signal

            # Atualiza estado do portfólio para RL
            await self._update_portfolio_state(trader)
            
            # Obtém sinal do RL
            rl_signal = await self.rl_trader.get_trading_signal(
                market_data=market,
                portfolio=self.portfolio_state,
                symbol=market['symbol']
            )
            
            # Combina sinais (RL tem prioridade se for forte)
            rl_strength = rl_signal.get('strength', 0)
            original_strength = original_signal.get('strength', 0)
            
            if rl_strength > 0.7 and rl_signal['action'] != 'hold':
                final_signal = rl_signal
                final_signal['combined_type'] = f"RL_{original_signal.get('type', 'TRADITIONAL')}"
                self.logger.info(f"🤖 DECISÃO RL: {market['symbol']} {rl_signal['action'].upper()} "
                               f"(Força RL: {rl_strength:.2f} vs Trad: {original_strength:.2f})")
            else:
                final_signal = original_signal
                final_signal['combined_type'] = f"TRADITIONAL_{original_signal.get('type', 'BASIC')}"
                self.logger.info(f"📊 DECISÃO TRADICIONAL: {market['symbol']} {original_signal['action'].upper()}")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"❌ Erro combinando sinais: {e}")
            return original_signal

    async def _update_portfolio_state(self, trader):
        """Atualiza estado do portfólio para o RL"""
        try:
            balance = await trader.get_account_balance()
            total_value = max(sum(balance.values()), 1)  # Evita divisão por zero
            
            self.portfolio_state = {
                'cash_ratio': balance.get('USDT', 0) / total_value,
                'position_size': len(self.open_positions),
                'current_pnl': self.daily_pnl,
                'total_trades': self.total_trades
            }
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando estado do portfólio: {e}")
            self.portfolio_state = {
                'cash_ratio': 0.5,
                'position_size': 0,
                'current_pnl': 0,
                'total_trades': 0
            }

    async def record_rl_training(self, position_id: str, exit_price: float, 
                               exit_reason: str, market_conditions: Dict):
        """Registra resultado do trade para treinamento do RL"""
        try:
            if not self.rl_enabled or not hasattr(self, 'last_rl_state'):
                return

            # Encontra a posição fechada
            closed_position = None
            for trade in self.profit_tracker.trade_history:
                if trade.get('id') == position_id:
                    closed_position = trade
                    break

            if not closed_position:
                return

            # Calcula métricas para recompensa
            pnl = closed_position.get('pnl', 0)
            holding_period = closed_position.get('holding_period', 1.0)
            action = self.last_rl_action

            # Cria próximo estado (estado atual)
            next_state = self.last_rl_state  # Simplificado por enquanto

            # Atualiza o modelo RL
            self.rl_trader.update_with_result(
                state=self.last_rl_state,
                action=action,
                pnl=pnl,
                next_state=next_state,
                market_conditions=market_conditions,
                holding_period=holding_period,
                done=True
            )

            self.logger.info(f"🧠 RL Training: {position_id} PnL=${pnl:.2f}, "
                           f"Action={action}, Reward calculada")

        except Exception as e:
            self.logger.error(f"❌ Erro registrando treinamento RL: {e}")

    # ... (mantenha o resto dos métodos existentes: _calculate_proper_position, etc.)

    async def get_risk_report(self):
        """Gera relatório completo de risco incluindo RL"""
        report = {
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.config['max_daily_trades'],
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.config['daily_loss_limit'],
            'total_trades': self.total_trades,
            'open_positions': len(self.open_positions),
            'cooldown_active': len(self.last_trade_time),
            'rl_enabled': self.rl_enabled,
            'rl_metrics': self.rl_trader.get_training_metrics() if self.rl_enabled else {}
        }
        
        self.logger.info("📋 Relatório de Risco (com RL):")
        for key, value in report.items():
            if key != 'rl_metrics':
                self.logger.info(f"   {key}: {value}")
        
        if self.rl_enabled:
            rl_metrics = report['rl_metrics']
            self.logger.info("   🤖 RL Metrics:")
            self.logger.info(f"      • Exploration (ε): {rl_metrics['epsilon']:.3f}")
            self.logger.info(f"      • Memory Size: {rl_metrics['memory_size']}")
            self.logger.info(f"      • Model Updated: {rl_metrics['model_updated']}")
            
        return report
