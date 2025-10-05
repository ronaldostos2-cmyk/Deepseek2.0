# risk_manager.py (VERSÃO CORRIGIDA)
import pandas as pd
from datetime import datetime, timedelta
import logging
from profit_tracker import ProfitTracker


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
        """Executa trade com gerenciamento de risco"""
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

            # Calcula posição com validações
            position_size = await self._calculate_proper_position(market, signal, trader)
            if position_size <= 0:
                return None

            # Executa trade
            trade = await trader.place_market_order(
                symbol=symbol,
                side=signal['action'],
                amount=position_size
            )

            if trade:
                # Atualiza métricas
                self.daily_trades += 1
                self.total_trades += 1
                self.last_trade_time[symbol] = datetime.now()

                # Registra no profit tracker
                current_price = await trader.get_current_price(symbol)
                await self.profit_tracker.record_entry(trade, market, signal, current_price)

                self.logger.info(f"✅ TRADE EXECUTADO: {symbol} {signal['action']} {position_size:.6f} unidades")
                self._log_trade_metrics()

            return trade

        except Exception as e:
            self.logger.error(f"❌ Erro executando trade: {e}")
            return None

    async def _calculate_proper_position(self, market, signal, trader):
        """Calcula posição que atende aos requisitos NOTIONAL - CORRIGIDO"""
        try:
            symbol = market['symbol']
            balance = await trader.get_account_balance()
            usdt_balance = balance.get('USDT', 1000)
            current_price = market['current_price']

            # VALORES MAIORES para atender NOTIONAL
            risk_amount = usdt_balance * self.config['risk_per_trade']

            # Garante NOTIONAL mínimo de $20
            trade_value = max(risk_amount, self.min_position_value)

            # Limita pelo máximo
            trade_value = min(trade_value, self.max_position_value)

            # Quantidade baseada no valor
            quantity = trade_value / current_price

            # Verifica tamanho mínimo do lote - VALORES CORRIGIDOS
            min_qty = await self._get_minimum_quantity(symbol)
            if quantity < min_qty:
                self.logger.warning(f"📏 Quantidade {quantity:.6f} abaixo do mínimo {min_qty:.6f} para {symbol}")
                # Ajusta para o mínimo
                quantity = min_qty
                # Recalcula o valor do trade
                trade_value = quantity * current_price
                
                # Verifica se ainda atende ao mínimo notional
                if trade_value < self.min_position_value:
                    self.logger.warning(f"💰 Valor ajustado ${trade_value:.2f} ainda abaixo do mínimo ${self.min_position_value}")
                    return 0

            # Verifica notional mínimo
            min_notional = self.min_position_value
            if trade_value < min_notional:
                self.logger.warning(f"💰 Valor do trade ${trade_value:.2f} abaixo do mínimo notional ${min_notional}")
                return 0

            # Verifica saldo para compras
            if signal['action'] == 'buy':
                max_affordable = (usdt_balance * 0.95) / current_price  # 5% de margem de segurança
                quantity = min(quantity, max_affordable)

            self.logger.info(f"💰 Cálculo posição: Balanço=${usdt_balance:.2f}, "
                           f"Valor=${trade_value:.2f}, Qtd={quantity:.6f}, "
                           f"Mín.Qtd={min_qty:.6f}")

            return quantity

        except Exception as e:
            self.logger.error(f"❌ Erro calculando posição: {e}")
            return 0

    async def _get_minimum_quantity(self, symbol):
        """Obtém quantidade mínima permitida para o símbolo - VALORES CORRIGIDOS"""
        # Valores mínimos realistas para Binance
        min_quantities = {
            'BTC/USDT': 0.00001,   # 0.00001 BTC
            'ETH/USDT': 0.0001,    # 0.0001 ETH
            'ADA/USDT': 1.0,       # 1 ADA
            'BNB/USDT': 0.001,     # 0.001 BNB
            'XRP/USDT': 1.0,       # 1 XRP
            'DOT/USDT': 0.1,       # 0.1 DOT
            'LINK/USDT': 0.01,     # 0.01 LINK
            'LTC/USDT': 0.001,     # 0.001 LTC
            'BCH/USDT': 0.001,     # 0.001 BCH
            'EOS/USDT': 0.1,       # 0.1 EOS
        }
        
        # Tenta diferentes formatos do símbolo
        for sym_format in [symbol, symbol.replace('/', ''), symbol.replace('/', '') + 'T']:
            if sym_format in min_quantities:
                return min_quantities[sym_format]
        
        # Default mais conservador
        return 0.001

    def _log_trade_metrics(self):
        """Loga métricas atuais de trading"""
        self.logger.info(f"📊 Métricas: Trades Hoje={self.daily_trades}/"
                        f"{self.config['max_daily_trades']}, "
                        f"Total Trades={self.total_trades}, "
                        f"PnL Diário=${self.daily_pnl:.2f}")

    def show_portfolio_status(self):
        """Mostra status completo do portfólio"""
        self.logger.info("📈 Status do Portfólio:")
        self.profit_tracker.show_current_status()
        self._log_trade_metrics()

    def _reset_daily_counters(self):
        """Reseta contadores diários"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_reset = now
            self.logger.info("🔄 Contadores diários resetados")

    async def can_trade(self):
        """Verifica se pode executar novos trades"""
        self._reset_daily_counters()

        if self.daily_trades >= self.config['max_daily_trades']:
            self.logger.warning("⏸️ Limite diário de trades atingido")
            return False

        if self.daily_pnl <= self.config['daily_loss_limit']:
            self.logger.warning("⏸️ Limite diário de perda atingido")
            return False

        return True

    async def get_risk_report(self):
        """Gera relatório completo de risco"""
        report = {
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.config['max_daily_trades'],
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.config['daily_loss_limit'],
            'total_trades': self.total_trades,
            'open_positions': len(self.open_positions),
            'cooldown_active': len(self.last_trade_time),
        }
        
        self.logger.info("📋 Relatório de Risco:")
        for key, value in report.items():
            self.logger.info(f"   {key}: {value}")
            
        return report
