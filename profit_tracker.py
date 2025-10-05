# profit_tracker.py (VERSÃO COMPLETA E CORRIGIDA)
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio


class ProfitTracker:
    def __init__(self, config: Dict):
        """Inicializa o rastreador de lucros e perdas"""
        self.config = config
        self.logger = self._setup_logger()
        
        # Estruturas de dados
        self.positions = {}  # Posições abertas
        self.trade_history = []  # Histórico de trades fechados
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }
        
        # Métricas diárias
        self.daily_metrics = {
            'date': datetime.now().date(),
            'trades_today': 0,
            'pnl_today': 0,
            'winning_trades_today': 0,
            'losing_trades_today': 0
        }
        
        self.logger.info("✅ ProfitTracker inicializado")

    def _setup_logger(self) -> logging.Logger:
        """Configura sistema de logging"""
        logger = logging.getLogger('ProfitTracker')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def record_entry(self, trade: Dict, market: Dict, signal: Dict, entry_price: float):
        """Registra entrada de trade"""
        try:
            position_id = trade.get('id', f"pos_{len(self.positions) + 1}")
            
            position = {
                'id': position_id,
                'symbol': market['symbol'],
                'entry_time': datetime.now(),
                'entry_price': entry_price,
                'size': trade.get('amount', 0),
                'side': signal['action'],
                'signal_strength': signal.get('strength', 0),
                'signal_type': signal.get('type', 'UNKNOWN'),
                'status': 'open',
                'fees': trade.get('fee', {}),
                'current_pnl': 0,
                'current_pnl_percent': 0
            }
            
            self.positions[position_id] = position
            
            self.logger.info(f"📥 POSIÇÃO ABERTA: {market['symbol']} {signal['action']} "
                           f"{position['size']:.6f} @ ${entry_price:.2f}")
            
            # Atualiza métricas diárias
            self._update_daily_metrics('entry')
            
        except Exception as e:
            self.logger.error(f"❌ Erro registrando entrada: {e}")

    async def record_exit(self, position_id: str, exit_price: float, exit_reason: str):
        """Registra saída de trade e calcula P&L"""
        try:
            if position_id not in self.positions:
                self.logger.error(f"❌ Posição não encontrada: {position_id}")
                return

            position = self.positions[position_id]
            pnl = self._calculate_pnl(position, exit_price)
            pnl_percent = self._calculate_pnl_percent(position, exit_price)
            
            # Atualiza posição com dados de saída
            position.update({
                'exit_time': datetime.now(),
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'exit_reason': exit_reason,
                'status': 'closed',
                'holding_period': (datetime.now() - position['entry_time']).total_seconds() / 3600  # em horas
            })
            
            # Move para histórico
            self.trade_history.append(position.copy())
            
            # Remove das posições abertas
            del self.positions[position_id]
            
            # Atualiza métricas de performance
            self._update_performance_metrics(position)
            
            # Atualiza métricas diárias
            self._update_daily_metrics('exit', pnl)
            
            self.logger.info(f"📤 POSIÇÃO FECHADA: {position['symbol']} {position['side']} "
                           f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%) - Motivo: {exit_reason}")

        except Exception as e:
            self.logger.error(f"❌ Erro registrando saída: {e}")

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calcula P&L em valor absoluto"""
        try:
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            
            if side == 'buy':
                pnl = (exit_price - entry_price) * size
            else:  # sell/short
                pnl = (entry_price - exit_price) * size
                
            # Subtrai fees (se disponíveis)
            fees = position.get('fees', {}).get('cost', 0)
            pnl -= fees
            
            return pnl
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando P&L: {e}")
            return 0

    def _calculate_pnl_percent(self, position: Dict, exit_price: float) -> float:
        """Calcula P&L em percentual"""
        try:
            entry_price = position['entry_price']
            side = position['side']
            
            if side == 'buy':
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # sell/short
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
            return pnl_percent
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando P&L percentual: {e}")
            return 0

    def _update_performance_metrics(self, position: Dict):
        """Atualiza métricas de performance"""
        try:
            pnl = position['pnl']
            
            # Contadores básicos
            self.performance_metrics['total_trades'] += 1
            
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
                # Maior ganho
                if pnl > self.performance_metrics['largest_win']:
                    self.performance_metrics['largest_win'] = pnl
            elif pnl < 0:
                self.performance_metrics['losing_trades'] += 1
                # Maior perda
                if pnl < self.performance_metrics['largest_loss']:
                    self.performance_metrics['largest_loss'] = pnl
            
            # P&L total
            self.performance_metrics['total_pnl'] += pnl
            
            # Win rate
            total = self.performance_metrics['winning_trades'] + self.performance_metrics['losing_trades']
            if total > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_trades'] / total * 100
                )
            
            # Médias
            if self.performance_metrics['winning_trades'] > 0:
                self.performance_metrics['avg_win'] = (
                    self.performance_metrics['total_pnl'] / self.performance_metrics['winning_trades']
                )
            
            if self.performance_metrics['losing_trades'] > 0:
                self.performance_metrics['avg_loss'] = (
                    self.performance_metrics['total_pnl'] / self.performance_metrics['losing_trades']
                )
            
            # Profit factor
            gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
            
            if gross_loss > 0:
                self.performance_metrics['profit_factor'] = gross_profit / gross_loss
            else:
                self.performance_metrics['profit_factor'] = float('inf')
                
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando métricas: {e}")

    def _update_daily_metrics(self, action: str, pnl: float = 0):
        """Atualiza métricas diárias"""
        try:
            today = datetime.now().date()
            
            # Reseta métricas se for um novo dia
            if today != self.daily_metrics['date']:
                self.daily_metrics = {
                    'date': today,
                    'trades_today': 0,
                    'pnl_today': 0,
                    'winning_trades_today': 0,
                    'losing_trades_today': 0
                }
            
            if action == 'entry':
                self.daily_metrics['trades_today'] += 1
            elif action == 'exit':
                self.daily_metrics['pnl_today'] += pnl
                if pnl > 0:
                    self.daily_metrics['winning_trades_today'] += 1
                elif pnl < 0:
                    self.daily_metrics['losing_trades_today'] += 1
                    
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando métricas diárias: {e}")

    async def update_open_positions(self, current_prices: Dict[str, float]):
        """Atualiza P&L das posições abertas com preços atuais"""
        try:
            for position_id, position in self.positions.items():
                if position['status'] == 'open':
                    symbol = position['symbol']
                    current_price = current_prices.get(symbol)
                    
                    if current_price:
                        position['current_pnl'] = self._calculate_pnl(position, current_price)
                        position['current_pnl_percent'] = self._calculate_pnl_percent(position, current_price)
                        
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando posições abertas: {e}")

    def show_current_status(self):
        """Mostra status atual do portfólio"""
        try:
            open_positions = [p for p in self.positions.values() if p['status'] == 'open']
            closed_positions = self.trade_history
            
            self.logger.info("📊 STATUS DO PROFIT TRACKER:")
            self.logger.info(f"   💼 Posições Abertas: {len(open_positions)}")
            self.logger.info(f"   📈 Posições Fechadas: {len(closed_positions)}")
            self.logger.info(f"   🎯 Total Trades: {self.performance_metrics['total_trades']}")
            self.logger.info(f"   📊 Win Rate: {self.performance_metrics['win_rate']:.1f}%")
            self.logger.info(f"   💰 P&L Total: ${self.performance_metrics['total_pnl']:.2f}")
            self.logger.info(f"   📉 Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
            
            # Métricas diárias
            self.logger.info(f"   📅 Hoje ({self.daily_metrics['date']}):")
            self.logger.info(f"      • Trades: {self.daily_metrics['trades_today']}")
            self.logger.info(f"      • P&L: ${self.daily_metrics['pnl_today']:.2f}")
            self.logger.info(f"      • Wins: {self.daily_metrics['winning_trades_today']}")
            self.logger.info(f"      • Losses: {self.daily_metrics['losing_trades_today']}")
            
            # Posições abertas detalhadas
            if open_positions:
                self.logger.info("   🔍 POSIÇÕES ABERTAS:")
                for position in open_positions:
                    self.logger.info(f"      • {position['symbol']} {position['side']} "
                                   f"{position['size']:.6f} @ ${position['entry_price']:.2f} "
                                   f"(P&L: ${position['current_pnl']:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Erro mostrando status: {e}")

    def get_performance_report(self) -> Dict:
        """Retorna relatório completo de performance"""
        try:
            report = {
                'summary': self.performance_metrics.copy(),
                'daily': self.daily_metrics.copy(),
                'open_positions': len([p for p in self.positions.values() if p['status'] == 'open']),
                'closed_positions': len(self.trade_history),
                'timestamp': datetime.now()
            }
            
            # Adiciona estatísticas avançadas
            if self.trade_history:
                # P&L por símbolo
                pnl_by_symbol = {}
                for trade in self.trade_history:
                    symbol = trade['symbol']
                    if symbol not in pnl_by_symbol:
                        pnl_by_symbol[symbol] = 0
                    pnl_by_symbol[symbol] += trade['pnl']
                
                report['pnl_by_symbol'] = pnl_by_symbol
                
                # Taxa de sucesso por tipo de sinal
                success_by_signal = {}
                for trade in self.trade_history:
                    signal_type = trade.get('signal_type', 'UNKNOWN')
                    if signal_type not in success_by_signal:
                        success_by_signal[signal_type] = {'wins': 0, 'total': 0}
                    
                    success_by_signal[signal_type]['total'] += 1
                    if trade['pnl'] > 0:
                        success_by_signal[signal_type]['wins'] += 1
                
                report['success_by_signal'] = success_by_signal
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Erro gerando relatório: {e}")
            return {}

    def get_open_positions(self) -> List[Dict]:
        """Retorna lista de posições abertas"""
        return [p for p in self.positions.values() if p['status'] == 'open']

    def get_closed_positions(self) -> List[Dict]:
        """Retorna lista de posições fechadas"""
        return self.trade_history.copy()

    def get_daily_pnl(self) -> float:
        """Retorna P&L do dia"""
        return self.daily_metrics['pnl_today']

    def get_total_pnl(self) -> float:
        """Retorna P&L total"""
        return self.performance_metrics['total_pnl']

    def get_win_rate(self) -> float:
        """Retorna taxa de acerto"""
        return self.performance_metrics['win_rate']

    def save_to_file(self, filename: str = 'trading_history.json'):
        """Salva histórico em arquivo JSON"""
        try:
            data = {
                'positions': self.positions,
                'trade_history': self.trade_history,
                'performance_metrics': self.performance_metrics,
                'daily_metrics': self.daily_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Converte datetime para string
            for key in ['positions', 'trade_history']:
                for item in data[key]:
                    if 'entry_time' in item:
                        item['entry_time'] = item['entry_time'].isoformat()
                    if 'exit_time' in item:
                        item['exit_time'] = item['exit_time'].isoformat()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Histórico salvo em: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro salvando histórico: {e}")

    def load_from_file(self, filename: str = 'trading_history.json'):
        """Carrega histórico de arquivo JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converte string para datetime
            for key in ['positions', 'trade_history']:
                for item in data[key]:
                    if 'entry_time' in item:
                        item['entry_time'] = datetime.fromisoformat(item['entry_time'])
                    if 'exit_time' in item:
                        item['exit_time'] = datetime.fromisoformat(item['exit_time'])
            
            self.positions = data.get('positions', {})
            self.trade_history = data.get('trade_history', [])
            self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
            self.daily_metrics = data.get('daily_metrics', self.daily_metrics)
            
            self.logger.info(f"📂 Histórico carregado de: {filename}")
            
        except FileNotFoundError:
            self.logger.warning(f"📂 Arquivo de histórico não encontrado: {filename}")
        except Exception as e:
            self.logger.error(f"❌ Erro carregando histórico: {e}")

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calcula Sharpe Ratio (anualizado)"""
        try:
            if not self.trade_history:
                return 0.0
            
            # P&L diário (simplificado)
            daily_returns = []
            current_date = None
            daily_pnl = 0
            
            for trade in sorted(self.trade_history, key=lambda x: x['exit_time']):
                trade_date = trade['exit_time'].date()
                
                if current_date is None:
                    current_date = trade_date
                
                if trade_date == current_date:
                    daily_pnl += trade['pnl']
                else:
                    if daily_pnl != 0:
                        daily_returns.append(daily_pnl)
                    daily_pnl = trade['pnl']
                    current_date = trade_date
            
            if daily_pnl != 0:
                daily_returns.append(daily_pnl)
            
            if len(daily_returns) < 2:
                return 0.0
            
            returns = np.array(daily_returns)
            excess_returns = returns - (risk_free_rate / 365)
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(365)
            return float(sharpe)
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando Sharpe Ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self) -> float:
        """Calcula máximo drawdown"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Ordena trades por tempo
            sorted_trades = sorted(self.trade_history, key=lambda x: x['exit_time'])
            
            cumulative_pnl = 0
            peak = 0
            max_drawdown = 0
            
            for trade in sorted_trades:
                cumulative_pnl += trade['pnl']
                
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                
                drawdown = peak - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando máximo drawdown: {e}")
            return 0.0

    def reset_daily_metrics(self):
        """Reseta métricas diárias (para uso externo)"""
        today = datetime.now().date()
        self.daily_metrics = {
            'date': today,
            'trades_today': 0,
            'pnl_today': 0,
            'winning_trades_today': 0,
            'losing_trades_today': 0
        }
        self.logger.info("🔄 Métricas diárias resetadas")
