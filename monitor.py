# monitor.py - Monitor em tempo real
import time
import json
from datetime import datetime
import pandas as pd

class BotMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.metrics_history = []
        
    def collect_metrics(self):
        """Coleta métricas atuais do bot"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'iteration': getattr(self.bot, 'current_iteration', 0),
                'open_positions': len(getattr(self.bot.risk_manager.profit_tracker, 'positions', {})),
                'daily_trades': getattr(self.bot.risk_manager, 'daily_trades', 0),
                'total_trades': getattr(self.bot.risk_manager, 'total_trades', 0),
                'trader_orders': self.bot.trader.get_trading_stats()['total_orders']
            }
            self.metrics_history.append(metrics)
            return metrics
        except Exception as e:
            print(f"❌ Erro coletando métricas: {e}")
            return {}
    
    def show_realtime_dashboard(self):
        """Mostra dashboard em tempo real"""
        metrics = self.collect_metrics()
        print("\n" + "="*60)
        print("📊 DASHBOARD DO TRADING BOT - TEMPO REAL")
        print("="*60)
        print(f"🕒 Última atualização: {metrics.get('timestamp', 'N/A')}")
        print(f"🔄 Iteração: {metrics.get('iteration', 0)}")
        print(f"💼 Posições abertas: {metrics.get('open_positions', 0)}")
        print(f"📈 Trades hoje: {metrics.get('daily_trades', 0)}")
        print(f"🏆 Total de trades: {metrics.get('total_trades', 0)}")
        print(f"🎯 Ordens executadas: {metrics.get('trader_orders', 0)}")
        print("="*60)
