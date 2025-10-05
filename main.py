# main.py (VERSÃO COM WARNING CORRIGIDO)
import asyncio
import logging
import signal
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from config_manager import ConfigManager
from risk_manager import RiskManager
from market_analyzer import MarketAnalyzer
from trader import Trader
from profit_tracker import ProfitTracker


class TradingBot:
    def __init__(self, config_path: str = 'config.yaml'):
        """Inicializa o bot de trading com todos os módulos"""
        self.config_path = config_path
        self.is_running = False
        self.start_time = datetime.now()
        
        # Configuração inicial
        self._setup_logging()
        self.logger = logging.getLogger('TradingBot')
        
        # Carrega configuração
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Inicializa módulos
        self._initialize_modules()
        
        # Configura graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("🤖 TradingBot inicializado com sucesso")

    def _setup_logging(self):
        """Configura sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('trading_bot.log', encoding='utf-8')
            ]
        )

    def _initialize_modules(self):
        """Inicializa todos os módulos do bot"""
        try:
            self.risk_manager = RiskManager(self.config['risk'])
            self.market_analyzer = MarketAnalyzer(self.config['trading'])
            self.trader = Trader(self.config['exchange'])
            self.profit_tracker = ProfitTracker(self.config)
            
            self.logger.info("✅ Todos os módulos inicializados")
            
        except Exception as e:
            self.logger.error(f"❌ Erro inicializando módulos: {e}")
            raise

    def _setup_signal_handlers(self):
        """Configura handlers para graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handler para sinais de shutdown"""
        self.logger.info(f"🛑 Recebido sinal {signum}, encerrando bot...")
        self.is_running = False

    async def analyze_markets(self) -> List[Dict]:
        """Analisa todos os mercados configurados"""
        signals = []
        symbols = self.config['trading'].get('enabled_pairs', ['BTC/USDT', 'ETH/USDT'])
        
        self.logger.info(f"🔍 Analisando {len(symbols)} mercados...")
        
        for symbol in symbols:
            try:
                # Obtém dados de mercado
                market_data = await self._get_market_data(symbol)
                if market_data is None:
                    continue

                # Analisa o mercado
                analysis = await self.market_analyzer.analyze_market(symbol, market_data)
                if analysis and analysis.get('signals'):
                    signals.append({
                        'symbol': symbol,
                        'analysis': analysis,
                        'timestamp': datetime.now(),
                        'market_data': market_data
                    })
                    
                    self.logger.info(f"📈 Sinal encontrado para {symbol}: {analysis['signals']}")

            except Exception as e:
                self.logger.error(f"❌ Erro analisando {symbol}: {e}")
                continue

        return signals

    async def _get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Obtém dados de mercado para análise"""
        try:
            # Tenta obter dados reais da exchange
            market_data = await self.trader.get_ohlcv_data(symbol, timeframe, limit)
            
            if market_data is not None and not market_data.empty:
                self.logger.info(f"📊 Dados reais obtidos para {symbol}: {len(market_data)} candles")
                return market_data
            
            # Fallback: dados simulados se não conseguir obter dados reais
            self.logger.warning(f"⚠️  Usando dados simulados para {symbol}")
            return self._get_simulated_market_data(symbol, timeframe, limit)

        except Exception as e:
            self.logger.error(f"❌ Erro obtendo dados para {symbol}: {e}")
            return self._get_simulated_market_data(symbol, timeframe, limit)

    def _get_simulated_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Gera dados de mercado simulados para teste"""
        try:
            # Preços base para diferentes símbolos
            base_prices = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'ADA/USDT': 0.5,
                'BNB/USDT': 400,
                'XRP/USDT': 0.6
            }
            
            base_price = base_prices.get(symbol, 100)
            volatility = 0.02  # 2% de volatilidade
            
            # CORREÇÃO: Use 'h' em vez de 'H' para frequência de horas
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h')  # CORRIGIDO
            
            prices = [base_price]
            
            for i in range(1, limit):
                change = np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # Cria DataFrame OHLCV
            data = []
            for i, date in enumerate(dates):
                open_price = prices[i]
                close_price = prices[i] * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.normal(1000, 100)
                
                data.append([date, open_price, high_price, low_price, close_price, volume])
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"🎲 Dados simulados gerados para {symbol}: {len(df)} candles")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro gerando dados simulados: {e}")
            # Retorna DataFrame vazio como fallback
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    async def execute_trades(self, signals: List[Dict]):
        """Executa trades baseado nos sinais recebidos"""
        if not signals:
            self.logger.info("📭 Nenhum sinal para executar")
            return

        self.logger.info(f"🎯 Processando {len(signals)} sinais...")

        for signal_data in signals:
            try:
                symbol = signal_data['symbol']
                analysis = signal_data['analysis']
                
                # Verifica se pode trade
                if not await self.risk_manager.can_trade():
                    self.logger.warning("⏸️ Trading suspenso por gerenciamento de risco")
                    return

                # Prepara dados do mercado
                market = {
                    'symbol': symbol,
                    'current_price': analysis.get('current_price', 0),
                    'volume': analysis.get('volume', 0),
                    'volatility': analysis.get('volatility', 0)
                }

                # Processa cada sinal
                for signal in analysis.get('signals', []):
                    # Log do sinal
                    self.logger.info(f"📢 Sinal: {symbol} {signal['action'].upper()} "
                                   f"(Força: {signal['strength']:.2f}) - {signal['description']}")
                    
                    # Executa trade através do risk manager
                    trade_result = await self.risk_manager.execute_trade(
                        market=market,
                        signal=signal,
                        trader=self.trader
                    )
                    
                    if trade_result:
                        self.logger.info(f"✅ Trade executado com sucesso: {symbol}")
                        # Atualiza estatísticas
                        await self._update_trading_stats(symbol, signal['action'], True)
                    else:
                        self.logger.info(f"⏭️ Trade não executado para {symbol}")
                        await self._update_trading_stats(symbol, signal['action'], False)

            except Exception as e:
                self.logger.error(f"❌ Erro executando trade para {signal_data['symbol']}: {e}")

    async def _update_trading_stats(self, symbol: str, action: str, executed: bool):
        """Atualiza estatísticas de trading"""
        try:
            # Aqui você pode implementar lógica para salvar estatísticas
            # em banco de dados ou arquivo
            status = "EXECUTADO" if executed else "REJEITADO"
            self.logger.debug(f"📊 Estatística: {symbol} {action} - {status}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando estatísticas: {e}")

    async def monitor_positions(self):
        """Monitora posições abertas e executa saídas"""
        try:
            self.logger.info("👀 Monitorando posições abertas...")
            
            # Obter posições abertas do profit tracker
            open_positions = getattr(self.risk_manager.profit_tracker, 'positions', {})
            
            if open_positions:
                open_count = len([p for p in open_positions.values() if p.get('status') == 'open'])
                self.logger.info(f"📊 Posições abertas: {open_count}")
                
                # Verifica condições de saída para cada posição aberta
                for position_id, position in open_positions.items():
                    if position.get('status') == 'open':
                        await self._check_exit_conditions(position_id, position)
            else:
                self.logger.info("💼 Nenhuma posição aberta")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro monitorando posições: {e}")

    async def _check_exit_conditions(self, position_id: str, position: Dict):
        """Verifica condições de saída para uma posição"""
        try:
            symbol = position.get('symbol')
            current_price = await self.trader.get_current_price(symbol)
            
            if not current_price:
                return

            entry_price = position.get('entry_price', 0)
            side = position.get('side', 'buy')
            
            # Calcula P&L atual
            if side == 'buy':
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:  # sell (short)
                pnl_percent = (entry_price - current_price) / entry_price * 100
            
            # Condições de saída
            take_profit = 2.0  # 2%
            stop_loss = -1.0   # -1%
            
            if pnl_percent >= take_profit:
                self.logger.info(f"🎯 Take profit atingido para {symbol}: {pnl_percent:.2f}%")
                await self._close_position(position_id, position, current_price, "TAKE_PROFIT")
            elif pnl_percent <= stop_loss:
                self.logger.warning(f"🛑 Stop loss atingido para {symbol}: {pnl_percent:.2f}%")
                await self._close_position(position_id, position, current_price, "STOP_LOSS")
            else:
                self.logger.debug(f"📈 Posição {symbol}: {pnl_percent:.2f}%")
                
        except Exception as e:
            self.logger.error(f"❌ Erro verificando condições de saída: {e}")

    async def _close_position(self, position_id: str, position: Dict, current_price: float, reason: str):
        """Fecha uma posição aberta"""
        try:
            symbol = position.get('symbol')
            side = position.get('side', 'buy')
            
            # Determina ação oposta para fechar
            close_side = 'sell' if side == 'buy' else 'buy'
            amount = position.get('size', 0)
            
            if amount <= 0:
                self.logger.error(f"❌ Quantidade inválida para fechar posição: {amount}")
                return

            # Executa ordem para fechar posição
            close_order = await self.trader.place_market_order(symbol, close_side, amount)
            
            if close_order:
                # Registra saída no profit tracker
                await self.risk_manager.profit_tracker.record_exit(
                    position_id, current_price, reason
                )
                self.logger.info(f"✅ Posição fechada: {symbol} {close_side} - Motivo: {reason}")
            else:
                self.logger.error(f"❌ Falha ao fechar posição: {symbol}")
                
        except Exception as e:
            self.loglogger.error(f"❌ Erro fechando posição: {e}")

    async def generate_report(self):
        """Gera relatório completo do bot"""
        try:
            runtime = datetime.now() - self.start_time
            
            report = {
                'runtime': str(runtime),
                'start_time': self.start_time.isoformat(),
                'status': 'running' if self.is_running else 'stopped',
                'modules_healthy': True
            }
            
            # Adiciona relatório de risco
            risk_report = await self.risk_manager.get_risk_report()
            report.update({'risk_report': risk_report})
            
            # Adiciona estatísticas do trader
            trader_stats = self.trader.get_trading_stats()
            report.update({'trader_stats': trader_stats})
            
            self.logger.info("📊 RELATÓRIO DO BOT:")
            self.logger.info(f"   ⏱️  Tempo de execução: {runtime}")
            self.logger.info(f"   🎯 Status: {report['status']}")
            self.logger.info(f"   📈 Trades hoje: {risk_report['daily_trades']}/{risk_report['max_daily_trades']}")
            self.logger.info(f"   💰 P&L Diário: ${risk_report['daily_pnl']:.2f}")
            self.logger.info(f"   🔄 Total de ordens: {trader_stats['total_orders']}")
            self.logger.info(f"   ✅ Ordens bem-sucedidas: {trader_stats['successful_orders']}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Erro gerando relatório: {e}")
            return {}

    async def health_check(self):
        """Verifica saúde de todos os módulos"""
        try:
            health_status = {
                'bot': 'healthy',
                'risk_manager': 'healthy',
                'market_analyzer': 'healthy',
                'trader': 'healthy',
                'profit_tracker': 'healthy'
            }
            
            # Testa conexão com a exchange
            trader_healthy = await self.trader.test_connection()
            if not trader_healthy:
                health_status['trader'] = 'unhealthy'
                health_status['bot'] = 'degraded'
            
            self.logger.info("❤️  Health Check:")
            for module, status in health_status.items():
                self.logger.info(f"   {module}: {status}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Erro no health check: {e}")
            return {'bot': 'unhealthy'}

    async def run(self):
        """Loop principal do bot"""
        self.is_running = True
        self.start_time = datetime.now()
        
        self.logger.info("🚀 Iniciando TradingBot...")
        self.logger.info(f"⭐ Configuração: {self.config['trading'].get('strategy', 'Padrão')}")
        self.logger.info(f"💰 Parâmetros de Risco: {self.config['risk']}")

        # Health check inicial
        health = await self.health_check()
        if health['bot'] == 'unhealthy':
            self.logger.error("❌ Bot não está saudável. Encerrando...")
            return

        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                self.logger.info(f"🔄 Iteração #{iteration}")
                
                # 1. Health check periódico
                if iteration % 10 == 0:
                    await self.health_check()
                
                # 2. Análise de mercado
                signals = await self.analyze_markets()
                
                # 3. Execução de trades
                await self.execute_trades(signals)
                
                # 4. Monitoramento de posições
                await self.monitor_positions()
                
                # 5. Relatório periódico
                if iteration % 5 == 0:  # A cada 5 iterações
                    await self.generate_report()
                    self.risk_manager.show_portfolio_status()
                
                # 6. Aguarda próximo ciclo
                interval = self.config['trading'].get('analysis_interval', 60)
                self.logger.info(f"⏰ Aguardando {interval} segundos...")
                await asyncio.sleep(interval)
                
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no loop principal: {e}")
            
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Desligamento graceful do bot"""
        self.is_running = False
        self.logger.info("🛑 Encerrando TradingBot...")
        
        # Fecha conexão com a exchange
        await self.trader.close()
        
        # Gera relatório final
        final_report = await self.generate_report()
        self.logger.info("📋 RELATÓRIO FINAL:")
        for key, value in final_report.items():
            if key != 'risk_report' and key != 'trader_stats':
                self.logger.info(f"   {key}: {value}")
        
        self.logger.info("👋 TradingBot encerrado com sucesso")


async def main():
    """Função principal"""
    try:
        # Inicializa e executa o bot
        bot = TradingBot()
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n👋 Encerrado pelo usuário")
        
    except Exception as e:
        logging.error(f"❌ Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configura event loop para Windows (se necessário)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Executa o bot
    asyncio.run(main())
