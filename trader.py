# trader.py (VERSÃO COM get_ohlcv_data IMPLEMENTADO)
import ccxt
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class Trader:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logger()
        self.exchange = None
        self.initialized = False
        
        # Verifica se é sandbox mode
        self.sandbox_mode = config.get('sandbox', True)
        
        if not self.sandbox_mode:
            self._initialize_exchange()
        else:
            self.logger.info("🔧 Modo Sandbox ativado - usando trader simulado")
            self.initialized = True

        # Estatísticas
        self.trading_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0
        }

    def _setup_logger(self):
        logger = logging.getLogger('Trader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_exchange(self):
        """Inicializa conexão real com a exchange"""
        try:
            exchange_name = self.config.get('name', 'binance')
            
            if exchange_name == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': self.config.get('api_key', ''),
                    'secret': self.config.get('secret', ''),
                    'sandbox': False,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"Exchange não suportada: {exchange_name}")

            self.exchange.load_markets()
            self.initialized = True
            self.logger.info(f"✅ Exchange {exchange_name} inicializada")

        except Exception as e:
            self.logger.error(f"❌ Erro inicializando exchange: {e}")
            self.initialized = False

    async def get_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Obtém dados OHLCV da exchange ou gera dados simulados"""
        try:
            if self.sandbox_mode or not self.initialized:
                self.logger.info(f"📊 Gerando dados OHLCV simulados para {symbol}")
                return self._generate_ohlcv_data(symbol, timeframe, limit)
            
            # Tenta obter dados reais da exchange
            self.logger.info(f"📊 Obtendo dados OHLCV reais para {symbol}")
            ohlcv = await self._fetch_ohlcv(symbol, timeframe, limit)
            
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.logger.info(f"✅ Dados OHLCV obtidos: {symbol} - {len(df)} candles")
                return df
            else:
                self.logger.warning(f"⚠️  Não foi possível obter dados OHLCV para {symbol}, usando dados simulados")
                return self._generate_ohlcv_data(symbol, timeframe, limit)
                
        except Exception as e:
            self.logger.error(f"❌ Erro obtendo dados OHLCV para {symbol}: {e}")
            return self._generate_ohlcv_data(symbol, timeframe, limit)

    async def _fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """Busca dados OHLCV da exchange"""
        try:
            # Usa asyncio para evitar blocking
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, None, limit)
            )
            return ohlcv
        except Exception as e:
            self.logger.error(f"❌ Erro buscando OHLCV da exchange: {e}")
            return None

    def _generate_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Gera dados OHLCV simulados realistas"""
        try:
            # Preços base realistas
            base_prices = {
                'BTC/USDT': 50000,
                'ETH/USDT': 3000,
                'ADA/USDT': 0.5,
                'BNB/USDT': 400,
                'XRP/USDT': 0.6,
                'DOT/USDT': 7.0,
                'LINK/USDT': 15.0,
                'LTC/USDT': 70.0,
                'BCH/USDT': 300.0,
                'DOGE/USDT': 0.15
            }
            
            base_price = base_prices.get(symbol, 100)
            
            # Volatilidade baseada no símbolo
            volatility_map = {
                'BTC/USDT': 0.015,
                'ETH/USDT': 0.02,
                'ADA/USDT': 0.03,
                'BNB/USDT': 0.025,
                'XRP/USDT': 0.035
            }
            volatility = volatility_map.get(symbol, 0.02)
            
            # Gera timestamps baseado no timeframe
            if timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                freq = f'{hours}h'
            elif timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                freq = f'{minutes}min'
            else:
                freq = '1h'  # default
            
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=freq)
            
            # Gera preços com tendência e volatilidade realistas
            prices = [base_price]
            trend = np.random.normal(0, 0.001)  # Pequena tendência
            
            for i in range(1, limit):
                # Combina tendência + ruído
                change = trend + np.random.normal(0, volatility)
                new_price = max(0.01, prices[-1] * (1 + change))  # Preço nunca negativo
                prices.append(new_price)
            
            # Cria DataFrame OHLCV realista
            data = []
            for i, date in enumerate(dates):
                open_price = prices[i]
                
                # Gera high/low realistas (dentro de uma faixa do open)
                daily_volatility = volatility * 2  # Mais volatilidade intraday
                price_change = np.random.normal(0, daily_volatility)
                close_price = max(0.01, open_price * (1 + price_change))
                
                # High e Low baseados no movimento do dia
                if close_price > open_price:
                    high_price = close_price * (1 + abs(np.random.normal(0, 0.005)))
                    low_price = open_price * (1 - abs(np.random.normal(0, 0.003)))
                else:
                    high_price = open_price * (1 + abs(np.random.normal(0, 0.003)))
                    low_price = close_price * (1 - abs(np.random.normal(0, 0.005)))
                
                # Garante que high > low
                high_price = max(high_price, max(open_price, close_price))
                low_price = min(low_price, min(open_price, close_price))
                
                # Volume correlacionado com volatilidade
                base_volume = 1000
                volume_variation = abs(price_change) * 50000  # Mais volume em dias voláteis
                volume = max(100, base_volume + volume_variation + np.random.normal(0, 200))
                
                data.append([date, open_price, high_price, low_price, close_price, volume])
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            
            self.logger.debug(f"🎲 Dados OHLCV simulados gerados para {symbol}: {len(df)} candles")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro gerando dados OHLCV simulados: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    async def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Executa ordem market"""
        if not self.initialized:
            return None

        try:
            if self.sandbox_mode:
                return await self._place_mock_order(symbol, side, amount)
            else:
                return await self._place_real_order(symbol, side, amount)

        except Exception as e:
            self.logger.error(f"❌ Erro executando ordem: {e}")
            return None

    async def _place_mock_order(self, symbol: str, side: str, amount: float) -> Dict:
        """Executa ordem simulada"""
        try:
            # Simula delay de execução
            await asyncio.sleep(0.1)
            
            current_price = await self.get_current_price(symbol)
            order_id = f"MOCK_{int(time.time())}"
            
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'amount': amount,
                'price': current_price,
                'filled': amount,
                'status': 'closed',
                'timestamp': datetime.now(),
                'fee': {'cost': amount * current_price * 0.001, 'currency': 'USDT'}
            }
            
            self.trading_stats['total_orders'] += 1
            self.trading_stats['successful_orders'] += 1
            self.trading_stats['total_volume'] += amount
            
            self.logger.info(f"🤖 [MOCK] Ordem executada: {symbol} {side} {amount:.6f} @ ${current_price:.2f}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"❌ Erro em ordem simulada: {e}")
            return None

    async def _place_real_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Executa ordem real na exchange"""
        try:
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=amount
                )
            )
            
            self.trading_stats['total_orders'] += 1
            self.trading_stats['successful_orders'] += 1
            self.trading_stats['total_volume'] += amount
            
            self.logger.info(f"✅ Ordem real executada: {symbol} {side} {amount:.6f}")
            return order

        except Exception as e:
            self.logger.error(f"❌ Erro executando ordem real: {e}")
            self.trading_stats['failed_orders'] += 1
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual"""
        try:
            if self.sandbox_mode:
                return self._get_mock_price(symbol)
            else:
                ticker = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.exchange.fetch_ticker(symbol)
                )
                return ticker['last'] if ticker else None

        except Exception as e:
            self.logger.error(f"❌ Erro obtendo preço: {e}")
            return self._get_mock_price(symbol)

    def _get_mock_price(self, symbol: str) -> float:
        """Retorna preço simulado baseado nos dados OHLCV"""
        try:
            # Usa o último preço dos dados OHLCV simulados para consistência
            ohlcv_data = self._generate_ohlcv_data(symbol, '1h', 2)  # Apenas 2 candles
            if not ohlcv_data.empty:
                return ohlcv_data['close'].iloc[-1]
            
            # Fallback para preços base
            base_prices = {
                'BTC/USDT': 50000.0,
                'ETH/USDT': 3000.0,
                'ADA/USDT': 0.5,
            }
            return base_prices.get(symbol, 100.0)
            
        except Exception as e:
            self.logger.error(f"❌ Erro obtendo preço mock: {e}")
            return 100.0

    async def get_account_balance(self) -> Dict[str, float]:
        """Obtém saldo da conta"""
        try:
            if self.sandbox_mode:
                return {
                    'USDT': 1000.0,
                    'BTC': 0.01,
                    'ETH': 0.1,
                    'ADA': 100.0
                }
            else:
                balance = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fetch_balance()
                )
                return {
                    'USDT': balance['USDT']['free'],
                    'BTC': balance['BTC']['free'],
                    'ETH': balance['ETH']['free']
                }

        except Exception as e:
            self.logger.error(f"❌ Erro obtendo saldo: {e}")
            return {'USDT': 1000.0}

    async def test_connection(self) -> bool:
        """Testa conexão"""
        if self.sandbox_mode:
            self.logger.info("✅ Conexão sandbox OK")
            return True
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_balance()
            )
            self.logger.info("✅ Conexão com exchange OK")
            return True
        except Exception as e:
            self.logger.error(f"❌ Falha na conexão: {e}")
            return False

    def get_trading_stats(self) -> Dict:
        return self.trading_stats.copy()

    async def close(self):
        """Fecha conexão"""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("🔌 Conexão com exchange fechada")
