# market_analyzer.py (VERSÃO COMPLETA E CORRIGIDA - SEM TA-Lib)
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio


class MarketAnalyzer:
    def __init__(self, config: Dict):
        """Inicializa o analisador de mercado com configurações"""
        self.config = config
        self.logger = self._setup_logger()
        self.technical_indicators = {}
        self.market_conditions = {}
        
        # Configurações de análise
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.bb_period = config.get('bollinger_period', 20)
        self.bb_std = config.get('bollinger_std', 2)
        self.volume_ma_period = config.get('volume_ma_period', 20)
        
        self.logger.info("✅ MarketAnalyzer inicializado (sem TA-Lib)")

    def _setup_logger(self) -> logging.Logger:
        """Configura sistema de logging"""
        logger = logging.getLogger('MarketAnalyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analisa mercado completo com múltiplos indicadores e gera sinais
        """
        try:
            if data.empty or len(data) < 50:
                self.logger.warning(f"📊 Dados insuficientes para {symbol}")
                return {}

            self.logger.info(f"🔍 Analisando {symbol} com {len(data)} candles")

            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': data['close'].iloc[-1],
                'signals': [],
                'confidence': 0,
                'market_conditions': {},
                'technical_indicators': {},
                'risk_metrics': {},
                'recommendation': 'HOLD'
            }

            # Análise técnica
            technical_analysis = self._technical_analysis(data)
            analysis['technical_indicators'] = technical_analysis

            # Análise de tendência
            trend_analysis = self._trend_analysis(data)
            analysis['market_conditions'].update(trend_analysis)

            # Análise de volatilidade
            volatility_analysis = self._volatility_analysis(data)
            analysis['market_conditions'].update(volatility_analysis)

            # Análise de volume
            volume_analysis = self._volume_analysis(data)
            analysis['market_conditions'].update(volume_analysis)

            # Avaliação de risco
            risk_assessment = self._risk_assessment(data, technical_analysis)
            analysis['risk_metrics'] = risk_assessment

            # Geração de sinais
            signals = self._generate_signals(technical_analysis, trend_analysis, data)
            analysis['signals'] = signals

            # Confiança geral
            analysis['confidence'] = self._calculate_confidence(analysis)
            
            # Recomendação final
            analysis['recommendation'] = self._generate_recommendation(analysis)

            self.logger.info(f"📈 Análise concluída para {symbol}: "
                           f"{analysis['recommendation']} (Confiança: {analysis['confidence']:.1f}%)")

            return analysis

        except Exception as e:
            self.logger.error(f"❌ Erro analisando {symbol}: {e}")
            return {}

    def _technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Executa análise técnica completa SEM TA-Lib"""
        try:
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            volume = data['volume'].values

            indicators = {}

            # RSI - Implementação manual
            indicators['rsi'] = self._calculate_rsi(close_prices, self.rsi_period)
            indicators['rsi_trend'] = self._get_rsi_trend(close_prices)

            # MACD - Implementação manual
            macd, macd_signal, macd_hist = self._calculate_macd(close_prices)
            indicators['macd'] = macd[-1] if len(macd) > 0 else 0
            indicators['macd_signal'] = macd_signal[-1] if len(macd_signal) > 0 else 0
            indicators['macd_histogram'] = macd_hist[-1] if len(macd_hist) > 0 else 0
            indicators['macd_trend'] = 'bullish' if indicators['macd_histogram'] > 0 else 'bearish'

            # Bollinger Bands - Implementação manual
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            indicators['bb_upper'] = bb_upper[-1] if len(bb_upper) > 0 else 0
            indicators['bb_middle'] = bb_middle[-1] if len(bb_middle) > 0 else 0
            indicators['bb_lower'] = bb_lower[-1] if len(bb_lower) > 0 else 0
            indicators['bb_position'] = self._get_bb_position(close_prices[-1], indicators['bb_upper'], indicators['bb_lower'])

            # Médias Móveis
            indicators['sma_20'] = self._sma(close_prices, 20)[-1] if len(close_prices) >= 20 else close_prices[-1]
            indicators['sma_50'] = self._sma(close_prices, 50)[-1] if len(close_prices) >= 50 else close_prices[-1]
            indicators['ema_12'] = self._ema(close_prices, 12)[-1] if len(close_prices) >= 12 else close_prices[-1]
            indicators['ema_26'] = self._ema(close_prices, 26)[-1] if len(close_prices) >= 26 else close_prices[-1]

            # Suporte e Resistência
            indicators['support_resistance'] = self._find_support_resistance(data)

            # Stochastic - Implementação manual
            stoch_k, stoch_d = self._calculate_stochastic(high_prices, low_prices, close_prices)
            indicators['stoch_k'] = stoch_k[-1] if len(stoch_k) > 0 else 50
            indicators['stoch_d'] = stoch_d[-1] if len(stoch_d) > 0 else 50

            # Williams %R - Implementação manual
            indicators['williams_r'] = self._calculate_williams_r(high_prices, low_prices, close_prices)

            # CCI - Implementação manual
            indicators['cci'] = self._calculate_cci(high_prices, low_prices, close_prices)

            # ATR - Volatilidade
            indicators['atr'] = self._calculate_atr(high_prices, low_prices, close_prices)

            # OBV - Volume
            indicators['obv'] = self._calculate_obv(close_prices, volume)

            return indicators

        except Exception as e:
            self.logger.error(f"❌ Erro na análise técnica: {e}")
            return {}

    # ===== IMPLEMENTAÇÕES MANUAIS DOS INDICADORES =====

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI manualmente"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')

        # Evita divisão por zero
        avg_losses = np.where(avg_losses == 0, 0.0001, avg_losses)
        
        rs = avg_gains[-1] / avg_losses[-1] if len(avg_gains) > 0 and len(avg_losses) > 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula MACD manualmente"""
        if len(prices) < self.macd_slow:
            return np.array([]), np.array([]), np.array([])

        ema_fast = self._ema(prices, self.macd_fast)
        ema_slow = self._ema(prices, self.macd_slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA do MACD)
        signal_line = self._ema(macd_line, self.macd_signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula Bollinger Bands manualmente"""
        if len(prices) < self.bb_period:
            return np.array([]), np.array([]), np.array([])

        sma = self._sma(prices, self.bb_period)
        std = pd.Series(prices).rolling(window=self.bb_period).std().values
        
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        
        return upper_band, sma, lower_band

    def _sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcula Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcula Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula Stochastic Oscillator"""
        if len(high) < k_period:
            return np.array([]), np.array([])

        stoch_k = []
        for i in range(k_period - 1, len(close)):
            high_period = high[i - k_period + 1:i + 1]
            low_period = low[i - k_period + 1:i + 1]
            current_close = close[i]
            
            highest_high = np.max(high_period)
            lowest_low = np.min(low_period)
            
            if highest_high != lowest_low:
                k_value = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
            else:
                k_value = 50
                
            stoch_k.append(k_value)
        
        stoch_k = np.array(stoch_k)
        stoch_d = self._sma(stoch_k, d_period) if len(stoch_k) >= d_period else stoch_k.copy()
        
        return stoch_k, stoch_d

    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             period: int = 14) -> float:
        """Calcula Williams %R"""
        if len(high) < period:
            return -50.0

        recent_high = np.max(high[-period:])
        recent_low = np.min(low[-period:])
        current_close = close[-1]
        
        if recent_high != recent_low:
            williams_r = -100 * (recent_high - current_close) / (recent_high - recent_low)
        else:
            williams_r = -50.0
            
        return float(williams_r)

    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      period: int = 20) -> float:
        """Calcula Commodity Channel Index"""
        if len(high) < period:
            return 0.0

        typical_price = (high[-period:] + low[-period:] + close[-period:]) / 3
        sma_tp = np.mean(typical_price)
        mean_deviation = np.mean(np.abs(typical_price - sma_tp))
        
        if mean_deviation != 0:
            cci = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)
        else:
            cci = 0.0
            
        return float(cci)

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      period: int = 14) -> float:
        """Calcula Average True Range"""
        if len(high) < period + 1:
            return 0.0

        tr = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range = max(tr1, tr2, tr3)
            tr.append(true_range)
        
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr) if tr else 0.0
        return float(atr)

    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calcula On Balance Volume"""
        if len(close) < 2:
            return 0.0

        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
        
        return float(obv)

    # ===== MÉTODOS RESTANTES =====

    def _trend_analysis(self, data: pd.DataFrame) -> Dict:
        """Analisa tendência do mercado"""
        try:
            close_prices = data['close'].values
            
            short_trend = self._calculate_trend(close_prices[-20:]) if len(close_prices) >= 20 else 'NEUTRAL'
            medium_trend = self._calculate_trend(close_prices[-50:]) if len(close_prices) >= 50 else 'NEUTRAL'
            long_trend = self._calculate_trend(close_prices[-100:]) if len(close_prices) >= 100 else 'NEUTRAL'
            
            trend_alignment = 'ALIGNED' if short_trend == medium_trend == long_trend else 'MIXED'
            trend_strength = self._calculate_trend_strength(close_prices)
            
            return {
                'short_term_trend': short_trend,
                'medium_term_trend': medium_trend,
                'long_term_trend': long_trend,
                'trend_alignment': trend_alignment,
                'trend_strength': trend_strength,
                'primary_trend': medium_trend
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de tendência: {e}")
            return {}

    def _volatility_analysis(self, data: pd.DataFrame) -> Dict:
        """Analisa volatilidade do mercado"""
        try:
            close_prices = data['close'].values
            returns = np.diff(np.log(close_prices)) if len(close_prices) > 1 else np.array([0])
            
            historical_vol = np.std(returns) * np.sqrt(365) * 100 if len(returns) > 0 else 0
            recent_vol = np.std(returns[-10:]) * np.sqrt(365) * 100 if len(returns) >= 10 else historical_vol
            
            bb_width = (data['close'].rolling(20).std() * 2 * 100 / data['close'].rolling(20).mean()).iloc[-1] if len(data) >= 20 else 0
            
            if historical_vol < 30:
                vol_class = 'LOW'
            elif historical_vol < 70:
                vol_class = 'MEDIUM'
            else:
                vol_class = 'HIGH'
                
            return {
                'historical_volatility': historical_vol,
                'recent_volatility': recent_vol,
                'volatility_class': vol_class,
                'bb_width': bb_width,
                'volatility_trend': 'INCREASING' if recent_vol > historical_vol else 'DECREASING'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de volatilidade: {e}")
            return {}

    def _volume_analysis(self, data: pd.DataFrame) -> Dict:
        """Analisa volume e liquidez"""
        try:
            volume = data['volume'].values
            close_prices = data['close'].values
            
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume) if len(volume) > 0 else 1
            current_volume = volume[-1] if len(volume) > 0 else 1
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_trend = 'INCREASING' if volume_ratio > 1.2 else 'DECREASING' if volume_ratio < 0.8 else 'STABLE'
            
            volume_price_correlation = self._calculate_volume_price_correlation(close_prices, volume)
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_class': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.5 else 'NORMAL',
                'volume_price_correlation': volume_price_correlation,
                'current_volume': current_volume,
                'average_volume': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de volume: {e}")
            return {}

    def _risk_assessment(self, data: pd.DataFrame, technical_indicators: Dict) -> Dict:
        """Avalia risco do mercado"""
        try:
            close_prices = data['close'].values
            current_price = close_prices[-1] if len(close_prices) > 0 else 0
            
            recent_high = np.max(close_prices[-20:]) if len(close_prices) >= 20 else current_price
            recent_low = np.min(close_prices[-20:]) if len(close_prices) >= 20 else current_price
            current_drawdown = (current_price - recent_high) / recent_high * 100 if recent_high > 0 else 0
            
            risk_score = 0
            rsi = technical_indicators.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                risk_score += 2
                
            volatility = self._volatility_analysis(data).get('recent_volatility', 50)
            if volatility > 70:
                risk_score += 2
            elif volatility > 40:
                risk_score += 1
                
            volume_ratio = self._volume_analysis(data).get('volume_ratio', 1)
            if volume_ratio < 0.5:
                risk_score += 1
                
            if risk_score >= 4:
                risk_class = 'HIGH'
            elif risk_score >= 2:
                risk_class = 'MEDIUM'
            else:
                risk_class = 'LOW'
                
            return {
                'risk_score': risk_score,
                'risk_class': risk_class,
                'current_drawdown': current_drawdown,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'risk_factors': {
                    'rsi_extreme': rsi > 80 or rsi < 20,
                    'high_volatility': volatility > 70,
                    'low_volume': volume_ratio < 0.5
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro na avaliação de risco: {e}")
            return {}

    def _generate_signals(self, technical_indicators: Dict, trend_analysis: Dict, data: pd.DataFrame) -> List[Dict]:
        """Gera sinais de trading baseado na análise"""
        signals = []
        current_price = data['close'].iloc[-1] if len(data) > 0 else 0
        
        try:
            # Sinal RSI
            rsi_signal = self._generate_rsi_signal(technical_indicators.get('rsi', 50))
            if rsi_signal:
                signals.append(rsi_signal)
                
            # Sinal MACD
            macd_signal = self._generate_macd_signal(
                technical_indicators.get('macd', 0),
                technical_indicators.get('macd_signal', 0),
                technical_indicators.get('macd_histogram', 0)
            )
            if macd_signal:
                signals.append(macd_signal)
                
            # Sinal Bollinger Bands
            bb_signal = self._generate_bb_signal(
                current_price,
                technical_indicators.get('bb_upper', 0),
                technical_indicators.get('bb_lower', 0),
                technical_indicators.get('bb_middle', 0)
            )
            if bb_signal:
                signals.append(bb_signal)
                
            # Sinal Tendência
            trend_signal = self._generate_trend_signal(trend_analysis)
            if trend_signal:
                signals.append(trend_signal)
                
            # Consolida sinais
            signals = self._consolidate_signals(signals)
            
            self.logger.info(f"🎯 {len(signals)} sinais gerados")
            
        except Exception as e:
            self.logger.error(f"❌ Erro gerando sinais: {e}")
            
        return signals

    def _generate_rsi_signal(self, rsi: float) -> Optional[Dict]:
        """Gera sinal baseado no RSI"""
        if rsi > 70:
            return {
                'type': 'RSI_OVERBOUGHT',
                'action': 'sell',
                'strength': min((rsi - 70) / 30, 1.0),
                'description': f'RSI sobrecomprado: {rsi:.1f}'
            }
        elif rsi < 30:
            return {
                'type': 'RSI_OVERSOLD', 
                'action': 'buy',
                'strength': min((30 - rsi) / 30, 1.0),
                'description': f'RSI sobrevendido: {rsi:.1f}'
            }
        return None

    def _generate_macd_signal(self, macd: float, signal: float, histogram: float) -> Optional[Dict]:
        """Gera sinal baseado no MACD"""
        if macd > signal and histogram > 0:
            return {
                'type': 'MACD_BULLISH',
                'action': 'buy',
                'strength': min(abs(histogram) * 10, 1.0),
                'description': f'MACD bullish: {histogram:.4f}'
            }
        elif macd < signal and histogram < 0:
            return {
                'type': 'MACD_BEARISH',
                'action': 'sell', 
                'strength': min(abs(histogram) * 10, 1.0),
                'description': f'MACD bearish: {histogram:.4f}'
            }
        return None

    def _generate_bb_signal(self, price: float, bb_upper: float, bb_lower: float, bb_middle: float) -> Optional[Dict]:
        """Gera sinal baseado nas Bollinger Bands"""
        if price >= bb_upper:
            return {
                'type': 'BB_OVERBOUGHT',
                'action': 'sell',
                'strength': 0.7,
                'description': f'Preço na banda superior: {price:.2f}'
            }
        elif price <= bb_lower:
            return {
                'type': 'BB_OVERSOLD',
                'action': 'buy', 
                'strength': 0.7,
                'description': f'Preço na banda inferior: {price:.2f}'
            }
        return None

    def _generate_trend_signal(self, trend_analysis: Dict) -> Optional[Dict]:
        """Gera sinal baseado na tendência"""
        primary_trend = trend_analysis.get('primary_trend', 'NEUTRAL')
        trend_strength = trend_analysis.get('trend_strength', 0)
        
        if primary_trend == 'UPTREND' and trend_strength > 0.6:
            return {
                'type': 'TREND_FOLLOWING',
                'action': 'buy',
                'strength': trend_strength,
                'description': f'Tendência de alta forte: {trend_strength:.1f}'
            }
        elif primary_trend == 'DOWNTREND' and trend_strength > 0.6:
            return {
                'type': 'TREND_FOLLOWING', 
                'action': 'sell',
                'strength': trend_strength,
                'description': f'Tendência de baixa forte: {trend_strength:.1f}'
            }
        return None

    def _consolidate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Consolida sinais removendo conflitos"""
        if not signals:
            return []
            
        buy_signals = [s for s in signals if s['action'] == 'buy']
        sell_signals = [s for s in signals if s['action'] == 'sell']
        
        consolidated = []
        
        if buy_signals:
            strongest_buy = max(buy_signals, key=lambda x: x['strength'])
            consolidated.append(strongest_buy)
            
        if sell_signals:
            strongest_sell = max(sell_signals, key=lambda x: x['strength'])
            consolidated.append(strongest_sell)
            
        return consolidated

    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calcula confiança geral da análise"""
        try:
            confidence = 0.5
            signals = analysis.get('signals', [])
            
            if signals:
                avg_signal_strength = np.mean([s.get('strength', 0) for s in signals])
                confidence += avg_signal_strength * 0.3
                
            trend_alignment = analysis['market_conditions'].get('trend_alignment', 'MIXED')
            if trend_alignment == 'ALIGNED':
                confidence += 0.1
                
            volume_trend = analysis['market_conditions'].get('volume_trend', 'STABLE')
            if volume_trend == 'INCREASING':
                confidence += 0.05
                
            return max(0, min(1, confidence))
            
        except Exception as e:
            self.logger.error(f"❌ Erro calculando confiança: {e}")
            return 0.5

    def _generate_recommendation(self, analysis: Dict) -> str:
        """Gera recomendação final"""
        signals = analysis.get('signals', [])
        risk_class = analysis['risk_metrics'].get('risk_class', 'MEDIUM')
        confidence = analysis.get('confidence', 0.5)
        
        if not signals:
            return 'HOLD'
            
        if risk_class == 'HIGH' and confidence < 0.7:
            return 'HOLD'
            
        buy_count = len([s for s in signals if s['action'] == 'buy'])
        sell_count = len([s for s in signals if s['action'] == 'sell'])
        
        if buy_count > sell_count:
            return 'BUY'
        elif sell_count > buy_count:
            return 'SELL'
        else:
            return 'HOLD'

    # ===== MÉTODOS AUXILIARES =====

    def _get_rsi_trend(self, prices: np.ndarray) -> str:
        """Determina tendência do RSI"""
        if len(prices) < self.rsi_period + 5:
            return 'NEUTRAL'
            
        rsi_values = []
        for i in range(len(prices) - self.rsi_period + 1):
            rsi = self._calculate_rsi(prices[i:i + self.rsi_period], self.rsi_period)
            rsi_values.append(rsi)
            
        recent_rsi = rsi_values[-5:] if len(rsi_values) >= 5 else rsi_values
        
        if len(recent_rsi) < 2:
            return 'NEUTRAL'
            
        differences = np.diff(recent_rsi)
        
        if all(diff > 0 for diff in differences):
            return 'BULLISH'
        elif all(diff < 0 for diff in differences):
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> str:
        """Determina posição do preço nas Bollinger Bands"""
        if bb_upper == bb_lower or bb_upper <= bb_lower:
            return 'MIDDLE'
            
        position = (price - bb_lower) / (bb_upper - bb_lower)
        
        if position > 0.8:
            return 'UPPER'
        elif position < 0.2:
            return 'LOWER'
        else:
            return 'MIDDLE'

    def _find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Encontra níveis de suporte e resistência"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window]):
                    resistance_levels.append(highs[i])
                if lows[i] == min(lows[i-window:i+window]):
                    support_levels.append(lows[i])
                    
            resistance_levels = self._cluster_levels(resistance_levels, tolerance=0.005)
            support_levels = self._cluster_levels(support_levels, tolerance=0.005)
            
            return {
                'resistance': sorted(resistance_levels[-3:]),
                'support': sorted(support_levels[-3:])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro encontrando S/R: {e}")
            return {'resistance': [], 'support': []}

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        """Agrupa níveis próximos"""
        if not levels:
            return []
            
        levels.sort()
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
                
        clusters.append(np.mean(current_cluster))
        return clusters

    def _calculate_trend(self, prices: np.ndarray) -> str:
        """Calcula tendência dos preços"""
        if len(prices) < 2:
            return 'NEUTRAL'
            
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = self._linregress(x, prices)
        
        if abs(r_value) < 0.3:  # Correlação fraca
            return 'NEUTRAL'
        elif slope > 0.001:
            return 'UPTREND'
        elif slope < -0.001:
            return 'DOWNTREND'
        else:
            return 'NEUTRAL'

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcula força da tendência"""
        if len(prices) < 10:
            return 0.5
            
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = self._linregress(x, prices)
        
        return float(abs(r_value))

    def _calculate_volume_price_correlation(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calcula correlação entre volume e preço"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0
            
        price_changes = np.diff(prices) / prices[:-1]
        volume_changes = np.diff(volumes) / volumes[:-1]
        
        if len(price_changes) == 0 or len(volume_changes) == 0:
            return 0
            
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
        return correlation if not np.isnan(correlation) else 0

    def _linregress(self, x, y):
        """Implementação simplificada de regressão linear"""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, intercept, r_squared, 0, 0

    async def get_market_summary(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Retorna resumo executivo do mercado"""
        analysis = await self.analyze_market(symbol, data)
        
        return {
            'symbol': symbol,
            'recommendation': analysis.get('recommendation', 'HOLD'),
            'confidence': analysis.get('confidence', 0),
            'current_price': analysis.get('current_price', 0),
            'primary_trend': analysis['market_conditions'].get('primary_trend', 'NEUTRAL'),
            'risk_level': analysis['risk_metrics'].get('risk_class', 'MEDIUM'),
            'key_signals': [s['type'] for s in analysis.get('signals', [])],
            'timestamp': analysis.get('timestamp', datetime.now())
        }
