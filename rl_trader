# rl_trader.py - Trading com Reinforcement Learning
import numpy as np
import pandas as pd
import logging
from collections import deque
import random
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


class RLTrader:
    def __init__(self, state_size: int = 10, enable_training: bool = True):
        self.state_size = state_size
        self.enable_training = enable_training
        self.logger = self._setup_logger()
        
        # Hiperparâmetros do RL
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Modelos
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Métricas
        self.episode_rewards = []
        self.training_history = []
        
        self.logger.info("🤖 RL Trader inicializado com Deep Q-Learning")

    def _setup_logger(self):
        logger = logging.getLogger('RLTrader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _build_model(self) -> Sequential:
        """Constrói a rede neural Deep Q-Learning"""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='linear')  # 3 ações: BUY, SELL, HOLD
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model

    def update_target_model(self):
        """Atualiza o modelo alvo (target network)"""
        self.target_model.set_weights(self.model.get_weights())

    def _create_state(self, market_data: Dict, portfolio: Dict) -> np.ndarray:
        """Cria o estado atual para o RL"""
        try:
            # Features do mercado
            features = [
                market_data.get('rsi', 50) / 100,  # Normalizado
                market_data.get('macd', 0),
                market_data.get('macd_histogram', 0),
                market_data.get('bb_position', 0.5),
                market_data.get('volume_ratio', 1),
                market_data.get('volatility', 0.1) * 10,
                market_data.get('trend_strength', 0.5),
                portfolio.get('cash_ratio', 0.5),
                portfolio.get('position_size', 0),
                portfolio.get('current_pnl', 0) / 100  # Normalizado
            ]
            
            # Garante que temos exatamente state_size features
            while len(features) < self.state_size:
                features.append(0.0)
            
            return np.array(features[:self.state_size]).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"❌ Erro criando estado: {e}")
            return np.zeros((1, self.state_size))

    def choose_action(self, state: np.ndarray) -> int:
        """Escolhe ação usando política ε-greedy"""
        if np.random.rand() <= self.epsilon and self.enable_training:
            # Exploração: ação aleatória
            return random.randrange(3)
        
        # Exploração: melhor ação prevista
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Armazena experiência no replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Treina o modelo com experiências anteriores"""
        if len(self.memory) < self.batch_size:
            return

        # Amostra batch aleatório
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0][0] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        
        # Previsões atuais e futuras
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Atualiza Q-values
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(next_q[i])
            
            current_q[i][action] = target
        
        # Treina o modelo
        self.model.fit(states, current_q, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Decai epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, action: int, pnl: float, market_conditions: Dict) -> float:
        """Calcula recompensa baseada no resultado do trade"""
        reward = 0.0
        
        # Recompensa/Punição baseada no P&L
        reward += pnl * 10  # Escala o P&L
        
        # Recompensas adicionais baseadas em condições de mercado
        if action == 0:  # BUY
            if market_conditions.get('trend', 'NEUTRAL') == 'UPTREND':
                reward += 0.1
            elif market_conditions.get('rsi', 50) < 30:  # RSI oversold
                reward += 0.2
                
        elif action == 1:  # SELL
            if market_conditions.get('trend', 'NEUTRAL') == 'DOWNTREND':
                reward += 0.1
            elif market_conditions.get('rsi', 50) > 70:  # RSI overbought
                reward += 0.2
                
        elif action == 2:  # HOLD
            if market_conditions.get('volatility', 0) > 0.05:  # Alta volatilidade
                reward += 0.1
        
        # Punição por ações muito frequentes
        reward -= 0.01  # Pequena punição por qualquer ação
        
        return reward

    async def get_trading_signal(self, market_data: Dict, portfolio: Dict, 
                               symbol: str) -> Dict:
        """Gera sinal de trading usando RL"""
        try:
            # Cria estado atual
            state = self._create_state(market_data, portfolio)
            
            # Escolhe ação
            action = self.choose_action(state)
            
            # Mapeia ação para sinal de trading
            action_map = {
                0: {'action': 'buy', 'type': 'RL_BUY', 'strength': 0.8},
                1: {'action': 'sell', 'type': 'RL_SELL', 'strength': 0.8},
                2: {'action': 'hold', 'type': 'RL_HOLD', 'strength': 0.5}
            }
            
            signal = action_map[action]
            signal['rl_state'] = state.tolist()
            signal['epsilon'] = self.epsilon
            
            self.logger.info(f"🎯 Sinal RL: {symbol} {signal['action'].upper()} "
                           f"(ε={self.epsilon:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Erro gerando sinal RL: {e}")
            return {'action': 'hold', 'type': 'RL_ERROR', 'strength': 0.1}

    def update_with_result(self, state: np.ndarray, action: int, 
                         next_state: np.ndarray, reward: float, done: bool = False):
        """Atualiza o modelo com resultado do trade"""
        if not self.enable_training:
            return
            
        self.remember(state, action, reward, next_state, done)
        self.replay()
        
        # Atualiza target model periodicamente
        if len(self.memory) % 100 == 0:
            self.update_target_model()

    def save_model(self, filepath: str = 'rl_trader_model.h5'):
        """Salva o modelo treinado"""
        try:
            self.model.save(filepath)
            self.logger.info(f"💾 Modelo RL salvo: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Erro salvando modelo: {e}")

    def load_model(self, filepath: str = 'rl_trader_model.h5'):
        """Carrega modelo treinado"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.epsilon = self.epsilon_min  # Minimiza exploração
            self.logger.info(f"📂 Modelo RL carregado: {filepath}")
        except Exception as e:
            self.logger.warning(f"⚠️  Não foi possível carregar modelo: {e}")

    def get_training_metrics(self) -> Dict:
        """Retorna métricas de treinamento"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'episode_count': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'model_updated': len(self.memory) >= self.batch_size
        }


# Integração com o Risk Manager existente
class RLEnhancedRiskManager(RiskManager):
    def __init__(self, config):
        super().__init__(config)
        self.rl_trader = RLTrader(enable_training=True)
        self.portfolio_state = {}
        
    async def execute_trade(self, market, signal, trader):
        """Executa trade com suporte do RL"""
        try:
            # Atualiza estado do portfólio
            await self._update_portfolio_state(trader)
            
            # Obtém sinal do RL
            rl_signal = await self.rl_trader.get_trading_signal(
                market_data=market,
                portfolio=self.portfolio_state,
                symbol=market['symbol']
            )
            
            # Combina sinais (RL tem prioridade)
            if rl_signal['action'] != 'hold' and rl_signal['strength'] > 0.7:
                final_signal = rl_signal
                self.logger.info(f"🤖 Usando sinal RL: {final_signal['action']}")
            else:
                final_signal = signal
                self.logger.info(f"📊 Usando sinal tradicional: {final_signal['action']}")
            
            # Executa trade normalmente
            return await super().execute_trade(market, final_signal, trader)
            
        except Exception as e:
            self.logger.error(f"❌ Erro no RL Enhanced Risk Manager: {e}")
            return await super().execute_trade(market, signal, trader)
    
    async def _update_portfolio_state(self, trader):
        """Atualiza estado do portfólio para o RL"""
        try:
            balance = await trader.get_account_balance()
            total_value = sum(balance.values())
            
            self.portfolio_state = {
                'cash_ratio': balance.get('USDT', 0) / max(total_value, 1),
                'position_size': len(self.open_positions),
                'current_pnl': self.daily_pnl,
                'total_trades': self.total_trades
            }
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando estado do portfólio: {e}")
