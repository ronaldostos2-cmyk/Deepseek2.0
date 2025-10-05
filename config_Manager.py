# config_manager.py (VERSÃO COMPLETA E CORRIGIDA)
import yaml
import logging
import os
from typing import Dict, Any, Optional


class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.logger = self._setup_logger()
        self.config = self._load_config()

    def _setup_logger(self) -> logging.Logger:
        """Configura sistema de logging"""
        logger = logging.getLogger('ConfigManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_config(self) -> Dict[str, Any]:
        """Carrega e valida configuração do arquivo YAML"""
        try:
            # Verifica se o arquivo existe
            if not os.path.exists(self.config_path):
                self.logger.warning(f"⚠️  Arquivo de configuração não encontrado: {self.config_path}")
                self.logger.info("📝 Usando configuração padrão...")
                return self._get_default_config()

            # Carrega configuração do arquivo
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            # Valida estrutura
            self._validate_config(config)
            
            self.logger.info("✅ Configuração carregada com sucesso")
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"❌ Erro de sintaxe no arquivo YAML: {e}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Erro carregando configuração: {e}")
            return self._get_default_config()

    def _validate_config(self, config: Dict[str, Any]):
        """Valida estrutura da configuração"""
        try:
            # Verifica seções obrigatórias
            required_sections = ['risk', 'trading', 'exchange']
            
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Seção obrigatória faltando: {section}")

            # Valida configurações de risco
            risk_config = config['risk']
            required_risk_fields = ['risk_per_trade', 'max_daily_trades', 'daily_loss_limit']
            for field in required_risk_fields:
                if field not in risk_config:
                    raise ValueError(f"Campo de risco obrigatório faltando: {field}")

            # Valida risk_per_trade
            risk_per_trade = risk_config['risk_per_trade']
            if not (0 < risk_per_trade <= 1):
                raise ValueError(f"risk_per_trade deve estar entre 0 e 1: {risk_per_trade}")

            # Valida max_daily_trades
            max_daily_trades = risk_config['max_daily_trades']
            if not (1 <= max_daily_trades <= 100):
                raise ValueError(f"max_daily_trades deve estar entre 1 e 100: {max_daily_trades}")

            # Valida configurações de trading
            trading_config = config['trading']
            if 'enabled_pairs' not in trading_config:
                raise ValueError("enabled_pairs é obrigatório na seção trading")

            enabled_pairs = trading_config['enabled_pairs']
            if not enabled_pairs or not isinstance(enabled_pairs, list):
                raise ValueError("enabled_pairs deve ser uma lista não vazia")

            # Valida exchange
            exchange_config = config['exchange']
            if 'name' not in exchange_config:
                raise ValueError("name é obrigatório na seção exchange")

            self.logger.info("✅ Configuração validada com sucesso")

        except Exception as e:
            self.logger.error(f"❌ Erro validando configuração: {e}")
            raise

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão de fallback"""
        self.logger.info("🔄 Carregando configuração padrão...")
        
        return {
            'risk': {
                'risk_per_trade': 0.02,
                'max_daily_trades': 10,
                'daily_loss_limit': -100,
                'cooldown_period': 30,
                'max_position_value': 50,
                'min_position_value': 20
            },
            'trading': {
                'enabled_pairs': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
                'strategy': 'mean_reversion',
                'analysis_interval': 60,
                'max_open_positions': 3
            },
            'exchange': {
                'name': 'binance',
                'api_key': '',
                'secret': '',
                'sandbox': True,
                'order_timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_bot.log'
            }
        }

    def get_risk_config(self) -> Dict[str, Any]:
        """Retorna configurações de risco"""
        return self.config.get('risk', {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Retorna configurações de trading"""
        return self.config.get('trading', {})

    def get_exchange_config(self) -> Dict[str, Any]:
        """Retorna configurações da exchange"""
        return self.config.get('exchange', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Retorna configurações de logging"""
        return self.config.get('logging', {})

    def update_config(self, new_config: Dict[str, Any]):
        """Atualiza configuração dinamicamente"""
        try:
            # Valida nova configuração
            self._validate_config(new_config)
            
            # Atualiza configuração
            self.config.update(new_config)
            
            # Salva no arquivo
            self._save_config()
            
            self.logger.info("✅ Configuração atualizada com sucesso")
            
        except Exception as e:
            self.logger.error(f"❌ Erro atualizando configuração: {e}")
            raise

    def _save_config(self):
        """Salva configuração no arquivo"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"💾 Configuração salva em: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro salvando configuração: {e}")

    def reload_config(self):
        """Recarrega configuração do arquivo"""
        try:
            self.config = self._load_config()
            self.logger.info("🔄 Configuração recarregada")
        except Exception as e:
            self.logger.error(f"❌ Erro recarregando configuração: {e}")

    def get_enabled_pairs(self) -> list:
        """Retorna lista de pares habilitados"""
        return self.get_trading_config().get('enabled_pairs', [])

    def get_risk_per_trade(self) -> float:
        """Retorna percentual de risco por trade"""
        return self.get_risk_config().get('risk_per_trade', 0.02)

    def get_max_daily_trades(self) -> int:
        """Retorna máximo de trades diários"""
        return self.get_risk_config().get('max_daily_trades', 10)

    def get_daily_loss_limit(self) -> float:
        """Retorna limite de perda diária"""
        return self.get_risk_config().get('daily_loss_limit', -100)

    def is_sandbox_mode(self) -> bool:
        """Verifica se está em modo sandbox"""
        return self.get_exchange_config().get('sandbox', True)

    def get_exchange_name(self) -> str:
        """Retorna nome da exchange"""
        return self.get_exchange_config().get('name', 'binance')

    def show_config_summary(self):
        """Mostra resumo da configuração"""
        try:
            self.logger.info("📋 RESUMO DA CONFIGURAÇÃO:")
            
            # Risco
            risk = self.get_risk_config()
            self.logger.info(f"   🎯 RISCO:")
            self.logger.info(f"      • Risk per Trade: {risk.get('risk_per_trade', 0) * 100}%")
            self.logger.info(f"      • Max Daily Trades: {risk.get('max_daily_trades', 0)}")
            self.logger.info(f"      • Daily Loss Limit: ${risk.get('daily_loss_limit', 0)}")
            self.logger.info(f"      • Cooldown: {risk.get('cooldown_period', 0)}s")
            
            # Trading
            trading = self.get_trading_config()
            self.logger.info(f"   📈 TRADING:")
            self.logger.info(f"      • Estratégia: {trading.get('strategy', 'N/A')}")
            self.logger.info(f"      • Intervalo: {trading.get('analysis_interval', 0)}s")
            self.logger.info(f"      • Pares: {', '.join(trading.get('enabled_pairs', []))}")
            
            # Exchange
            exchange = self.get_exchange_config()
            self.logger.info(f"   💱 EXCHANGE:")
            self.logger.info(f"      • Nome: {exchange.get('name', 'N/A')}")
            self.logger.info(f"      • Sandbox: {exchange.get('sandbox', False)}")
            self.logger.info(f"      • Timeout: {exchange.get('order_timeout', 0)}s")
            
        except Exception as e:
            self.logger.error(f"❌ Erro mostrando resumo da configuração: {e}")

    def validate_api_credentials(self) -> bool:
        """Valida se as credenciais da API estão configuradas"""
        try:
            exchange_config = self.get_exchange_config()
            
            # Em modo sandbox, as credenciais podem ser vazias
            if self.is_sandbox_mode():
                self.logger.info("🔧 Modo sandbox - credenciais não são obrigatórias")
                return True
            
            # Em modo real, verifica credenciais
            api_key = exchange_config.get('api_key', '')
            secret = exchange_config.get('secret', '')
            
            if not api_key or not secret:
                self.logger.error("❌ Credenciais da API não configuradas para modo real")
                return False
            
            if api_key.startswith('${') or secret.startswith('${'):
                self.logger.error("❌ Variáveis de ambiente não resolvidas nas credenciais")
                return False
            
            self.logger.info("✅ Credenciais da API validadas")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro validando credenciais: {e}")
            return False

    def create_sample_config(self):
        """Cria arquivo de configuração de exemplo"""
        try:
            sample_config = {
                'risk': {
                    'risk_per_trade': 0.02,
                    'max_daily_trades': 10,
                    'daily_loss_limit': -100,
                    'cooldown_period': 30,
                    'max_position_value': 50,
                    'min_position_value': 20
                },
                'trading': {
                    'enabled_pairs': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
                    'strategy': 'mean_reversion',
                    'analysis_interval': 60,
                    'max_open_positions': 3
                },
                'exchange': {
                    'name': 'binance',
                    'api_key': 'SUA_API_KEY_AQUI',
                    'secret': 'SEU_SECRET_AQUI',
                    'sandbox': True,
                    'order_timeout': 30
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'trading_bot.log'
                }
            }
            
            with open('config_sample.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(sample_config, file, default_flow_style=False, allow_unicode=True)
            
            self.logger.info("📝 Arquivo de exemplo criado: config_sample.yaml")
            self.logger.info("💡 Copie para config.yaml e configure com suas credenciais")
            
        except Exception as e:
            self.logger.error(f"❌ Erro criando arquivo de exemplo: {e}")
