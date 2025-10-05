# run_bot.py - Script de inicialização rápida
import asyncio
import logging
from main import TradingBot

async def quick_start():
    """Inicialização rápida do bot para testes"""
    print("🚀 INICIALIZAÇÃO RÁPIDA DO TRADING BOT")
    print("=" * 50)
    
    # Configuração mínima de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Inicializa o bot
        bot = TradingBot('config.yaml')
        
        # Health check rápido
        print("🔍 Verificando saúde dos módulos...")
        health = await bot.health_check()
        
        if health['bot'] == 'healthy':
            print("✅ Todos os módulos estão saudáveis!")
            print("🎯 Iniciando loop principal...")
            await bot.run()
        else:
            print("❌ Problemas detectados:")
            for module, status in health.items():
                print(f"   {module}: {status}")
                
    except KeyboardInterrupt:
        print("\n🛑 Bot interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro fatal: {e}")

if __name__ == "__main__":
    asyncio.run(quick_start())
