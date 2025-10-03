from pathlib import Path

from rag import DocumentRAGBot

# Запуск бота
if __name__ == "__main__":
    TELEGRAM_TOKEN = Path('tg-bot-token.txt').open().read()
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден в переменных окружения")

    bot = DocumentRAGBot(TELEGRAM_TOKEN)
    bot.run()