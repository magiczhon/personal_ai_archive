from pathlib import Path

from rag import DocumentRAGBot

# Запуск бота
if __name__ == "__main__":
    # strip(): перевод строки в конце файла ломает авторизацию бота
    TELEGRAM_TOKEN = Path('tg-bot-token.txt').read_text().strip()
    if not TELEGRAM_TOKEN:
        raise ValueError("Токен бота не найден: файл tg-bot-token.txt пуст")

    bot = DocumentRAGBot(TELEGRAM_TOKEN)
    bot.run()