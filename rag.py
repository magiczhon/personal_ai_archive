import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict

import telebot
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from telebot.util import quick_markup

from db import add_document, document_exists, get_current_week_documents

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAG:
    def __init__(self):
        logger.info('Initializing ML models')
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # Используем локальную модель для суммаризации
        self.llm = ChatOllama(
            model="gemma3:4b",
            temperature=0,
            validate_model_on_init=True
            # openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.vectorstore = None
        self.load_vectorstore()
        logger.info('Initializing complete')

    def load_vectorstore(self):
        """Загрузка векторного хранилища"""
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def get_retriever(self, user_id: int, k: int = 5):
        """Создание retriever с фильтром по user_id
        Тут есть другие примеры:
        https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.as_retriever
        """

        return self.vectorstore.as_retriever(
            # search_type="similarity_score_threshold",
            search_type="mmr",
            search_kwargs={
                "k": k,
                "filter": {"user_id": user_id},
                #"score_threshold": 0.7
            },
        )

    def query_to_rag(self, question: str, user_id: int) -> Dict:
        logger.info('query_to_rag')
        """Основной метод для запросов к RAG системе"""
        # Получаем retriever для конкретного пользователя
        retriever = self.get_retriever(user_id)
        docs = retriever.invoke(question)

        # Дедупликация по содержанию
        unique_contents = set()
        unique_docs = []

        for doc in docs:
            if doc.page_content not in unique_contents:
                unique_contents.add(doc.page_content)
                unique_docs.append(doc)

        context = "\n\n".join(doc.page_content for doc in unique_docs)

        prompt_template = f"""Используй приведенные ниже фрагменты контекста, чтобы ответить на вопрос пользователя. 
        Если в контексте нет информации для ответа, вежливо сообщи, что не можешь ответить на основе доступных данных.

        Контекст:
        {context}

        Вопрос: {question}
        Тщательный ответ:
"""
        ai_msg = self.llm.invoke(prompt_template)
        return {"answer": ai_msg.content, "documents": unique_docs}

    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Добавление новых документов в векторное хранилище"""
        logger.info('add_documents_to_vectorstore')
        # Разбиваем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        # Добавляем в хранилище
        self.vectorstore.add_documents(chunks)


class DocumentRAGBot(RAG):
    # Сколько секунд заметка ждёт подтверждения, прежде чем черновик будет удалён
    PENDING_NOTE_TTL = 60 * 60

    def __init__(self, telegram_token: str):
        super().__init__()
        self.bot = telebot.TeleBot(telegram_token)
        # Черновики заметок, ожидающие подтверждения: note_id -> данные заметки.
        # Раньше текст заметки жил в замыкании обработчика кнопки, и при каждой
        # заметке регистрировался новый обработчик. Срабатывал всегда первый,
        # поэтому подтверждение второй заметки сохраняло текст первой.
        self.pending_notes: Dict[str, Dict] = {}
        self.pending_lock = threading.Lock()
        self.setup_handlers()

    def setup_handlers(self):
        """Настройка обработчиков команд"""

        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            welcome_text = """
🤖 *Вас приветствуует персональный ассистент по цифровым находкам. Добро пожаловать!*

*Я могу:*
🌐 Парсить веб-сайты и ваши заметки
📨 Сохранять пересланные посты из каналов и чатов
🔍 Выполнять семантический поиск по документам
📊 Создавать суммаризации
📅 Предоставлять отчеты за неделю

*Под капотом*
Эмбеддинги: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
Языковая модель: [gemma3:4b](https://ollama.com/library/gemma3:4b)

*Доступные команды:*
/start - Начать работу
/help - Помощь
/search - Поиск по документам
/weekly\\_report - Отчет по находкам за неделю
"""
            self.bot.reply_to(message, welcome_text, parse_mode='Markdown', disable_web_page_preview=True)

        @self.bot.message_handler(commands=['search'])
        def handle_search(message):
            msg = self.bot.reply_to(message, "Введите ваш поисковый запрос:")
            self.bot.register_next_step_handler(msg, self.process_search)

        @self.bot.message_handler(commands=['weekly_report'])
        def handle_weekly_report(message):
            self.generate_weekly_report(message)

        @self.bot.message_handler(content_types=['text', 'document', 'photo', 'video', 'animation', 'audio'])
        def handle_message(message):
            # Пересланный пост сохраняем сразу: пересылка уже и есть явное намерение сохранить
            if is_forwarded(message):
                self.process_forward(message)
            elif message.text and (message.text.startswith('http://') or message.text.startswith('https://')):
                self.process_url(message)
            elif message.document:
                self.bot.reply_to(message, "Обработчик документов находится на стадии разработки")
                # self.process_document(message)
            elif message.text:
                self.process_text(message)
            else:
                self.bot.reply_to(message, "Пока я умею сохранять только текст, ссылки и пересланные посты")

        # Один обработчик кнопок на всех: какую заметку подтверждают, узнаём по note_id в callback_data
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith(('note_confirm:', 'note_cancel:')))
        def handle_note_button(call):
            action, note_id = call.data.split(':', 1)
            with self.pending_lock:
                note = self.pending_notes.get(note_id)
                # Нажать кнопку может только автор заметки
                if note is not None and note['user_id'] == call.from_user.id:
                    self.pending_notes.pop(note_id)
                elif note is not None:
                    self.bot.answer_callback_query(call.id, "Это не ваша заметка")
                    return

            self.bot.answer_callback_query(call.id)
            chat_id, message_id = call.message.chat.id, call.message.id

            if note is None:
                self.bot.edit_message_text("Заметка устарела или уже обработана. Отправьте её ещё раз.",
                                           chat_id=chat_id, message_id=message_id)
                return

            if action == 'note_cancel':
                self.bot.edit_message_text("Операция добавления заметки отменена",
                                           chat_id=chat_id, message_id=message_id)
                return

            self.bot.send_chat_action(chat_id, 'typing')
            self.bot.edit_message_text("Идет обработка заметки...", chat_id=chat_id, message_id=message_id)
            try:
                document = Document(page_content=note['text'], metadata={'source': note['source']})
                summary = self.save_to_archive(document, note['user_id'], note['message'])
                self.bot.edit_message_text(f"✅ Заметка успешно добавлена!\n\n📝 Краткое содержание:\n{summary}",
                                           chat_id=chat_id, message_id=message_id)
            except Exception as e:
                logger.error(f"Error processing note: {e}")
                self.bot.edit_message_text(f"Ошибка при обработке заметки: {str(e)}",
                                           chat_id=chat_id, message_id=message_id)

    def save_to_archive(self, document: Document, user_id: int, message: telebot.types.Message) -> str:
        """Общий путь сохранения: суммаризация, запись в SQLite и в векторное хранилище.
        Возвращает краткое содержание."""
        document.metadata['user_id'] = user_id
        document.metadata['added_date'] = datetime.now().isoformat()
        document.metadata['summary'] = self.summarize_document(document, message)

        add_document(
            user_id=user_id,
            added_date=datetime.fromisoformat(document.metadata['added_date']),
            source=document.metadata.get('source', ''),
            title=document.metadata.get('title', ''),
            all_content=document.page_content,
            summarize=document.metadata['summary'],
        )
        self.add_documents_to_vectorstore([document])
        return document.metadata['summary']

    def process_text(self, message: telebot.types.Message):
        """Текстовая заметка: сохраняем после подтверждения кнопкой"""
        self.drop_expired_notes()
        note_id = uuid.uuid4().hex[:16]  # callback_data ограничена 64 байтами
        with self.pending_lock:
            self.pending_notes[note_id] = {
                'text': message.text,
                'user_id': message.from_user.id,
                'source': message_link(message.chat, message.message_id),
                'message': message,
                'created': time.time(),
            }

        markup = quick_markup({
            "✅ Да, добавить": {'callback_data': f"note_confirm:{note_id}"},
            '❌ Нет, отменить': {'callback_data': f"note_cancel:{note_id}"}
        }, row_width=2)
        self.bot.reply_to(message, "Вы действительно хотите добавить данную заметку в базу знаний?", reply_markup=markup)

    def drop_expired_notes(self):
        """Удаляем черновики, которые так и не подтвердили, чтобы они не копились в памяти"""
        deadline = time.time() - self.PENDING_NOTE_TTL
        with self.pending_lock:
            for note_id in [k for k, v in self.pending_notes.items() if v['created'] < deadline]:
                del self.pending_notes[note_id]

    def process_forward(self, message: telebot.types.Message):
        """Пересланный пост из канала, чата или от пользователя"""
        text = message.text or message.caption or ''
        if not text.strip():
            self.bot.reply_to(message, "В пересланном сообщении нет текста — пока я умею сохранять только текст")
            return

        source, title = forward_source(message)
        user_id = message.from_user.id

        if source and document_exists(user_id, source):
            self.bot.reply_to(message, "Этот пост уже есть в вашем архиве")
            return

        self.bot.send_chat_action(message.chat.id, 'typing')
        process_message = self.bot.reply_to(message, "Идет обработка поста...")
        try:
            document = Document(page_content=text, metadata={
                'source': source or message_link(message.chat, message.message_id),
                'title': title,
            })
            summary = self.save_to_archive(document, user_id, message)
            self.bot.reply_to(message, f"✅ Пост сохранён: {title}\n\n📝 Краткое содержание:\n{summary}",
                              disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Error processing forward: {e}")
            self.bot.reply_to(message, f"Ошибка при обработке поста: {str(e)}")
        finally:
            self.bot.delete_message(chat_id=message.chat.id, message_id=process_message.id)

    def process_url(self, message: telebot.types.Message):
        """Обработка URL"""
        url = message.text
        logger.info(f'Process url {url}')
        self.bot.send_chat_action(message.chat.id, 'typing')
        process_message = self.bot.send_message(chat_id=message.chat.id, reply_to_message_id=message.id,
                                                text="Идет обработка контента...")
        try:
            # Загрузка и парсинг веб-страницы
            loader = WebBaseLoader(url)
            documents = loader.load()
            if documents:
                assert 1 == len(documents)
                document = documents[0]

                document.metadata['user_id'] = message.from_user.id
                document.metadata['added_date'] = datetime.now().isoformat()

                document.metadata['summary'] = self.summarize_document(document, message)

                if not document.page_content.replace('\n', '').replace('\t', '').replace(' ', ''):
                    self.bot.reply_to(message, f"Не удалось спарсить сайт (пустой page_content) {url}")
                    return

                # add doc to db
                add_document(
                    user_id=document.metadata.get('user_id', ''),
                    added_date=datetime.fromisoformat(document.metadata.get('added_date', '')),
                    source=document.metadata.get('source', ''),
                    title=document.metadata.get('title', ''),
                    all_content=document.page_content,
                    summarize=document.metadata.get('summary', '')
                )

                # TODO: make semantic search
                self.add_documents_to_vectorstore([document])

                response = f"✅ Сайт успешно добавлен!\n\n📝 Краткое содержание:\n{document.metadata['summary']}"
                self.bot.reply_to(message, response)
            else:
                self.bot.reply_to(message, "Не удалось загрузить контент с сайта")
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            self.bot.reply_to(message, f"Ошибка при обработке сайта: {str(e)}")
        finally:
            # удаляем сообщение о процессинге
            self.bot.delete_message(chat_id=message.chat.id, message_id=process_message.id)

    def summarize_document(self, document: Document, message: telebot.types.Message) -> str:
        """Суммаризация документов"""
        try:
            # Русские промпты для суммаризации
            map_prompt = """
            Сделай качественное суммаризирование следующего текста:
            {text}
            КРАТКОЕ СОДЕРЖАНИЕ:
            """
            map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

            combine_prompt = """
            Сделай качественное суммаризирование следующего текста, сохраняя ключевые идеи. Суммаризируй не более чем в 4 предложения).
            Также не нужно вставлять вводные фразы перед суммаризацией. Сразу генерируй суммаризированный текст:
            {text}
            ПОЛНОЕ КРАТКОЕ СОДЕРЖАНИЕ:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

            # Загрузка цепи с русскими промптами
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt_template,
                combine_prompt=combine_prompt_template,
                # verbose=True
            )
            summary = chain.invoke({"input_documents": [document]}, return_only_outputs=True)
            return summary['output_text']

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            self.bot.reply_to(message, "Не удалось суммаризировать документ")
            return 'Не удалось суммаризировать текст'

    def generate_weekly_report(self, message: telebot.types.Message):
        """Генерация отчета за неделю"""
        try:
            # Фильтруем документы за последнюю неделю
            all_docs_current_week = get_current_week_documents(user_id=message.from_user.id)
            docs_current_week_set = set()
            docs_current_week = []
            for doc in all_docs_current_week:
                if doc.source in docs_current_week_set:
                    continue
                docs_current_week_set.add(doc.source)
                docs_current_week.append(doc)

            if not docs_current_week:
                self.bot.reply_to(message, "За последнюю неделю документы не добавлялись")
                return

            # Выводим компактный список
            texts = format_documents_compact(docs_current_week)
            messages = texts.split('📄')
            idx_report_message = self.bot.reply_to(message, text=messages[0], parse_mode="Markdown", disable_web_page_preview=True)
            for i, text in enumerate(messages[1:]):
                if len(text) > 4096:
                    text = text[:4000] + '...'
                try:
                    self.bot.reply_to(idx_report_message, text=text, parse_mode="Markdown", disable_web_page_preview=True)
                except Exception as e:
                    logger.warning(f'Dont perse in markdown mode message:\n{text}')
                    self.bot.reply_to(idx_report_message, text=text, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            self.bot.reply_to(message, f"Ошибка при генерации отчета: {str(e)}")

    def process_search(self, message: telebot.types.Message):
        """Обработка поискового запроса"""
        try:
            query = message.text
            self.bot.send_chat_action(message.chat.id, 'typing')

            # Поиск релевантных документов
            res = self.query_to_rag(question=query, user_id=message.from_user.id)
            rag_answer, docs = res['answer'], res['documents']
            if not docs:
                self.bot.reply_to(message, "По вашему запросу ничего не найдено")
                return

            # Формируем ответ
            response = f"🔍 Результаты поиска по запросу: '{query}'\n\n➡️ Ответ ЛЛМ: {rag_answer}\n"

            source_line = 'ℹ️Источники:\n'
            set_of_sources = set()
            idx_source = 1
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Неизвестный источник')
                if source in set_of_sources:
                    continue
                source_line += f"[{idx_source}]📄 - {source}\n"
                set_of_sources.add(source)
                idx_source += 1

            response += source_line
            self.bot.reply_to(message, response)

        except Exception as e:
            logger.error(f"Error in search: {e}")
            self.bot.reply_to(message, f"Ошибка при поиске: {str(e)}")

    def process_document(self, message):
        """Обработка документа"""
        logger.info('Run process_document')
        self.bot.send_chat_action(message.chat.id, 'typing')
        process_message = self.bot.send_message(chat_id=message.chat.id, reply_to_message_id=message.id, text="Идет обработка документа...")

        try:
            file_info = self.bot.get_file(message.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)

            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_info.file_path)[1]) as tmp_file:
                tmp_file.write(downloaded_file)
                tmp_file_path = tmp_file.name

            # Определяем тип документа и загружаем
            if message.document.mime_type == 'application/pdf' or tmp_file_path.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path)

            documents = loader.load()

            if documents:
                assert 1 == len(documents)
                document = documents[0]
                # Добавляем метаданные

                document.metadata['user_id'] = message.from_user.id
                document.metadata['added_date'] = datetime.now().isoformat()

                document.metadata['summary'] = self.summarize_document(document, message)

                # add doc to db
                add_document(
                    user_id=document.metadata.get('user_id', ''),
                    added_date=datetime.fromisoformat(document.metadata.get('added_date', '')),
                    source=document.metadata.get('source', ''),
                    title=document.metadata.get('title', ''),
                    all_content=document.page_content,
                    summarize=document.metadata.get('summary', '')
                )
                # Без этого документ попадал только в SQLite и не находился поиском
                self.add_documents_to_vectorstore([document])

                response = f"✅ Документ успешно добавлен!\n\n📝 Краткое содержание:\n{document.metadata['summary']}"
                self.bot.reply_to(message, response)
            else:
                self.bot.reply_to(message, "Не удалось прочитать документ")

            # Удаляем временный файл
            os.unlink(tmp_file_path)

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            self.bot.reply_to(message, f"Ошибка при обработке документа: {str(e)}")
        finally:
            # удаляем сообщение о процессинге
            self.bot.delete_message(chat_id=message.chat.id, message_id=process_message.id)

    def run(self):
        """Запуск бота"""
        logger.info("Бот запущен...")
        self.bot.infinity_polling()


def format_documents_compact(documents):
    if not documents:
        return "📂 У вас пока нет документов"
    week_ago = datetime.now() - timedelta(days=7)
    # Формируем отчет
    lines = [f"📅 Отчет за неделю (с {week_ago.strftime('%d.%m.%Y')})", f"📊 Добавлено находок: {len(documents)}\n\n"]

    for i, doc in enumerate(documents, 1):
        lines.append(
            f"📄/{i}/ - *{doc.title}*\n"
            f"📅 Дата: {doc.added_date}\n"
            f"🏷️ Источник: {doc.source}\n"
            f"📝 Краткое содержание: {doc.summarize}\n"
        )
    text = "\n".join(lines)
    return text



def is_forwarded(message: telebot.types.Message) -> bool:
    """Переслано ли сообщение (новое поле forward_origin и старые поля Bot API)"""
    return bool(getattr(message, 'forward_origin', None)
                or getattr(message, 'forward_from_chat', None)
                or getattr(message, 'forward_from', None)
                or getattr(message, 'forward_sender_name', None))


def message_link(chat, message_id: int) -> str:
    """Ссылка на сообщение. Для публичных каналов и чатов — t.me/<username>/<id>,
    для приватных супергрупп и каналов — t.me/c/<id>/<id>. У личной переписки
    с ботом публичной ссылки нет, поэтому возвращаем уникальный идентификатор:
    по source дедуплицируется недельный отчёт."""
    if chat is None or message_id is None:
        return ''
    if getattr(chat, 'username', None):
        return f'https://t.me/{chat.username}/{message_id}'
    chat_id = str(chat.id)
    if chat_id.startswith('-100'):
        return f'https://t.me/c/{chat_id[4:]}/{message_id}'
    return f'telegram:{chat_id}/{message_id}'


def forward_source(message: telebot.types.Message):
    """Возвращает (ссылка на оригинал или '', название источника) для пересланного сообщения"""
    origin = getattr(message, 'forward_origin', None)
    if origin is not None:
        if origin.type == 'channel':
            return message_link(origin.chat, origin.message_id), origin.chat.title or 'Канал'
        if origin.type == 'chat':
            return '', origin.sender_chat.title or 'Чат'
        if origin.type == 'user':
            return '', full_name(origin.sender_user)
        if origin.type == 'hidden_user':
            return '', origin.sender_user_name or 'Скрытый пользователь'

    # Старые поля Bot API (pyTelegramBotAPI < 4.15)
    chat = getattr(message, 'forward_from_chat', None)
    if chat is not None:
        return message_link(chat, getattr(message, 'forward_from_message_id', None)), chat.title or 'Канал'
    if getattr(message, 'forward_from', None):
        return '', full_name(message.forward_from)
    return '', getattr(message, 'forward_sender_name', None) or 'Пересланное сообщение'


def full_name(user) -> str:
    return ' '.join(filter(None, [user.first_name, user.last_name])) or user.username or 'Пользователь'
