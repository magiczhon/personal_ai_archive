from typing import List, Any, Sequence
from typing import Optional
from sqlalchemy import create_engine, select, and_, Row, RowMapping
from sqlalchemy import String, Text, Date
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from datetime import date, timedelta



class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(nullable=False)
    added_date: Mapped[date] = mapped_column(Date, nullable=False)
    source: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    all_content: Mapped[str] = mapped_column(Text)
    summarize: Mapped[str] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"Doc (id={self.id!r}, user_id={self.user_id!r}, date={self.added_date!r})"


# Создаем подключение к базе данных
# Замените 'sqlite:///documents.sqlite3' на вашу строку подключения
engine = create_engine('sqlite:///documents.sqlite3', echo=True)

# Создаем таблицу
Base.metadata.create_all(engine)

# Для работы с сессиями
Session = sessionmaker(bind=engine)


def add_document(user_id, added_date, source, title, all_content, summarize):
    session = Session()
    try:
        document = Document(
            user_id=user_id,
            added_date=added_date,
            source=source,
            title=title,
            all_content=all_content,
            summarize=summarize,
        )
        session.add(document)
        session.commit()
        return document
    finally:
        session.close()


def get_current_week_documents(user_id: Optional[int] = None) -> Sequence[Row[Any] | RowMapping | Any]:
    """
    Получение документов за текущую неделю
    user_id: если указан - фильтр по пользователю, иначе все документы
    """
    session = Session()
    # Вычисляем начало и конец текущей недели
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Строим запрос
    stmt = select(Document).where(
        and_(
            Document.added_date >= week_ago,
            Document.user_id == user_id
        )
    )

    # Сортируем по дате (новые сначала)
    stmt = stmt.order_by(Document.added_date.desc())

    return session.execute(stmt).scalars().all()