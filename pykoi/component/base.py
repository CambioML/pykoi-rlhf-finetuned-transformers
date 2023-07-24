"""Base classes for components."""
import uuid
from typing import Callable, List, Optional, Union

from pykoi.component.chatbot_database_factory import ChatbotDatabaseFactory
from pykoi.component.constants import FeedbackType
from pykoi.db.qa_database import QuestionAnswerDatabase
from pykoi.db.ranking_database import RankingDatabase
from pykoi.llm.abs_llm import AbsLlm


class DataSource:
    """
    DataSource class is used to fetch data from a source using a provided function.

    Attributes:
        id (str): The unique identifier for the data source.
        fetch_func (Callable): The function to fetch data from the source.
    """

    def __init__(self, id: str, fetch_func: Callable):
        """
        Initialize a new instance of DataSource.

        Args:
            id (str): The unique identifier for the data source.
            fetch_func (Callable): The function to fetch data from the source.
        """
        self.id = id
        self.fetch_func = fetch_func


class Component:
    """
    Component class is the base class for all components.

    Attributes:
        id (str): The unique identifier for the component.
        data_source (DataSource): The data source for the component.
        svelte_component (str): The name of the Svelte component.
        props (Dict[str, Any]): Additional properties for the component.
    """

    def __init__(self, fetch_func: Optional[Callable], svelte_component: str, **kwargs):
        """
        Initialize a new instance of Component.

        Args:
            fetch_func (Callable, optional): The function to fetch data for the component.
            svelte_component (str): The name of the Svelte component.
            kwargs: Additional properties for the component.
        """
        self.id = str(uuid.uuid4())  # Generate a unique ID
        self.data_source = DataSource(self.id, fetch_func) if fetch_func else None
        self.svelte_component = svelte_component
        self.props = kwargs


class Dropdown(Component):
    """
    Dropdown class represents a dropdown component.

    Attributes:
        value_column (str): The column to use for the dropdown values.
    """

    def __init__(self, fetch_func: Callable, value_column: List[str], **kwargs):
        """
        Initialize a new instance of Dropdown.

        Args:
            fetch_func (Callable): The function to fetch data for the dropdown.
            value_column (List[str]): The column to use for the dropdown values.
            kwargs: Additional properties for the dropdown.
        """
        super().__init__(fetch_func, "Dropdown", **kwargs)
        self.value_column = value_column


class Chatbot(Component):
    """
    Chatbot class represents a chatbot component.

    Attributes:
        model (str): The model to use for the chatbot.
        database (str): The database to use for the chatbot.
    """

    def __init__(self, model: AbsLlm, **kwargs):
        """
        Initialize a new instance of Chatbot.

        Args:
            model (AbsLlm): The model to use for the chatbot.
            kwargs: Additional properties for the chatbot.
        """
        super().__init__(None, "Chatbot", **kwargs)
        self.model = model
        self.database = ChatbotDatabaseFactory.create(
            feedback=kwargs.get("feedback", "vote")
        )


class Dashboard(Component):
    """
    Dashboard class represents a dashboard component.

    Attributes:
        database (str): The database to use for the dashboard.
    """

    def __init__(self, database: QuestionAnswerDatabase, **kwargs):
        """
        Initialize a new instance of Dashboard.

        Args:
            database (QuestionAnswerDatabase): The database to use for the dashboard.
            kwargs: Additional properties for the dashboard.
        """
        super().__init__(None, "Feedback", **kwargs)
        self.database = database
