import uuid


class DataSource:
    def __init__(self, id, fetch_func):
        self.id = id
        self.fetch_func = fetch_func


class Component:
    def __init__(self, fetch_func, svelte_component, **kwargs):
        self.id = str(uuid.uuid4())  # Generate a unique ID
        self.data_source = DataSource(self.id, fetch_func) if fetch_func else None
        self.svelte_component = svelte_component
        self.props = kwargs


class Dropdown(Component):
    def __init__(self, fetch_func, value_column, **kwargs):
        super().__init__(fetch_func, "Dropdown", **kwargs)
        self.value_column = value_column


class Chatbot(Component):
    def __init__(self, model, database=None, **kwargs):
        super().__init__(None, "Chatbot", **kwargs)  # No data source
        self.model = model
        self.database = database
