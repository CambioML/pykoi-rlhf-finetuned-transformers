import os
from plotano.cambio import (
    Application,
    Chatbot,
    Dropdown)
from plotano.llm.model_factory import ModelFactory


# Define a function that represents your model
def uppercase_model(text):
    return text.upper()


# Define a function that fetches data for dropdown
def fetch_dropdown_data():
    return ["Option 1", "Option 2", "Option 3"]


dropdown = Dropdown(
    "Dropdown",
    fetch_dropdown_data,
    value_column="Option 1",
)


# Define a function that fetches data for dropdown
def get_data():
    return ["a", "b", "c"]


dropdown2 = Dropdown(
    "Dropdown2",
    get_data,
    value_column="a",
)

os.environ["OPENAI_API_KEY"] = ""
if os.environ["OPENAI_API_KEY"] == "":
    raise ValueError("OPENAI_API_KEY is not set")

# Create a chatbot component
chatbot = Chatbot("Chatbot",
                  ModelFactory.create_model("openai").predict,
                  feedback=False)

# Create the application
app = Application()

# Add the components to the application
app.add_component(chatbot)
app.add_component(dropdown)
app.add_component(dropdown2)

# Run the application
app.run()
