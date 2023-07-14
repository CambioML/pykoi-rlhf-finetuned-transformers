from cambio import Chatbot, Application, Dropdown


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

# Create a chatbot component
chatbot = Chatbot("Chatbot", uppercase_model, feedback=False)

# Create the application
app = Application()

# Add the components to the application
app.add_component(chatbot)
app.add_component(dropdown)
app.add_component(dropdown2)

# Run the application
app.run()
