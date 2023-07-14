from cambio import Chatbot, Application, Dropdown, DataSource


# Define a function that represents your model
def uppercase_model(text):
    return text.upper()


# Define a function that fetches data for dropdown
def fetch_dropdown_data():
    return ["Option 1", "Option 2", "Option 3"]


dropdown_data_source = DataSource("aaa", fetch_dropdown_data)
dropdown = Dropdown(
    "Dropdown",
    dropdown_data_source,
    value_column="Option 1",
    data_endpoint="aaa",
)


# Define a function that fetches data for dropdown
def get_data():
    return ["a", "b", "c"]


ddata = DataSource("bbb", get_data)
dropdown2 = Dropdown(
    "Dropdown2",
    ddata,
    value_column="a",
    data_endpoint="bbb",
)


# Create a chatbot component
chatbot = Chatbot("Chatbot", uppercase_model, feedback=False)

# Create the application
app = Application()

# Add the data source to the application
app.add_data_source(dropdown_data_source)
app.add_data_source(ddata)

# Add the components to the application
app.add_component(chatbot)
app.add_component(dropdown)
app.add_component(dropdown2)


# Run the application
app.run()
