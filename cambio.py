from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


class DataSource:
    def __init__(self, id, fetch_func):
        self.id = id
        self.fetch_func = fetch_func


class Component:
    def __init__(self, id, data_source):
        self.id = id
        self.data_source = data_source


class Dropdown(Component):
    def __init__(self, id, data_source, value_column):
        super().__init__(id, data_source)
        self.value_column = value_column


class Chatbot(Component):
    def __init__(self, id, model):
        print("id", id)
        super().__init__(id, None)  # No data source
        self.model = model


class Application:
    def __init__(self):
        self.data_sources = {}
        self.components = {}

    def add_data_source(self, data_source):
        self.data_sources[data_source.id] = data_source

    def add_component(self, component):
        print("adding component", component)
        self.components[component.id] = component

    def run(self):
        app = Flask(__name__)
        CORS(app)  # Allows cross-origin requests

        @app.route("/components", methods=["GET"])
        def get_components():
            return jsonify(list(self.components.keys()))

        for id, data_source in self.data_sources.items():

            @app.route(f"/data/{id}", methods=["GET"], endpoint=id)
            def get_data(id=id):  # Default argument to capture the current id
                data = data_source.fetch_func()
                return jsonify(data)

        for id, component in self.components.items():
            if isinstance(component, Chatbot):

                @app.route("/chat/<message>", methods=["POST"])
                def inference(message):
                    try:
                        output = component.model(message)
                        print("6575675", output)
                        return {
                            "log": "Inference complete",
                            "status": "200",
                            "question": message,
                            "answer": output,
                        }
                    except Exception as ex:
                        return {"log": f"Inference failed: {ex}", "status": "500"}

        @app.route("/")
        def base():
            return send_from_directory("frontend/dist", "index.html")

        @app.route("/<path:path>")
        def home(path):
            return send_from_directory("frontend/dist", path)

        app.run(debug=True)
