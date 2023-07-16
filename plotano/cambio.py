import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


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
    def __init__(self, model, **kwargs):
        super().__init__(None, "Chatbot", **kwargs)  # No data source
        self.model = model


class Application:
    def __init__(self):
        self.data_sources = {}
        self.components = []

    def add_component(self, component):
        if component.data_source:
            self.data_sources[component.id] = component.data_source
            # set data_endpoint if it's a Dropdown component
            if isinstance(component, Dropdown):
                component.props["data_endpoint"] = component.id
        self.components.append(
            {
                "id": component.id,
                "component": component,
                "svelte_component": component.svelte_component,
                "props": component.props,
            }
        )

    def create_chatbot_route(self, app, component):
        @app.route("/chat/<message>", methods=["POST"])
        def inference(message):
            try:
                output = component["component"].model(message)
                return {
                    "log": "Inference complete",
                    "status": "200",
                    "question": message,
                    "answer": output,
                }
            except Exception as ex:
                return {"log": f"Inference failed: {ex}", "status": "500"}

    def run(self):
        app = Flask(__name__)
        CORS(app)  # Allows cross-origin requests

        @app.route("/components", methods=["GET"])
        def get_components():
            return jsonify(
                [
                    {
                        "id": component["id"],
                        "svelte_component": component["svelte_component"],
                        "props": component["props"],
                    }
                    for component in self.components
                ]
            )

        def create_data_route(id, data_source):
            @app.route(f"/data/{id}", methods=["GET"], endpoint=id)
            def get_data():
                data = data_source.fetch_func()
                return jsonify(data)

        for id, data_source in self.data_sources.items():
            create_data_route(id, data_source)

        for component in self.components:
            if component["svelte_component"] == "Chatbot":
                self.create_chatbot_route(app, component)

        @app.route("/")
        def base():
            return send_from_directory("../frontend/dist", "index.html")

        @app.route("/<path:path>")
        def home(path):
            return send_from_directory("../frontend/dist", path)

        app.run(debug=True)
