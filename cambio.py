from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


class DataSource:
    def __init__(self, id, fetch_func):
        self.id = id
        self.fetch_func = fetch_func


class Component:
    def __init__(self, id, data_source, svelte_component, **kwargs):
        self.id = id
        self.data_source = data_source
        self.svelte_component = svelte_component
        self.props = kwargs


class Dropdown(Component):
    def __init__(self, id, data_source, value_column, **kwargs):
        super().__init__(id, data_source, "Dropdown", **kwargs)
        self.value_column = value_column


class Chatbot(Component):
    def __init__(self, id, model, **kwargs):
        super().__init__(id, None, "Chatbot", **kwargs)  # No data source
        self.model = model


class Application:
    def __init__(self):
        self.data_sources = {}
        self.components = []

    def add_data_source(self, data_source):
        self.data_sources[data_source.id] = data_source

    def add_component(self, component):
        self.components.append(
            {
                "id": component.id,
                "component": component,
                "svelte_component": component.svelte_component,
                "props": component.props,
            }
        )

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

        for id, data_source in self.data_sources.items():

            @app.route(f"/data/{id}", methods=["GET"], endpoint=id)
            def get_data(id=id):  # Default argument to capture the current id
                data = data_source.fetch_func()
                return jsonify(data)

        for component in self.components:
            if component["svelte_component"] == "Chatbot":

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

        @app.route("/")
        def base():
            return send_from_directory("frontend/dist", "index.html")

        @app.route("/<path:path>")
        def home(path):
            return send_from_directory("frontend/dist", path)

        app.run(debug=True)
