from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from plotano.component.base import Dropdown


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
                output = component["component"].model.predict(message)

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
        # TODO: debug mode should be set to False in production because
        # it will start two processes.
        app.run(debug=True)
