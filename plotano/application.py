from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from plotano.component.base import Dropdown


class Application:
    def __init__(self, debug: bool = False):
        self._debug = debug
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
                # insert question and answer into database
                id = component["component"].database.insert_question_answer(
                    message, output
                )
                return {
                    "id": id,
                    "log": "Inference complete",
                    "status": "200",
                    "question": message,
                    "answer": output,
                }
            except Exception as ex:
                return {"log": f"Inference failed: {ex}", "status": "500"}

        @app.route("/chat/qa_table/update", methods=["POST"])
        def update_qa_table():
            try:
                request_body = request.get_json()
                component["component"].database.update_vote_status(
                    request_body["id"], request_body["vote_status"]
                )
                return {"log": "Table updated", "status": "200"}
            except Exception as ex:
                return {"log": f"Table update failed: {ex}", "status": "500"}

        @app.route("/chat/qa_table/retrieve", methods=["GET"])
        def retrieve_qa_table():
            try:
                rows = component["component"].database.retrieve_all_question_answers()
                return {"rows": rows, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

        @app.route("/chat/qa_table/close", methods=["GET"])
        def close_qa_table():
            try:
                component["component"].database.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

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
            return send_from_directory("frontend/dist", "index.html")

        @app.route("/<path:path>")
        def home(path):
            return send_from_directory("frontend/dist", path)

        # debug mode should be set to False in production because
        # it will start two processes when debug mode is enabled.
        app.run(debug=self._debug)
