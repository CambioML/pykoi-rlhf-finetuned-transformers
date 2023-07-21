"""Application module."""
from typing import Any, Dict
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from pyngrok import ngrok

from plotano.component.base import Dropdown


class Application:
    """
    The Application class.
    """
    def __init__(self, share: bool = False, debug: bool = False):
        """
        Initialize the Application.

        Args:
            share (bool, optional): If True, the application will be shared via ngrok. Defaults to False.
            debug (bool, optional): If True, the application will run in debug mode. Defaults to False.
        """
        self._debug = debug
        self._share = share
        self.data_sources = {}
        self.components = []

    def add_component(self, component: Any):
        """
        Add a component to the application.

        Args:
            component (Any): The component to be added.
        """
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

    def create_chatbot_route(self, app: Flask, component: Dict[str, Any]):
        """
        Create chatbot routes for the application.

        Args:
            app (Flask): The Flask application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """
        @app.route("/chat/<message>", methods=["POST"])
        def inference(message: str):
            try:
                output = component["component"].model.predict(message)[0]
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

        @app.route("/chat/qa_table/close", methods=["GET"])
        def close_qa_table():
            try:
                component["component"].database.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

        @app.route("/chat/ranking_table/<message>", methods=["POST"])
        def inference_ranking_table(message: str):
            try:
                request_body = request.get_json()
                num_of_response = request_body.get("n", 2)
                output = component["component"].model.predict(message, num_of_response)
                # Check the type of each item in the output list
                return {
                    "log": "Inference complete",
                    "status": "200",
                    "question": message,
                    "answer": output,
                }
            except Exception as ex:
                return {"log": f"Inference failed: {ex}", "status": "500"}

        @app.route("/chat/ranking_table/update", methods=["POST"])
        def update_ranking_table():
            try:
                request_body = request.get_json()
                component["component"].database.insert_ranking(
                    request_body["question"],
                    request_body["up_ranking_answer"],
                    request_body["low_ranking_answer"],
                )
                return {"log": "Table updated", "status": "200"}
            except Exception as ex:
                return {"log": f"Table update failed: {ex}", "status": "500"}

        @app.route("/chat/ranking_table/retrieve", methods=["GET"])
        def retrieve_ranking_table():
            try:
                rows = component["component"].database.retrieve_all_question_answers()
                return {"rows": rows, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

    def create_feedback_route(self, app: Flask, component: Dict[str, Any]):
        """
        Create feedback routes for the application.

        Args:
            app (Flask): The Flask application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """
        @app.route("/chat/qa_table/retrieve", methods=["GET"])
        def retrieve_qa_table():
            try:
                rows = component["component"].database.retrieve_all_question_answers()
                return {"rows": rows, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

        @app.route("/chat/ranking_table/close", methods=["GET"])
        def close_ranking_table():
            try:
                component["component"].database.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

    def run(self):
        """
        Run the application.
        """
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

        def create_data_route(id: str, data_source: Any):
            """
            Create data route for the application.

            Args:
                id (str): The id of the data source.
                data_source (Any): The data source.
            """
            @app.route(f"/data/{id}", methods=["GET"], endpoint=id)
            def get_data():
                data = data_source.fetch_func()
                return jsonify(data)

        for id, data_source in self.data_sources.items():
            create_data_route(id, data_source)

        for component in self.components:
            if component["svelte_component"] == "Chatbot":
                self.create_chatbot_route(app, component)
            if component["svelte_component"] == "Feedback":
                self.create_feedback_route(app, component)

        @app.route("/")
        def base():
            return send_from_directory("frontend/dist", "index.html")

        @app.route("/<path:path>")
        def home(path: str):
            return send_from_directory("frontend/dist", path)

        # debug mode should be set to False in production because
        # it will start two processes when debug mode is enabled.

        # Set the ngrok tunnel if share is True
        if self._share:
            public_url = ngrok.connect("http://127.0.0.1:5000")
            print("Public URL:", public_url)

            app.run(debug=self._debug)

            print("Stopping server...")
            ngrok.disconnect(public_url)
        else:
            app.run(debug=self._debug)
