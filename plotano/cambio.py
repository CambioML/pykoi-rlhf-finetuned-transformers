import uuid

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import openai

# from gpt4all import GPT4All


# class GPT4AllModel:
#     def __init__(self, model_path, max_tokens=3):
#         self.model = GPT4All(model_path)
#         self.max_tokens = max_tokens

#     def predict(self, message):
#         output = self.model.generate(message, max_tokens=self.max_tokens)
#         return output


class FunctionModel:
    def __init__(self, func):
        self.func = func

    def predict(self, prompt):
        return self.func(prompt)


class OpenAIModel:
    def __init__(self, api_key, engine="davinci", max_tokens=100, temperature=0.5):
        openai.api_key = api_key
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature

    def predict(self, message):
        prompt = f"Question: {message}\nAnswer:"
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=self.temperature,
        )
        return response.choices[0].text.split("\n")[0]


class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type.lower() == "openai":
            return OpenAIModel(**kwargs)
        elif model_type.lower() == "gpt4all":
            return GPT4AllModel(**kwargs)
        elif model_type.lower() == "function":
            return FunctionModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


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

        app.run(debug=True)
