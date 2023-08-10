"""Application module."""
import os
import socket
import time
from datetime import datetime
from typing import List, Optional, Any, Dict, Union
from fastapi import FastAPI, Depends, HTTPException, UploadFile, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyngrok import ngrok
from starlette.middleware.cors import CORSMiddleware
from pykoi.component.base import Dropdown
from pykoi.telemetry.telemetry import Telemetry
from pykoi.telemetry.events import (
    AppStartEvent,
    AppStopEvent)


oauth_scheme = HTTPBasic()


class UpdateQATable(BaseModel):
    id: int
    vote_status: str


class RankingTableUpdate(BaseModel):
    question: str
    up_ranking_answer: str
    low_ranking_answer: str


class InferenceRankingTable(BaseModel):
    n: Optional[int] = 2


class ModelAnswer(BaseModel):
    model: str
    qid: int
    rank: int
    answer: str


class ComparatorInsertRequest(BaseModel):
    data: List[ModelAnswer]


class UserInDB:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password


class Application:
    """
    The Application class.
    """

    def __init__(
        self,
        share: bool = False,
        debug: bool = False,
        username: Union[None, str, List] = None,
        password: Union[None, str, List] = None,
        host: str = "127.0.0.1",
        port: int = 5000,
        enable_telemetry: bool = True,
    ):
        """
        Initialize the Application.

        Args:
            share (bool, optional): If True, the application will be shared via ngrok. Defaults to False.
            debug (bool, optional): If True, the application will run in debug mode. Defaults to False.
            username (str, optional): The username for authentication. Defaults to None.
            password (str, optional): The password for authentication. Defaults to None.
            host (str): The host to run the application on. Defaults to None.
            port (int): The port to run the application on. Defaults to None.
            enable_telemetry (bool, optional): If True, enable_telemetry will be enabled. Defaults to True.
        """
        self._debug = debug
        self._share = share
        self._host = host
        self._port = port
        self.data_sources = {}
        self.components = []
        if username and password:
            self._auth = True
        else:
            self._auth = False
        if isinstance(username, str):
            username = [username]
        if isinstance(password, str):
            password = [password]
        if (
            username is not None
            and password is not None
            and len(username) != len(password)
        ):
            raise ValueError(
                "The length of username and password must be the same."
            )
        self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        self._fake_users_db = {}
        if username is not None and password is not None:
            for user_name, pass_word in zip(username, password):
                self._fake_users_db[user_name] = UserInDB(
                    username=user_name,
                    hashed_password=self._pwd_context.hash(pass_word),
                )
        self._telemetry = Telemetry(enable_telemetry)

    def authenticate_user(self, fake_db, username: str, password: str):
        if self._auth:
            user = fake_db.get(username)
            if not user:
                return False
            if not self._pwd_context.verify(password, user.hashed_password):
                return False
            return user
        else:
            return "no_auth"

    def auth_required(
        self, credentials: HTTPBasicCredentials = Depends(oauth_scheme)
    ):
        user = self.authenticate_user(
            self._fake_users_db, credentials.username, credentials.password
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return user

    def dummy_auth(self):
        return None

    def get_auth_dependency(self):
        if self._auth:
            return self.auth_required
        else:
            return self.dummy_auth

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

    def create_chatbot_route(self, app: FastAPI, component: Dict[str, Any]):
        """
        Create chatbot routes for the application.

        Args:
            app (FastAPI): The FastAPI application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """

        @app.post("/chat/{message}")
        async def inference(
            message: str,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
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

        @app.post("/chat/qa_table/update")
        async def update_qa_table(
            request_body: UpdateQATable,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            try:
                component["component"].database.update_vote_status(
                    request_body.id, request_body.vote_status
                )
                return {"log": "Table updated", "status": "200"}
            except Exception as ex:
                return {"log": f"Table update failed: {ex}", "status": "500"}

        @app.get("/chat/qa_table/close")
        async def close_qa_table(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                component["component"].database.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

        @app.post("/chat/multi_responses/{message}")
        async def inference_ranking_table(
            message: str,
            request_body: InferenceRankingTable,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            try:
                num_of_response = request_body.n
                output = component["component"].model.predict(
                    message, num_of_response
                )
                # Check the type of each item in the output list
                return {
                    "log": "Inference complete",
                    "status": "200",
                    "question": message,
                    "answer": output,
                }
            except Exception as ex:
                return {"log": f"Inference failed: {ex}", "status": "500"}

        @app.post("/chat/ranking_table/update")
        async def update_ranking_table(
            request_body: RankingTableUpdate,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            try:
                component["component"].database.insert_ranking(
                    request_body.question,
                    request_body.up_ranking_answer,
                    request_body.low_ranking_answer,
                )
                return {"log": "Table updated", "status": "200"}
            except Exception as ex:
                return {"log": f"Table update failed: {ex}", "status": "500"}

        @app.get("/chat/ranking_table/retrieve")
        async def retrieve_ranking_table(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                print("retrieve_ranking_table")
                rows = component[
                    "component"
                ].database.retrieve_all_question_answers()
                return {"rows": rows, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

    def create_feedback_route(self, app: FastAPI, component: Dict[str, Any]):
        """
        Create feedback routes for the application.

        Args:
            app (FastAPI): The FastAPI application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """

        @app.get("/chat/qa_table/retrieve")
        async def retrieve_qa_table(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                rows = component[
                    "component"
                ].database.retrieve_all_question_answers()
                return {"rows": rows, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

        @app.get("/chat/ranking_table/close")
        async def close_ranking_table(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                component["component"].database.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

    def create_chatbot_comparator_route(
        self, app: FastAPI, component: Dict[str, Any]
    ):
        """
        Create chatbot comparator routes for the application.

        Args:
            app (FastAPI): The FastAPI application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """

        @app.post("/chat/comparator/{message}")
        async def compare(
            message: str,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            try:
                output_dict = {}
                # insert question and answer into database
                qid = component["component"].question_db.insert(
                    question=message,
                )
                # TODO: refactor to run multiple models in parallel using threading
                for model_name, model in component["component"].models.items():
                    output = model.predict(message)[0]
                    # TODO: refactor this into using another comparator database
                    output_dict[model_name] = output
                    component["component"].comparator_db.insert(
                        model=model_name,
                        qid=qid,
                        rank=1,  # default rank is 1
                        answer=output,
                    )
                return {
                    "qid": qid,
                    "log": "Inference complete",
                    "status": "200",
                    "question": message,
                    "answer": output_dict,
                }
            except Exception as ex:
                return {"log": f"Inference failed: {ex}", "status": "500"}

        @app.post("/chat/comparator/db/update")
        async def update_comparator(
            request: ComparatorInsertRequest,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            print("REQ", request.data)
            try:
                for model_answer in request.data:
                    component["component"].comparator_db.update(
                        model=model_answer.model,
                        qid=model_answer.qid,
                        rank=model_answer.rank,
                    )
                return {"log": "Table updated", "status": "200"}
            except Exception as ex:
                return {"log": f"Table update failed: {ex}", "status": "500"}

        @app.get("/chat/comparator/db/retrieve")
        async def retrieve_comparator(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                rows = component["component"].comparator_db.retrieve_all()
                data = []
                for row in rows:
                    _, model_name, qid, rank, answer, _ = row

                    data.append(
                        {
                            "model": model_name,
                            "qid": qid,
                            "rank": rank,
                            "answer": answer,
                        }
                    )
                return {"data": data, "log": "Table retrieved", "status": "200"}
            except Exception as ex:
                return {"log": f"Table retrieval failed: {ex}", "status": "500"}

        @app.get("/chat/comparator/db/close")
        async def close_comparator(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                component["component"].question_db.close_connection()
                component["component"].comparator_db.close_connection()
                return {"log": "Table closed", "status": "200"}
            except Exception as ex:
                return {"log": f"Table close failed: {ex}", "status": "500"}

    def create_qa_retrieval_route(self, app: FastAPI, component: Dict[str, Any]):
        """
        Create QA retrieval routes for the application.

        Args:
            app (FastAPI): The FastAPI application.
            component (Dict[str, Any]): The component for which the routes are being created.
        """
        @app.get("/retrieval/file/get")
        async def get_files(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            print("[/retrieval/file/get]: getting files...")
            # create folder if it doesn't exist
            dir_path = os.environ["DOC_PATH"]
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            files = os.listdir(dir_path)

            # Create a list of dictionaries, each containing the file's name, size and type
            file_data = []
            for file in files:
                size = os.path.getsize(
                    os.path.join(dir_path, file)
                )  # get size of file in bytes
                _, ext = os.path.splitext(file)  # split the file name into name and extension
                file_data.append(
                    {
                        "name": file,
                        "size": size,
                        "type": ext[1:],  # remove the period from the extension
                    }
                )
            return {"files": file_data}

        @app.post("/retrieval/file/upload")
        async def upload_files(files: List[UploadFile],
                               user: Union[None, UserInDB] = Depends(self.get_auth_dependency())):
            try:
                # create folder if it doesn't exist
                if not os.path.exists(os.getenv("DOC_PATH")):
                    os.makedirs(os.getenv("DOC_PATH"))

                print("[/retrieval/file/upload]: upload files...")
                # Check if any file is sent
                if not files:
                    raise HTTPException(status_code=400, detail="No file part")

                filenames = []
                # Iterate over each file
                for file in files:
                    # Check if file is selected
                    if not file.filename:
                        raise HTTPException(status_code=400, detail="No selected file")

                    print(f"[/retrieval/file/upload]: saving file {file.filename}")
                    # Save or process the file
                    with open(os.path.join(os.getenv("DOC_PATH"), file.filename), "wb") as buffer:
                        buffer.write(await file.read())
                    filenames.append(file.filename)

                # List all files in the DOC_PATH directory
                file_list = os.listdir(os.getenv("DOC_PATH"))

                return JSONResponse({"status": "ok", "filenames": filenames, "files": file_list})

            except Exception as e:
                return JSONResponse({"status": "error", "message": str(e)})

        @app.post("/retrieval/vector_db/index")
        async def index_vector_db(user: Union[None, UserInDB] = Depends(self.get_auth_dependency())):
            try:
                print("[/retrieval/vector_db/index]: indexing files...")
                component["component"].vector_db.index()
                return {'log': 'Indexing complete',
                        'status': '200'}
            except Exception as ex:
                return {'log': f'Indexing failed: {ex}',
                        'status': '500'}

        @app.get("/retrieval/{message}")
        async def inference(
            message: str,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            try:
                print("[/retrieval]: model inference...")
                output = component["component"].retrieval_model.run(message)
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
                return {'log': f'Inference failed: {ex}',
                        'status': '500'}

        @app.get("/retrieval/vector_db/get")
        async def get_vector_db(user: Union[None, UserInDB] = Depends(self.get_auth_dependency())):
            try:
                print("[/retrieval/vector_db/get]: get embedding...")
                response_dict = component["component"].vector_db.get_embedding()
                return response_dict
            except Exception as ex:
                return {"log": "Failed to get embedding: {}".format(ex),
                        "status": "500"}

    def run(self):
        """
        Run the application.
        """
        app = FastAPI()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/token")
        def login(credentials: HTTPBasicCredentials = Depends(oauth_scheme)):
            user = self.authenticate_user(
                self._fake_users_db, credentials.username, credentials.password
            )
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Basic"},
                )
            return {"message": "Logged in successfully"}

        @app.get("/components")
        async def get_components(
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency())
        ):
            return JSONResponse(
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

            @app.get(f"/data/{id}")
            async def get_data(
                user: Union[None, UserInDB] = Depends(
                    self.get_auth_dependency()
                )
            ):
                data = data_source.fetch_func()
                return JSONResponse(data)

        for id, data_source in self.data_sources.items():
            create_data_route(id, data_source)

        for component in self.components:
            if component["svelte_component"] == "Chatbot":
                self.create_chatbot_route(app, component)
            if component["svelte_component"] == "Feedback":
                self.create_feedback_route(app, component)
            if component["svelte_component"] == "Compare":
                self.create_chatbot_comparator_route(app, component)
            if component["svelte_component"] == "RetrievalQA":
                self.create_qa_retrieval_route(app, component)

        app.mount(
            "/",
            StaticFiles(
                directory=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "frontend/dist"
                ),
                html=True,
            ),
            name="static",
        )

        @app.get("/{path:path}")
        async def read_item(
            path: str,
            user: Union[None, UserInDB] = Depends(self.get_auth_dependency()),
        ):
            return {"path": path}

        # debug mode should be set to False in production because
        # it will start two processes when debug mode is enabled.

        # Set the ngrok tunnel if share is True
        start_event = AppStartEvent(
            start_time=time.time(),
            date_time=datetime.utcfromtimestamp(time.time())
        )
        self._telemetry.capture(start_event)

        if self._share:
            public_url = ngrok.connect(self._host + ":" + str(self._port))
            print("Public URL:", public_url)
            import uvicorn

            uvicorn.run(app, host=self._host, port=self._port)
            print("Stopping server...")
            ngrok.disconnect(public_url)
        else:
            import uvicorn

            uvicorn.run(app, host=self._host, port=self._port)
        self._telemetry.capture(AppStopEvent(
            end_time=time.time(),
            date_time=datetime.utcfromtimestamp(time.time()),
            duration=time.time() - start_event.start_time
        ))
