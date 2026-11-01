# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
from contextlib import asynccontextmanager

import uuid
import ray
import ray.exceptions
from fastapi import FastAPI, HTTPException, Request

from sllm.logger import init_logger
from sllm.routers.router_utils import await_backend_call

logger = init_logger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connect to the Ray cluster
        # ray.init()
        yield
        # Shutdown the Ray cluster
        # ray.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/register")
    async def register_handler(request: Request):
        body = await request.json()

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )
        try:
            await await_backend_call(controller.register.remote(model_config=body))
        except ray.exceptions.RayTaskError as e:
            cause = getattr(e, "cause", None) or e
            logger.exception("Register failed (Ray task): %s", cause)
            raise HTTPException(status_code=500, detail=str(cause))
        except Exception as e:
            logger.exception("Register failed")
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": "ok"}

    @app.post("/update")
    async def update_handler(request: Request):
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to update model {model_name}")
        try:
            await await_backend_call(controller.update.remote(model_name=model_name, model_config=body))
        except ray.exceptions.RayTaskError as e:
            raise HTTPException(status_code=400, detail=str(e.cause))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"status": f"updated model {model_name}"}

    @app.post("/delete")
    async def delete_model(request: Request):
        body = await request.json()

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        controller = ray.get_actor("controller")
        if not controller:
            raise HTTPException(
                status_code=500, detail="Controller not initialized"
            )

        logger.info(f"Received request to delete model {model_name}")
        await await_backend_call(controller.delete.remote(model_name=model_name))

        return {"status": f"deleted model {model_name}"}

    async def inference_handler(request: Request, action: str):
        body = await request.json()
        model_name = body.get("model")
        if body.get("request_id") is None:
            request_id = str(uuid.uuid4())
            body["request_id"] = request_id
        else:
            request_id = body.get("request_id")
        logger.info(f"Received request for model {model_name}")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Missing model_name in request body"
            )

        try:
            controller = ray.get_actor("controller")
        except ValueError:
            raise HTTPException(status_code=500, detail="Controller not found")

        try:
            result = await await_backend_call(controller.generate_stream.remote(model_name=model_name, request_data=body))
            return result
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    async def generate_handler(request: Request):
        return await inference_handler(request, "generate")

    @app.post("/save_stats")
    async def save_stats_handler(request: Request):
        """Flush solver timing and resource stats to disk.

        Optional JSON body: ``{"output_dir": "/path/to/dir"}``
        Returns the paths of the written files.
        """
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        output_dir = body.get("output_dir", None)
        try:
            controller = ray.get_actor("controller")
        except ValueError:
            raise HTTPException(status_code=500, detail="Controller not found")
        try:
            paths = await await_backend_call(controller.save_stats.remote(output_dir=output_dir))
            return {"status": "ok", "files": paths}
        except Exception as e:
            logger.error("save_stats failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def get_models():
        logger.info("Attempting to retrieve the controller actor")
        try:
            controller = ray.get_actor("controller")
            if not controller:
                logger.error("Controller not initialized")
                raise HTTPException(
                    status_code=500, detail="Controller not initialized"
                )
            logger.info("Controller actor found")
            result = await await_backend_call(controller.status.remote())
            logger.info("Controller status retrieved successfully")
            return result
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve models"
            )

    return app
