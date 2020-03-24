import json

from multiprocessing import Process

import aiohttp
import asyncio
import asynctest
import uvicorn

from examples.serving.sklearn import app


class TestSklearnAPI(asynctest.TestCase):
    """Tests the API class"""

    async def setUp(self):
        """ Bring server up. """
        self.process = Process(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "info"},
            daemon=True,
        )
        self.process.start()
        await asyncio.sleep(0.1)  # time for the server to start

    async def tearDown(self):
        """ Shutdown the app. """
        self.process.terminate()

    async def test_predict(self):
        """ Fetch an endpoint from the app. """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/predict",
                data=json.dumps(
                    {
                        "alcohol": 0,
                        "chlorides": 0,
                        "citric_acid": 0,
                        "density": 0,
                        "fixed_acidity": 0,
                        "free_sulfur_dioxide": 0,
                        "pH": 0,
                        "residual_sugar": 0,
                        "sulphates": 0,
                        "total_sulfur_dioxide": 0,
                        "volatile_acidity": 0,
                    }
                ),
            ) as resp:
                data = await resp.json()
        self.assertEqual(data["result"][0], 4.946541376763105)
