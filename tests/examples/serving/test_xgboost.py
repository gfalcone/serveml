import json

from multiprocessing import Process

import aiohttp
import asyncio
import asynctest
import pytest
import uvicorn

from examples.serving.xgboost import app


class TestXgboostAPI(asynctest.TestCase):
    """Tests the API class"""

    @pytest.mark.asyncio
    async def setUp(self):
        """ Bring server up. """
        self.process = Process(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": 8005, "log_level": "info"},
            daemon=True,
        )
        self.process.start()
        await asyncio.sleep(0.1)  # time for the server to start

    @pytest.mark.asyncio
    async def tearDown(self):
        """ Shutdown the app. """
        self.process.terminate()

    @pytest.mark.asyncio
    async def test_predict(self):
        """ Fetch an endpoint from the app. """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8005/predict",
                data=json.dumps(
                    {
                        "sepal_length": 4.3,
                        "sepal_width": 2.0,
                        "petal_length": 1.0,
                        "petal_width": 0.1,
                    }
                ),
            ) as resp:
                data = await resp.json()
        self.assertEqual(
            data["result"],
            [[0.9459934234619141, 0.025751503184437752, 0.028255023062229156]],
        )
