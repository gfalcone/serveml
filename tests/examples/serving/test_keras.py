import json

from multiprocessing import Process

import aiohttp
import asyncio
import asynctest
import uvicorn

from examples.serving.keras import app


class TestKerasAPI(asynctest.TestCase):
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
        await asyncio.sleep(5)  # time for the server to start

    async def tearDown(self):
        """ Shutdown the app. """
        self.process.terminate()

    @asynctest.skip("Does not work at the moment")
    async def test_predict(self):
        """ Fetch an endpoint from the app. """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/predict/",
                data=json.dumps({"sequence": [1, 2]}),
            ) as resp:
                data = await resp.json()
                print(data)
                max_category = -1
                maximum = 0
                for key in data["result"].keys():
                    if data["result"][key]["0"] > maximum:
                        maximum = data["result"][key]["0"]
                        max_category = key

                self.assertEqual(max_category, "3")
