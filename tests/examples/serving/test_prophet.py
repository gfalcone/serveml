import json

from multiprocessing import Process

import aiohttp
import asyncio
import asynctest
import uvicorn

from examples.serving.prophet import app


class TestProphetAPI(asynctest.TestCase):
    """Tests the API class"""

    async def setUp(self):
        """ Bring server up. """
        self.process = Process(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "info"
            },
            daemon=True
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
                    data=json.dumps({"periods": 5})
            ) as resp:
                data = await resp.json()
        keys = set(data["result"][0].keys())
        expected_keys = {
            'ds',
            'trend',
            'yhat_lower',
            'yhat_upper',
            'trend_lower',
            'trend_upper',
            'additive_terms',
            'additive_terms_lower',
            'additive_terms_upper',
            'weekly',
            'weekly_lower',
            'weekly_upper',
            'yearly',
            'yearly_lower',
            'yearly_upper',
            'multiplicative_terms',
            'multiplicative_terms_lower',
            'multiplicative_terms_upper',
            'yhat'
        }
        self.assertEqual(keys.intersection(expected_keys), expected_keys)
