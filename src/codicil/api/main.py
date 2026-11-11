"""ASGI entrypoint: uvicorn codicil.api.main:app"""

import logging

from codicil.api.app import create_app
from codicil.settings import Settings

_settings = Settings()
logging.basicConfig(level=_settings.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
app = create_app(_settings)
