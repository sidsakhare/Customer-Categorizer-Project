from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import Trainpipeline