import asyncio
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from semantic_router.encoder import HuggingFaceEncoder
from semantic_router.router import SemanticRouter, Route


# --------------------------------------------------------------------------- #
#                              Initialise router                              #
# --------------------------------------------------------------------------- #
ENCODER = HuggingFaceEncoder()  # loads once at startup

ROUTES: List[Route] = [
    Route(
        name="joke",
        description="A route to tell a light-hearted or funny joke.",
        examples=[
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call fake spaghetti? An impasta.",
        ],
    ),
    Route(
        name="weather",
        description="A route to talk about the weather and forecasts.",
        examples=[
            "What's the weather like today?",
            "Is it going to rain tomorrow?",
        ],
    ),
]

DEFAULT_TOP_K = 2  # Default number of top matches to return
SEM_ROUTER = SemanticRouter(encoder=ENCODER, routes=ROUTES, top_k=DEFAULT_TOP_K)


# --------------------------------------------------------------------------- #
#                               FastAPI set-up                                #
# --------------------------------------------------------------------------- #
app = FastAPI(title="Semantic-Router Demo", version="1.0")


class RouteRequest(BaseModel):
    query: str = Field(..., description="User input string to be routed")
    top_k: int | None = Field(
        None, ge=1, description="Override global top-k (optional)"
    )


class Match(BaseModel):
    rank: int
    route_name: str
    cosine_similarity: float


class RouteResponse(BaseModel):
    query: str
    routes: List[Match]


class BatchRouteRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, description="List of user queries")
    top_k: int | None = Field(
        None, ge=1, description="Override global top-k (optional)"
    )


class BatchRouteResponse(BaseModel):
    results: List[RouteResponse]


@app.post("/route", response_model=RouteResponse)
async def route_query(req: RouteRequest) -> RouteResponse:
    """
    Return the top-k matching routes for the supplied query.
    """
    try:
        top_k = req.top_k or SEM_ROUTER.top_k
        SEM_ROUTER.top_k = top_k  # temporary override if provided
        result = SEM_ROUTER.route(req.query)[0]  # router returns a list
        return RouteResponse(query=result["query"], routes=result["route"])
    finally:
        SEM_ROUTER.top_k = DEFAULT_TOP_K  # reset to default


@app.post("/route/batch", response_model=BatchRouteResponse)
async def route_batch(req: BatchRouteRequest) -> BatchRouteResponse:
    top_k = req.top_k or SEM_ROUTER.top_k
    try:
        SEM_ROUTER.top_k = top_k
        routed = SEM_ROUTER.route(list(req.queries))
        results = [
            RouteResponse(query=item["query"], routes=item["route"]) for item in routed
        ]
        return BatchRouteResponse(results=results)
    finally:
        SEM_ROUTER.top_k = DEFAULT_TOP_K  # reset default


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
#                                Entry point                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Use uvicorn workers >1 only if model fits into memory multiple times.
    uvicorn.run(
        "example_server:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        workers=1,
        loop="uvloop" if hasattr(asyncio, "new_event_loop_policy") else "auto",
    )
