# semantic-router

A simple webserver that assigns objects to categories based on the meaning of their content rather than predefined rules.

The idea behind semantic routing and underlying concepts are described in [this notebook](./notebooks/introduction.ipynb). This is not a new idea and this project is intended only for demonstration purposes.

# Get started

## Testing

Execute `pytest` with `python -m pytest`

## Linting

Execute `flake8`with `flake8 . --exclude=.venv`

```python
from semantic_router.encoder import HuggingFaceEncoder
from semantic_router.router import SemanticRouter, Route
from pprint import pprint

encoder = HuggingFaceEncoder()

routes = [
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

router = SemanticRouter(encoder=encoder, routes=routes, top_k=2)

result = router.route("I think it is snowing outside...")


for r in result:
    pprint(f"{r}")
```

## todos
- add embeddings for
    - image
        - transformers powered models
- add REST-API server
    - routes
    - request schemas
- add tests
- remove .gitkeep files that are not needed