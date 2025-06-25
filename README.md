# semantic-router

A simple webserver that assigns objects to categories based on the meaning of their content rather than predefined rules.

The idea behind semantic routing and underlying concepts are described in [this notebook](./notebooks/introduction.ipynb). This is not a new idea and this project is intended only for demonstration purposes.

# Get started

## Testing

Execute `pytest` with `python -m pytest`

## Linting

Execute `flake8`with `flake8 . --exclude=.venv`


## todos
- add embeddings for
    - text
        - sentence-transformer powered models
    - image
        - transformers powered models
- add REST-API server
    - routes
    - request schemas
- add tests
- remove .gitkeep files that are not needed