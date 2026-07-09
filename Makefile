.PHONY: setup lint format test run-api run-ui train clean help

PYTHON = python3
PIP = pip

help:
	@echo "ForecastIQ Make Command List:"
	@echo "  setup       Install dependencies"
	@echo "  lint        Run linting and static checks (ruff, mypy)"
	@echo "  format      Auto-format code using ruff"
	@echo "  test        Run unit tests with coverage"
	@echo "  run-api     Start the FastAPI backend server"
	@echo "  run-ui      Start the Streamlit frontend"
	@echo "  train       Train the default model"
	@echo "  clean       Remove cache and build artifacts"

setup:
	$(PYTHON) -m $(PIP) install --upgrade pip
	$(PYTHON) -m $(PIP) install -r requirements.txt

lint:
	ruff check .
	mypy src app tests

format:
	ruff format .

test:
	pytest tests/

run-api:
	uvicorn app.api:app --host 0.0.0.0 --port 8002 --reload

run-ui:
	streamlit run app/ui.py --server.port 8503

train:
	$(PYTHON) -m src.training.train --model patchtst

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
