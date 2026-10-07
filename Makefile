.PHONY: install train test lint run docker
install:
	python -m pip install -r requirements.txt
train:
	python scripts/train.py --output models
test:
	pytest -q
lint:
	python -m compileall -q app scripts tests
run:
	uvicorn app.main:app --reload
