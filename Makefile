.PHONY: install setup test eval eval-quick app docker

install:
	pip install -r requirements.txt && pip install -e .

setup:
	python -m fraud_agent.bootstrap

test:
	LLM_PROVIDER=mock RETRIEVER=tfidf pytest -q

eval:
	python -m fraud_agent.evaluate --systems langgraph,single_prompt,react

eval-quick:
	python -m fraud_agent.evaluate --systems langgraph,single_prompt,react --limit 20

app:
	streamlit run app.py

docker:
	docker compose up --build
