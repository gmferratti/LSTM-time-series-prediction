# Variáveis de projeto
PROJECT_ID := tcc-fiap-mlet
REGION := southamerica-east1
REGISTRY := $(REGION)-docker.pkg.dev/$(PROJECT_ID)
REPOSITORY := ml-containers
API_IMAGE := $(REGISTRY)/$(REPOSITORY)/stock-pred-api:latest
BUCKET_NAME := tcc-fiap-mlet-models

# Variáveis de desenvolvimento local
PYTHON := python3
MLFLOW_PORT := 5000

# Configuração do ambiente
setup:
	pip install -r requirements.dev.txt

# Criar bucket no GCS (se não existir)
create-bucket:
	gcloud storage buckets create gs://$(BUCKET_NAME) \
		--location=$(REGION) \
		--uniform-bucket-level-access || true

# Treinar modelo localmente
train:
	export GOOGLE_APPLICATION_CREDENTIALS=$(PWD)/secrets/key.json && \
	$(PYTHON) src/model/train.py

# Rodar MLflow UI localmente
mlflow-ui:
	mlflow ui --port=$(MLFLOW_PORT)

# Criar repositório no Artifact Registry
create-repo:
	gcloud artifacts repositories create $(REPOSITORY) \
		--repository-format=docker \
		--location=$(REGION) \
		--description="ML containers repository"


# Rodar API localmente para testes
run-api-local:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 
	

# Build container localmente
build-container-local:
	docker build -t stock-pred-api:latest -f Dockerfiles/api/Dockerfile .

# Rodar container localmente
run-container-local:
	docker run --platform linux/amd64 -p 8080:8080 stock-pred-api:latest 

# Autenticar no GCP usando a service account
auth-gcloud:
	gcloud auth activate-service-account --key-file=secrets/key.json
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

# Build e push da API
build-api:
	docker build -t stock-pred-api -f Dockerfiles/api/Dockerfile .
	docker tag stock-pred-api $(API_IMAGE)
	docker push $(API_IMAGE)

# Deploy da API no Cloud Run
deploy-api:
	gcloud run deploy stock-pred-api \
		--image $(API_IMAGE) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--min-instances 1

# Build e deploy completo
deploy-all: auth-gcloud create-repo build-api deploy-api

# Testar API local com exemplo
test-local-api:
	curl -X POST http://localhost:8080/predict \
		-H "Content-Type: application/json" \
		-d @examples/test_request.json

test-api:
	curl -X POST https://stock-pred-api-426705406065.southamerica-east1.run.app/predict \
		-H "Content-Type: application/json" \
		-d @examples/test_request.json

.PHONY: setup create-bucket train mlflow-ui create-repo build-api deploy-api run-api-local auth-gcloud deploy-all clean test-local-api test-api