import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Gerar dados sintéticos para classificação
def generate_synthetic_data():
    # Gerar 1000 amostras e 20 features
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_classes=2, random_state=42)
    return X, y

# Definir o modelo MLP simples
def create_mlp_model():
    model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, activation='relu', solver='adam', random_state=42)
    return model

mlflow.set_tracking_uri("http://localhost:5000")

# Configuração do MLflow
mlflow.start_run()

# Gerar dados sintéticos
X, y = generate_synthetic_data()

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo MLP
model = create_mlp_model()

# Treinar o modelo
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.4f}')

# Logar o modelo no MLflow
mlflow.sklearn.log_model(model, "mlp_model")

# Logar a métrica de acurácia
mlflow.log_metric("accuracy", accuracy)

# Finalizar a execução no MLflow
mlflow.end_run()

print("Modelo e métricas salvos com sucesso no MLflow.")
