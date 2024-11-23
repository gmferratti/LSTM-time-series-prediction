""" Classes utilizadas no pré-processamento de dados."""

import torch
from torch.utils.data import Dataset
import torch.nn as nn

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        """
        Inicializa o dataset para séries temporais.

        Parâmetros:
        X (array-like): Dados de entrada (features) escalonados.
        y (array-like): Dados de saída (target) escalonados.
        sequence_length (int): Comprimento da sequência para gerar janelas de tempo.
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        """
        Retorna o número total de amostras possíveis no dataset.
        Isso é ajustado para garantir que as janelas de sequência
        não excedam o tamanho dos dados disponíveis.
        """
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        """
        Gera uma única amostra de dados no formato de janelas deslizantes.
        
        Parâmetros:
        idx (int): Índice inicial da janela de sequência.

        Retorno:
        tuple: Uma tupla contendo:
            - x_seq (torch.Tensor): Sequência de entrada com tamanho definido por sequence_length.
            - y_seq (torch.Tensor): Valor alvo correspondente à sequência (o próximo valor após a sequência).
        """
        # Cria a janela de sequência para as features (entrada)
        x_seq = self.X[idx: idx + self.sequence_length]
        # Define o próximo valor (target) correspondente à janela
        y_seq = self.y[idx + self.sequence_length]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)
    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Modelo de rede neural baseado em LSTM para previsão de séries temporais.

        Parâmetros:
        input_size (int): Dimensão da entrada para cada timestep (número de features por timestep).
        hidden_size (int): Dimensão do estado oculto do LSTM.
        num_layers (int): Número de camadas empilhadas do LSTM.
        output_size (int): Dimensão da saída final (geralmente 1 para previsão de valores únicos).
        """
        super(LSTMModel, self).__init__()

        # Definição do LSTM
        # - input_size: Número de features por timestep.
        # - hidden_size: Dimensão do estado oculto.
        # - num_layers: Número de camadas empilhadas do LSTM.
        # - batch_first: Define se o batch será a primeira dimensão da entrada (True).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Camada totalmente conectada (fully connected)
        # - Recebe a saída do LSTM (hidden_size) e gera a previsão (output_size).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Passo de forward da rede, processando os dados de entrada e retornando a previsão.

        Parâmetros:
        x (torch.Tensor): Tensor de entrada no formato [batch_size, sequence_length, input_size].

        Retorno:
        torch.Tensor: Previsões no formato [batch_size, output_size].
        """
        # Passa os dados de entrada pelo LSTM.
        # out: Saída do LSTM para cada timestep, no formato [batch_size, sequence_length, hidden_size].
        # h_n: Último estado oculto para cada camada, no formato [num_layers, batch_size, hidden_size].
        # c_n: Último estado da célula para cada camada, no formato [num_layers, batch_size, hidden_size].
        out, (h_n, c_n) = self.lstm(x)

        # Seleciona a saída do último timestep (out[:, -1, :]).
        # - out[:, -1, :] extrai a saída do último timestep para cada amostra no batch.
        # - Isso representa o estado oculto final do LSTM para a sequência de entrada.
        out = self.fc(out[:, -1, :])

        # Retorna a previsão final.
        return out
