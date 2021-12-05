# datascience_seg

O projeto possui um notebook para visualização mais detalhada do processo.
Também possui o script `scripts/run_model.py` que mostra os resultados de forma mais reduzida
e gera os gráficos no diretório `model/charts`.


### Configurando o ambiente. Se já souber criar um ambiente virtual, pule para o passo 3
1- Criar um ambiente virtual com python3.9

```python3.9 -m venv venv-twitter```

2- Ativar o ambiente virtual

```source venv-twitter/bin/activate```

3- Instalar todos os pacotes necessários

```pip install -r requirements.txt```

### Executando o script `scripts/run_model.py`

1- Se der `ModuleNotFoundError`, atualize seu `PYTHONPATH`

No diretório `datascience_seg` faça o comando:

```export PYTHONPATH="${PYTHONPATH}:${pwd}/datascience_seg"```

2- Para passar por todos os processos (os gráficos do dataset de treino e do dataset utilizado no 
predict ficarão no diretório `model/charts/{nome_dataset}):

```scripts/run_model.py -p```

(Caso não queira passar pelo pré-processamento novamente, retire a flag `-p`)