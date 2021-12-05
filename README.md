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


### Executando o script `scripts/get_tweets.py`

1- As variáveis `API_KEY`, `API_SECRET` e `BEARER_TOKEN` precisam ser setadas

2- Para executar:

```python scripts/get_tweets.py -t {palavra/tag sendo buscada} -n {quantidade de tweets} -o {output pro resultado}```
    
    python scripts/get_tweets.py -t "clique agora" -n 2500 -o tweets_selvagens/cliqueagora_file.csv

### Executando o script `scripts/run_model.py`

1- Se der `ModuleNotFoundError`, atualize seu `PYTHONPATH`

No diretório `datascience_seg` faça o comando:

```export PYTHONPATH="${PYTHONPATH}:${pwd}/datascience_seg"```

2- Para passar por todos os processos (os gráficos do dataset de treino e do dataset utilizado no 
predict ficarão no diretório `model/charts/{nome_classificador}_{nome_dataset}):

```python scripts/run_model.py -p -f {arquivo com os tweets para predict} -c {nome do classificador}```

    python scripts/run_model.py -p -f tweets_selvagens/giveaway_file.csv -c KNN

(Caso não queira passar pelo pré-processamento novamente, retire a flag `-p`)

Classificadores atuais: RandomForest, KNN, MLP, SVC