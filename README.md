# Execução de prompts em LLMs locais com Ollama

Projeto acadêmico de Sistemas de Informação. É um pipeline de backend em Python que lê prompts de um arquivo JSON, envia cada um aos modelos locais do Ollama e salva as respostas para comparação posterior.

## O que ele faz

1. Lê os prompts de `inputs.json`.
2. Verifica se o Ollama está ativo e se os modelos estão instalados.
3. Envia cada prompt a cada modelo, um de cada vez.
4. Salva cada resposta em um arquivo Markdown.
5. Registra todas as execuções (inclusive falhas) em um CSV.

Modelos usados: `qwen3.5:9b`, `deepseek-r1:8b` e `llama3.1:8b`. Para trocar, edite a lista `MODELOS` no notebook.

## Arquivos

```
files/
├── notebook.ipynb      # roteiro que você executa
├── pipeline.py         # funções do projeto
├── inputs.json         # prompts de entrada
├── requirements.txt    # dependências Python
├── tests/              # testes automáticos
└── outputs/            # resultados (criada ao executar)
```

## Como usar (Windows)

1. Instale o Ollama em https://ollama.com/download e baixe os modelos:

```powershell
ollama pull qwen3.5:9b
ollama pull deepseek-r1:8b
ollama pull llama3.1:8b
ollama list
```

2. Na pasta do projeto, crie o ambiente e instale as dependências:

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

3. Abra a **pasta** no VS Code, abra o `notebook.ipynb`, selecione o kernel `.venv` e clique em **Run All**.

A primeira execução de cada modelo pode levar vários minutos.

## Adicionar prompts

Acrescente um objeto à lista em `inputs.json`:

```json
{
  "id": "PROMPT-003",
  "titulo": "Título do prompt",
  "texto": "Texto enviado aos modelos."
}
```

O `id` deve ser único. O texto é enviado exatamente como escrito.

## Resultados

* `outputs/<prompt>/<modelo>\_\_<id\_execucao>.md`: um arquivo por execução, com o prompt, a resposta, o status, o tempo e os parâmetros.
* `outputs/resultados.csv`: uma linha por execução (prompt, modelo, data e hora, status, tempo, caminho do `.md` e mensagem de erro).

Resultados anteriores nunca são apagados.

## Problemas comuns

|Problema|Solução|
|-|-|
|`No module named 'requests'`|Selecione o kernel `.venv` e instale com `pip install -r requirements.txt`.|
|Ollama não responde|Abra o aplicativo Ollama ou rode `ollama serve`.|
|Modelo não encontrado|Confira a tag com `ollama list` e instale com `ollama pull <tag>`.|
|Tempo limite excedido|Aumente `TIMEOUT\_S` no notebook.|
|CSV não grava|Feche o arquivo se estiver aberto no Excel.|



