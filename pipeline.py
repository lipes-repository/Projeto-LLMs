"""
pipeline.py - funções do pipeline que executa prompts em LLMs locais (Ollama).

Por que as funções ficam neste arquivo e não dentro do notebook?
  - O notebook fica curto e mostra a ORDEM das etapas (ler -> verificar -> executar -> salvar).
  - Os testes automáticos (pasta tests/) conseguem importar estas funções;
    isso não é possível com código que mora dentro de um arquivo .ipynb.
  - No notebook há células que exibem o código de cada função, para você ler e explicar.

Abordagem escolhida para falar com o Ollama: requisições HTTP com a biblioteca "requests".
  - O Ollama mantém um servidor local em http://localhost:11434.
  - Enviar um prompt é simplesmente fazer um POST de um JSON para /api/generate.
  - Assim você enxerga exatamente o que é enviado e o que volta (bom para aprender e explicar).
  - Dá para controlar o tempo limite (timeout) e tratar cada tipo de erro de forma explícita.
"""

import csv
import difflib
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests

# Endereço padrão do servidor local do Ollama.
URL_PADRAO_OLLAMA = "http://localhost:11434"

# Colunas do CSV consolidado (uma linha por execução).
# Os dicionários de resultado usam exatamente estes nomes como chaves; assim, para
# gravar uma linha no CSV basta entregar o dicionário ao csv.DictWriter.
COLUNAS_CSV = [
    "id_execucao",
    "id_prompt",
    "titulo_prompt",
    "modelo",
    "data_hora",
    "status",
    "tempo_execucao_s",
    "arquivo_markdown",
    "mensagem_erro",
    "tokens_prompt",
    "tokens_gerados",
    "duracao_ollama_s",
    "motivo_termino",
    "parametros",
]


class ErroPipeline(Exception):
    """Erro "esperado" do projeto (arquivo inválido, Ollama fora do ar etc.).

    Criamos uma classe própria para separar os problemas que já sabemos explicar ao usuário
    (com mensagem clara) de erros inesperados de programação, que devem aparecer normalmente.
    """


# ---------------------------------------------------------------------------
# 1) Verificação do Ollama e dos modelos
# ---------------------------------------------------------------------------

def listar_modelos_instalados(url=URL_PADRAO_OLLAMA, timeout_s=10):
    """Pergunta ao Ollama quais modelos estão instalados e devolve a lista de tags.

    Usa GET /api/tags, o mesmo que o comando `ollama list` consulta.
    Se o Ollama não responder, levanta ErroPipeline explicando o provável motivo.
    """
    try:
        resposta = requests.get(f"{url}/api/tags", timeout=timeout_s)
        resposta.raise_for_status()  # transforma códigos de erro HTTP (404, 500...) em exceção
        dados = resposta.json()      # converte o texto JSON em dicionário Python
    except requests.exceptions.ConnectionError as erro:
        # ConnectionError = ninguém está escutando no endereço. Pode ser "não instalado"
        # ou "instalado, mas não iniciado". shutil.which procura o comando no PATH do Windows.
        if shutil.which("ollama") is None:
            raise ErroPipeline(
                "O Ollama não parece estar instalado (o comando 'ollama' não foi encontrado). "
                "Instale em https://ollama.com/download e reinicie o VS Code para atualizar o PATH."
            ) from erro
        raise ErroPipeline(
            f"O Ollama está instalado, mas o serviço não respondeu em {url}. "
            "Abra o aplicativo Ollama ou execute 'ollama serve' em um terminal separado."
        ) from erro
    except requests.exceptions.Timeout as erro:
        raise ErroPipeline(f"O Ollama não respondeu em {timeout_s} s ao listar os modelos.") from erro
    except (requests.exceptions.RequestException, ValueError) as erro:
        raise ErroPipeline(f"Resposta inesperada do Ollama ao listar os modelos: {erro}") from erro

    # dados["models"] é uma lista de dicionários; de cada um queremos só o campo "name" (a tag).
    return [modelo["name"] for modelo in dados.get("models", [])]


def verificar_modelos(modelos_desejados, modelos_instalados):
    """Compara os modelos pedidos com os instalados, SEM alterar nenhum nome.

    Devolve (disponiveis, ausentes):
      - disponiveis: lista dos nomes pedidos que existem exatamente iguais no Ollama;
      - ausentes: dicionário {nome pedido: [tags parecidas instaladas]} para ajudar a achar
        a tag correta. Só sugerimos; nunca trocamos o nome por conta própria.
    """
    disponiveis = []
    ausentes = {}
    for modelo in modelos_desejados:
        if modelo in modelos_instalados:
            disponiveis.append(modelo)
            continue
        # Sugestão 1: mesma família (parte antes dos dois-pontos), por exemplo "llama3.1:latest".
        familia = modelo.split(":")[0]
        parecidos = [m for m in modelos_instalados if m.split(":")[0] == familia]
        # Sugestão 2: nomes escritos de forma parecida (ajuda quando há erro de digitação).
        if not parecidos:
            parecidos = difflib.get_close_matches(modelo, modelos_instalados, n=3, cutoff=0.5)
        ausentes[modelo] = parecidos
    return disponiveis, ausentes


# ---------------------------------------------------------------------------
# 2) Leitura e validação dos prompts (inputs.json)
# ---------------------------------------------------------------------------

def criar_slug(texto):
    """Transforma um texto em um nome seguro para arquivos/pastas.

    Exemplos: "qwen3.5:9b" -> "qwen3_5_9b" ; "PROMPT-001" -> "prompt_001".
    O Windows não aceita ":" em nomes de arquivo, por isso precisamos dessa troca.
    """
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def carregar_prompts(caminho):
    """Lê o arquivo JSON e devolve a lista de prompts já validada.

    Estrutura esperada: uma LISTA de objetos, cada um com "id", "titulo" e "texto".
    No Python, isso vira uma list de dict:
        [{"id": "PROMPT-001", "titulo": "...", "texto": "..."}, ...]

    O texto de cada prompt é devolvido exatamente como está no arquivo (sem strip, sem
    reescrita), pois o experimento exige enviar o prompt original aos modelos.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise ErroPipeline(f"Arquivo de prompts não encontrado: {caminho}")

    try:
        # "utf-8-sig" lê UTF-8 normal e também ignora o BOM que alguns editores do Windows
        # inserem no início do arquivo. Sem isso, o json.load falharia nesses arquivos.
        with open(caminho, "r", encoding="utf-8-sig") as arquivo:
            prompts = json.load(arquivo)  # lê o arquivo inteiro e converte para list/dict
    except json.JSONDecodeError as erro:
        raise ErroPipeline(
            f"JSON inválido em {caminho.name}: {erro.msg} (linha {erro.lineno}, coluna {erro.colno}). "
            "Verifique vírgulas, aspas e colchetes."
        ) from erro
    except UnicodeDecodeError as erro:
        raise ErroPipeline(f"{caminho.name} precisa estar salvo com codificação UTF-8.") from erro
    except OSError as erro:
        raise ErroPipeline(f"Não foi possível abrir {caminho}: {erro}") from erro

    if not isinstance(prompts, list) or not prompts:
        raise ErroPipeline(f"{caminho.name} deve conter uma lista não vazia de prompts ([ {{...}}, {{...}} ]).")

    # Em vez de parar no primeiro defeito, juntamos todos os problemas para mostrar de uma vez.
    problemas = []
    ids_vistos = set()
    pastas_vistas = {}  # nome da pasta -> id que a usou; evita dois prompts na mesma pasta
    for posicao, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, dict):
            problemas.append(f"item {posicao}: deveria ser um objeto {{...}}")
            continue
        for campo in ("id", "titulo", "texto"):
            valor = prompt.get(campo)
            if not isinstance(valor, str) or not valor.strip():
                problemas.append(f"item {posicao}: o campo '{campo}' está ausente ou vazio")
        identificador = prompt.get("id")
        if isinstance(identificador, str) and identificador.strip():
            if identificador in ids_vistos:
                problemas.append(f"item {posicao}: o id '{identificador}' está repetido")
            ids_vistos.add(identificador)
            pasta = criar_slug(identificador)
            if not pasta:
                problemas.append(f"item {posicao}: o id '{identificador}' não tem letras ou números")
            elif pasta in pastas_vistas and pastas_vistas[pasta] != identificador:
                problemas.append(f"item {posicao}: o id '{identificador}' gera a mesma pasta de '{pastas_vistas[pasta]}'")
            pastas_vistas.setdefault(pasta, identificador)
    if problemas:
        raise ErroPipeline(f"Problemas em {caminho.name}:\n- " + "\n- ".join(problemas))

    return prompts


# ---------------------------------------------------------------------------
# 3) Identificadores
# ---------------------------------------------------------------------------

def gerar_id_lote():
    """Identifica uma rodada completa do experimento pelo instante em que começou.

    Exemplo: "20261028-211530". Como o horário nunca se repete, rodadas diferentes
    recebem identificadores diferentes e nunca se misturam.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def gerar_id_execucao(id_lote, numero):
    """Monta o ID único de uma execução: lote + contador. Ex.: EXEC-20261028-211530-003."""
    return f"EXEC-{id_lote}-{numero:03d}"


# ---------------------------------------------------------------------------
# 4) Execução de um prompt em um modelo
# ---------------------------------------------------------------------------

def _interpretar_resposta_http(resposta_http, modelo):
    """Converte a resposta HTTP do Ollama em dicionário ou levanta ErroPipeline clara."""
    if resposta_http.status_code != 200:
        try:
            corpo = resposta_http.json()
            detalhe = corpo.get("error", resposta_http.text) if isinstance(corpo, dict) else resposta_http.text
        except ValueError:
            detalhe = resposta_http.text
        # O Ollama responde 404 "model ... not found" quando a tag não está instalada.
        if resposta_http.status_code == 404 and "not found" in str(detalhe).lower():
            raise ErroPipeline(
                f"Modelo '{modelo}' não encontrado no Ollama. Confira a tag com 'ollama list' "
                f"e instale com 'ollama pull {modelo}'."
            )
        raise ErroPipeline(f"O Ollama respondeu com erro HTTP {resposta_http.status_code}: {detalhe}")

    try:
        dados = resposta_http.json()
    except ValueError as erro:
        raise ErroPipeline(f"A resposta do Ollama não é um JSON válido: {resposta_http.text[:200]!r}") from erro
    if not isinstance(dados, dict):
        raise ErroPipeline("A resposta do Ollama tem um formato inesperado.")
    if "error" in dados:
        raise ErroPipeline(f"O Ollama informou um erro durante a geração: {dados['error']}")
    return dados


def _nanos_para_segundos(valor):
    """O Ollama informa durações em nanossegundos; convertemos para segundos."""
    if isinstance(valor, (int, float)):
        return round(valor / 1e9, 3)
    return ""  # se o Ollama não informou, deixamos em branco: não inventamos dados


def executar_prompt_em_modelo(id_prompt, texto_prompt, modelo,
                              url=URL_PADRAO_OLLAMA, opcoes=None, timeout_s=600):
    """Envia UM prompt a UM modelo e devolve um dicionário com tudo o que aconteceu.

    Entradas (argumentos):
      id_prompt    - identificador do prompt (só para registrar no resultado);
      texto_prompt - texto integral enviado ao modelo, sem nenhuma alteração;
      modelo       - tag exata do modelo no Ollama (ex.: "llama3.1:8b");
      opcoes       - parâmetros de geração (ex.: {"temperature": 0, "seed": 42});
      timeout_s    - tempo máximo de espera pela resposta, em segundos.

    Saída (retorno): dicionário com status "sucesso" ou "falha". A função NÃO levanta erro
    para problemas esperados (modelo ausente, timeout...): ela os registra em
    "mensagem_erro". Assim, quem chama (o laço) pode seguir para a próxima execução.
    """
    opcoes = dict(opcoes or {})

    # O dicionário começa como "falha"; só viramos "sucesso" depois de receber uma resposta
    # não vazia. Assim, nenhum caminho de erro é registrado como sucesso por engano.
    resultado = {
        "id_prompt": id_prompt,
        "modelo": modelo,
        "data_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "falha",
        "tempo_execucao_s": "",
        "mensagem_erro": "",
        "tokens_prompt": "",
        "tokens_gerados": "",
        "duracao_ollama_s": "",
        "motivo_termino": "",
        "parametros": json.dumps({"options": opcoes, "timeout_s": timeout_s}, ensure_ascii=False),
        "texto_prompt": texto_prompt,
        "resposta": "",
        "raciocinio": "",
    }

    # Corpo da requisição enviada ao Ollama. "stream": False faz o Ollama devolver a resposta
    # inteira de uma vez (em vez de pedaço por pedaço), o que simplifica o código.
    corpo = {"model": modelo, "prompt": texto_prompt, "stream": False, "options": opcoes}

    # perf_counter é um cronômetro de alta precisão feito para medir intervalos
    # (diferente de datetime.now(), que pode ser ajustado pelo relógio do sistema).
    inicio = time.perf_counter()
    try:
        # timeout=(10, timeout_s): até 10 s para conectar e timeout_s para receber a resposta.
        # Como não usamos streaming, o Ollama só envia dados quando termina de gerar.
        resposta_http = requests.post(f"{url}/api/generate", json=corpo, timeout=(10, timeout_s))
        dados = _interpretar_resposta_http(resposta_http, modelo)

        resposta_modelo = dados.get("response") or ""
        resultado["resposta"] = resposta_modelo
        # Modelos de raciocínio (deepseek-r1, qwen3...) podem devolver o "pensamento" separado.
        resultado["raciocinio"] = dados.get("thinking") or ""
        # Metadados que o Ollama devolve; se faltar algum, ficam em branco.
        resultado["tokens_prompt"] = dados.get("prompt_eval_count", "")
        resultado["tokens_gerados"] = dados.get("eval_count", "")
        resultado["duracao_ollama_s"] = _nanos_para_segundos(dados.get("total_duration"))
        resultado["motivo_termino"] = dados.get("done_reason", "")

        if not resposta_modelo.strip():
            resultado["mensagem_erro"] = (
                "O modelo devolveu uma resposta vazia "
                f"(motivo de término: {resultado['motivo_termino'] or 'não informado'}). "
                "Modelos de raciocínio podem gastar todos os tokens 'pensando'."
            )
        else:
            resultado["status"] = "sucesso"

    except requests.exceptions.ConnectionError:
        resultado["mensagem_erro"] = (
            f"Não foi possível conectar ao Ollama em {url}. Verifique se o serviço está iniciado."
        )
    except requests.exceptions.Timeout:
        resultado["mensagem_erro"] = (
            f"Tempo limite excedido: o modelo não respondeu em {timeout_s} s. "
            "Aumente TIMEOUT_S no notebook ou use um modelo menor."
        )
    except ErroPipeline as erro:
        resultado["mensagem_erro"] = str(erro)
    except requests.exceptions.RequestException as erro:
        resultado["mensagem_erro"] = f"Falha na comunicação com a API do Ollama: {erro}"

    # Este trecho roda em todos os casos (sucesso ou falha): registra quanto tempo passou.
    resultado["tempo_execucao_s"] = round(time.perf_counter() - inicio, 3)
    return resultado


# ---------------------------------------------------------------------------
# 5) Armazenamento: Markdown individual e CSV consolidado
# ---------------------------------------------------------------------------

def _cerca_de_codigo(texto):
    """Escolhe uma cerca de crases (```) maior que qualquer sequência de crases no texto.

    Assim, um prompt que contenha blocos de código não "quebra" o formato do Markdown.
    """
    maior_sequencia = max((len(s) for s in re.findall(r"`+", texto)), default=0)
    return "`" * max(3, maior_sequencia + 1)


def montar_markdown(resultado):
    """Monta o texto do arquivo Markdown de uma execução a partir do dicionário de resultado."""
    parametros = json.loads(resultado.get("parametros") or "{}")
    linhas_parametros = [f"- {nome}: {valor}" for nome, valor in parametros.get("options", {}).items()]
    linhas_parametros.append(f"- timeout (s): {parametros.get('timeout_s', 'não informado')}")

    def ou_nao_informado(valor):
        return valor if valor != "" else "não informado"

    cerca = _cerca_de_codigo(resultado["texto_prompt"])
    partes = [
        "# Resultado da execução",
        "",
        "## Identificação",
        "",
        f"- ID da execução: {resultado['id_execucao']}",
        f"- ID do prompt: {resultado['id_prompt']}",
        f"- Título do prompt: {resultado['titulo_prompt']}",
        f"- Modelo (tag exata): `{resultado['modelo']}`",
        f"- Data e hora: {resultado['data_hora']}",
        f"- Status: {resultado['status']}",
        f"- Tempo de execução: {resultado['tempo_execucao_s']} s",
        "",
        "## Parâmetros de geração",
        "",
        *linhas_parametros,
        "",
        "## Metadados informados pelo Ollama",
        "",
        f"- Tokens do prompt: {ou_nao_informado(resultado['tokens_prompt'])}",
        f"- Tokens gerados: {ou_nao_informado(resultado['tokens_gerados'])}",
        f"- Duração total no Ollama (s): {ou_nao_informado(resultado['duracao_ollama_s'])}",
        f"- Motivo de término: {ou_nao_informado(resultado['motivo_termino'])}",
        "",
        "## Input",
        "",
        cerca,
        resultado["texto_prompt"],
        cerca,
        "",
    ]
    if resultado["raciocinio"]:
        partes += ["## Raciocínio do modelo (thinking)", "", resultado["raciocinio"], ""]
    if resultado["status"] == "sucesso":
        partes += ["## Output", "", resultado["resposta"], ""]
    else:
        partes += ["## Erro", "", resultado["mensagem_erro"], ""]
        if resultado["resposta"]:
            partes += ["## Output parcial", "", resultado["resposta"], ""]
    return "\n".join(partes)


def salvar_markdown(resultado, pasta_saida):
    """Grava o Markdown da execução em outputs/<prompt>/<modelo>__<id_execucao>.md.

    Devolve o caminho relativo à pasta_saida (é o que vai para o CSV).

    Proteção contra sobrescrita: o modo "x" do open() só cria arquivos NOVOS; se o arquivo
    já existir, o Python levanta FileExistsError em vez de apagar o conteúdo antigo.
    """
    pasta_saida = Path(pasta_saida)
    pasta_prompt = pasta_saida / criar_slug(resultado["id_prompt"])
    pasta_prompt.mkdir(parents=True, exist_ok=True)  # cria a pasta se necessário; não falha se já existir

    nome_arquivo = f"{criar_slug(resultado['modelo'])}__{resultado['id_execucao']}.md"
    caminho = pasta_prompt / nome_arquivo

    # newline="\n" evita que o Windows converta as quebras de linha do texto original.
    with open(caminho, "x", encoding="utf-8", newline="\n") as arquivo:
        arquivo.write(montar_markdown(resultado))
    return caminho.relative_to(pasta_saida).as_posix()


def registrar_no_csv(resultado, caminho_csv):
    """Acrescenta UMA linha ao CSV consolidado (cria o arquivo com cabeçalho se não existir).

    O modo "a" (append) adiciona ao final, preservando as linhas anteriores.
    Usamos "utf-8-sig" para que o Excel no Windows mostre os acentos corretamente;
    o Python e outras ferramentas leem esse formato normalmente.
    """
    caminho_csv = Path(caminho_csv)
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    arquivo_novo = not caminho_csv.exists() or caminho_csv.stat().st_size == 0

    linha = dict(resultado)
    # Quebras de linha dentro de uma célula dificultam a leitura do CSV; achatamos a mensagem.
    linha["mensagem_erro"] = " ".join(str(linha.get("mensagem_erro", "")).split())

    with open(caminho_csv, "a", newline="", encoding="utf-8-sig") as arquivo:
        # extrasaction="ignore": campos extras do dicionário (texto, resposta...) não vão ao CSV.
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_CSV, extrasaction="ignore")
        if arquivo_novo:
            escritor.writeheader()
        escritor.writerow(linha)


def ler_resultados_csv(caminho_csv):
    """Lê o CSV consolidado e devolve uma lista de dicionários (uma por execução)."""
    caminho_csv = Path(caminho_csv)
    if not caminho_csv.exists():
        return []
    with open(caminho_csv, "r", newline="", encoding="utf-8-sig") as arquivo:
        return list(csv.DictReader(arquivo))


# ---------------------------------------------------------------------------
# 6) Coordenação do experimento
# ---------------------------------------------------------------------------

def executar_tarefa(tarefa, url=URL_PADRAO_OLLAMA, opcoes=None, timeout_s=600):
    """Executa UMA tarefa (um prompt x um modelo). Não escreve nada em disco.

    Esta função foi separada de propósito: ela só depende da tarefa recebida. No futuro, para
    rodar em paralelo, bastaria aplicá-la a todas as tarefas com concurrent.futures
    (ThreadPoolExecutor) e continuar gravando os resultados em sequência.
    """
    prompt = tarefa["prompt"]
    resultado = executar_prompt_em_modelo(
        prompt["id"], prompt["texto"], tarefa["modelo"], url=url, opcoes=opcoes, timeout_s=timeout_s
    )
    resultado["id_execucao"] = tarefa["id_execucao"]
    resultado["titulo_prompt"] = prompt["titulo"]
    return resultado


def guardar_resultado_markdown(resultado, pasta_saida):
    """Salva o Markdown e preenche o campo "arquivo_markdown" do resultado.

    Se o arquivo não puder ser salvo, o resultado vira "falha" (a resposta continua na
    memória do notebook, mas não está em disco, então não pode constar como sucesso).
    """
    try:
        resultado["arquivo_markdown"] = salvar_markdown(resultado, pasta_saida)
    except OSError as erro:  # FileExistsError, PermissionError etc. são subtipos de OSError
        resultado["arquivo_markdown"] = ""
        aviso = f"Não foi possível salvar o arquivo Markdown: {erro}"
        if resultado["status"] == "sucesso":
            resultado["status"] = "falha"
            resultado["mensagem_erro"] = "Resposta gerada, mas " + aviso[0].lower() + aviso[1:]
        else:
            resultado["mensagem_erro"] = f"{resultado['mensagem_erro']} | {aviso}"


def gravar_pendentes_no_csv(pendentes, caminho_csv):
    """Tenta gravar no CSV as linhas ainda não gravadas, na ordem.

    Caso comum no Windows: o CSV está aberto no Excel e fica bloqueado (PermissionError).
    Então não perdemos nada: as linhas ficam na lista "pendentes" e a gravação é tentada
    de novo após a próxima execução (e no final).
    """
    while pendentes:
        try:
            registrar_no_csv(pendentes[0], caminho_csv)
        except OSError as erro:
            print(f"   AVISO: não consegui gravar no CSV ({erro}). Feche o arquivo se estiver aberto no Excel; "
                  f"{len(pendentes)} linha(s) aguardam gravação.", flush=True)
            return
        pendentes.pop(0)


def processar_experimentos(prompts, modelos, pasta_saida, url=URL_PADRAO_OLLAMA,
                           opcoes=None, timeout_s=600, id_lote=None):
    """Executa TODOS os prompts em TODOS os modelos, um de cada vez, e grava os resultados.

    Ordem: para cada prompt, percorre todos os modelos. Se houver 5 prompts e 3 modelos,
    são 15 execuções independentes, cada uma com o texto original do prompt.
    Falhas são registradas e o laço continua; nada interrompe as demais execuções.
    """
    id_lote = id_lote or gerar_id_lote()
    pasta_saida = Path(pasta_saida)
    caminho_csv = pasta_saida / "resultados.csv"

    # Primeiro montamos a lista de tarefas (cada uma é um dicionário). Dois "for" aninhados
    # geram todas as combinações prompt x modelo, e o contador dá o número de cada execução.
    tarefas = []
    numero = 0
    for prompt in prompts:
        for modelo in modelos:
            numero += 1
            tarefas.append({
                "id_execucao": gerar_id_execucao(id_lote, numero),
                "prompt": prompt,
                "modelo": modelo,
            })

    total = len(tarefas)
    print(f"Lote {id_lote}: {len(prompts)} prompt(s) x {len(modelos)} modelo(s) = {total} execução(ões).", flush=True)

    resultados = []
    pendentes = []  # linhas que ainda não conseguiram entrar no CSV
    for indice, tarefa in enumerate(tarefas, start=1):
        print(f"[{indice}/{total}] {tarefa['prompt']['id']} x {tarefa['modelo']} ... (aguardando resposta)", flush=True)

        resultado = executar_tarefa(tarefa, url=url, opcoes=opcoes, timeout_s=timeout_s)
        guardar_resultado_markdown(resultado, pasta_saida)

        # Gravamos já no CSV, a cada execução: se você interromper o notebook no meio,
        # tudo o que já foi concluído continua salvo.
        pendentes.append(resultado)
        gravar_pendentes_no_csv(pendentes, caminho_csv)

        resultados.append(resultado)
        detalhe = "" if resultado["status"] == "sucesso" else f" - {resultado['mensagem_erro']}"
        print(f"   -> {resultado['status']} em {resultado['tempo_execucao_s']} s{detalhe}", flush=True)

    gravar_pendentes_no_csv(pendentes, caminho_csv)  # última tentativa
    if pendentes:
        ids = ", ".join(r["id_execucao"] for r in pendentes)
        print(f"ATENÇÃO: estas execuções NÃO entraram no CSV (os .md foram salvos): {ids}")
    return resultados


# ---------------------------------------------------------------------------
# 7) Consulta resumida dos resultados
# ---------------------------------------------------------------------------

def resumir_por_modelo(linhas):
    """Conta execuções, sucessos e falhas por modelo e calcula o tempo médio dos sucessos."""
    resumo = {}
    for linha in linhas:
        item = resumo.setdefault(linha["modelo"], {
            "modelo": linha["modelo"], "execucoes": 0, "sucessos": 0, "falhas": 0, "_tempos": [],
        })
        item["execucoes"] += 1
        if linha["status"] == "sucesso":
            item["sucessos"] += 1
            try:
                item["_tempos"].append(float(linha["tempo_execucao_s"]))
            except (TypeError, ValueError):
                pass
        else:
            item["falhas"] += 1

    tabela = []
    for item in resumo.values():
        tempos = item.pop("_tempos")
        item["tempo_medio_s"] = round(sum(tempos) / len(tempos), 2) if tempos else ""
        tabela.append(item)
    return tabela


def mostrar_tabela(linhas, colunas):
    """Imprime uma lista de dicionários como tabela de texto alinhada (sem usar pandas)."""
    if not linhas:
        print("(nenhum registro)")
        return
    larguras = {c: max(len(c), *(len(str(l.get(c, ""))) for l in linhas)) for c in colunas}
    print(" | ".join(c.ljust(larguras[c]) for c in colunas))
    print("-+-".join("-" * larguras[c] for c in colunas))
    for linha in linhas:
        print(" | ".join(str(linha.get(c, "")).ljust(larguras[c]) for c in colunas))
