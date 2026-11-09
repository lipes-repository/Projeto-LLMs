"""
Testes automáticos do pipeline (usam apenas a biblioteca padrão: unittest).

IMPORTANTE: estes testes NÃO usam o Ollama real nem modelos reais.
  - As funções de armazenamento são testadas com resultados FICTÍCIOS.
  - A comunicação HTTP é testada contra um servidor falso (FakeOllama), que imita as
    respostas da API do Ollama (sucesso, modelo inexistente, resposta vazia, timeout...).
Isso valida a NOSSA lógica, mas não prova que os modelos reais funcionam: para isso,
execute o notebook com o Ollama instalado.

Como rodar (na pasta do projeto):
    python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))  # permite "import pipeline" mesmo rodando de outra pasta

import pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Servidor falso que imita a API do Ollama
# ---------------------------------------------------------------------------

class ManipuladorFalso(BaseHTTPRequestHandler):
    """Responde como o Ollama, escolhendo o comportamento pelo nome do modelo recebido."""

    def log_message(self, *args):  # silencia o log no console
        pass

    def _responder(self, codigo, corpo, tipo="application/json"):
        dados = corpo.encode("utf-8") if isinstance(corpo, str) else json.dumps(corpo).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        try:
            self.wfile.write(dados)
        except (BrokenPipeError, ConnectionResetError):
            pass  # o cliente desistiu (ex.: teste de timeout); não é erro do teste

    def do_GET(self):
        if self.path == "/api/tags":
            self._responder(200, {"models": [{"name": n} for n in self.server.modelos_instalados]})
        else:
            self._responder(404, {"error": "rota desconhecida"})

    def do_POST(self):
        tamanho = int(self.headers.get("Content-Length", 0))
        pedido = json.loads(self.rfile.read(tamanho).decode("utf-8"))
        self.server.pedidos.append(pedido)  # guardamos o que chegou para conferir nos testes
        modelo = pedido["model"]

        if modelo.startswith("inexistente"):
            self._responder(404, {"error": f"model '{modelo}' not found"})
        elif modelo.startswith("quebrado"):
            self._responder(200, "<html>isto não é JSON</html>", tipo="text/html")
        elif modelo.startswith("lento"):
            time.sleep(2.5)
            self._responder(200, {"response": "tarde demais", "done": True})
        elif modelo.startswith("vazio"):
            self._responder(200, {"response": "", "done": True, "done_reason": "length"})
        else:
            self._responder(200, {
                "model": modelo,
                "response": f"Resposta fictícia de {modelo} para: {pedido['prompt']}",
                "thinking": "raciocínio fictício" if modelo.startswith("pensa") else "",
                "done": True,
                "done_reason": "stop",
                "total_duration": 1_500_000_000,
                "prompt_eval_count": 12,
                "eval_count": 34,
            })


class FakeOllama:
    """Sobe o servidor falso em uma porta livre, em uma thread separada."""

    def __init__(self, modelos_instalados=("fake-a:1b", "fake-b:1b")):
        self.servidor = ThreadingHTTPServer(("127.0.0.1", 0), ManipuladorFalso)
        self.servidor.daemon_threads = True
        self.servidor.modelos_instalados = list(modelos_instalados)
        self.servidor.pedidos = []
        self.url = f"http://127.0.0.1:{self.servidor.server_address[1]}"
        self.pedidos = self.servidor.pedidos
        threading.Thread(target=self.servidor.serve_forever, daemon=True).start()

    def parar(self):
        self.servidor.shutdown()
        self.servidor.server_close()


def url_sem_servidor():
    """Devolve uma URL local onde, com certeza, ninguém está escutando."""
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), ManipuladorFalso)
    porta = servidor.server_address[1]
    servidor.server_close()
    return f"http://127.0.0.1:{porta}"


def resultado_ficticio(**mudancas):
    """Resultado de execução inventado, só para testar o armazenamento."""
    resultado = {
        "id_execucao": "EXEC-TESTE-001",
        "id_prompt": "PROMPT-001",
        "titulo_prompt": "Título de teste com acentuação: coração, ação, você",
        "modelo": "qwen3.5:9b",
        "data_hora": "2026-01-01T10:00:00-03:00",
        "status": "sucesso",
        "tempo_execucao_s": 1.234,
        "arquivo_markdown": "",
        "mensagem_erro": "",
        "tokens_prompt": 10,
        "tokens_gerados": 20,
        "duracao_ollama_s": 1.2,
        "motivo_termino": "stop",
        "parametros": json.dumps({"options": {"temperature": 0, "seed": 42}, "timeout_s": 600}),
        "texto_prompt": "Texto do prompt\ncom duas linhas",
        "resposta": "Resposta fictícia com acentos: ç ã é",
        "raciocinio": "",
    }
    resultado.update(mudancas)
    return resultado


class TesteComPastaTemporaria(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pasta = Path(self._tmp.name)


# ---------------------------------------------------------------------------
# Leitura e validação dos prompts
# ---------------------------------------------------------------------------

class TesteCarregarPrompts(TesteComPastaTemporaria):
    def escrever(self, conteudo, nome="entrada.json", codificacao="utf-8"):
        caminho = self.pasta / nome
        caminho.write_text(conteudo, encoding=codificacao)
        return caminho

    def test_inputs_json_do_projeto_e_valido(self):
        prompts = pipeline.carregar_prompts(RAIZ / "inputs.json")
        self.assertGreaterEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertEqual(set(prompt), {"id", "titulo", "texto"})

    def test_texto_do_prompt_e_preservado_exatamente(self):
        texto = "  Linha 1 com espaços  \n\nLinha 3: ação, coração!  \n"
        caminho = self.escrever(json.dumps([{"id": "A", "titulo": "T", "texto": texto}], ensure_ascii=False))
        self.assertEqual(pipeline.carregar_prompts(caminho)[0]["texto"], texto)

    def test_arquivo_com_bom_do_windows_e_aceito(self):
        caminho = self.escrever('[{"id": "A", "titulo": "T", "texto": "x"}]', codificacao="utf-8-sig")
        self.assertEqual(len(pipeline.carregar_prompts(caminho)), 1)

    def test_arquivo_inexistente(self):
        with self.assertRaisesRegex(pipeline.ErroPipeline, "não encontrado"):
            pipeline.carregar_prompts(self.pasta / "nao_existe.json")

    def test_json_invalido_informa_linha(self):
        caminho = self.escrever('[\n {"id": "A", "titulo": "T" "texto": "x"}\n]')
        with self.assertRaisesRegex(pipeline.ErroPipeline, "JSON inválido.*linha 2"):
            pipeline.carregar_prompts(caminho)

    def test_estrutura_que_nao_e_lista(self):
        caminho = self.escrever('{"id": "A"}')
        with self.assertRaises(pipeline.ErroPipeline):
            pipeline.carregar_prompts(caminho)

    def test_lista_vazia(self):
        with self.assertRaises(pipeline.ErroPipeline):
            pipeline.carregar_prompts(self.escrever("[]"))

    def test_campos_ausentes_ou_vazios(self):
        caminho = self.escrever('[{"id": "A", "titulo": "", "texto": "x"}, {"id": "B", "titulo": "T"}]')
        with self.assertRaises(pipeline.ErroPipeline) as contexto:
            pipeline.carregar_prompts(caminho)
        mensagem = str(contexto.exception)
        self.assertIn("'titulo'", mensagem)
        self.assertIn("'texto'", mensagem)

    def test_ids_repetidos(self):
        caminho = self.escrever('[{"id":"A","titulo":"T","texto":"x"},{"id":"A","titulo":"T","texto":"y"}]')
        with self.assertRaisesRegex(pipeline.ErroPipeline, "repetido"):
            pipeline.carregar_prompts(caminho)

    def test_ids_diferentes_que_geram_a_mesma_pasta(self):
        caminho = self.escrever('[{"id":"A-1","titulo":"T","texto":"x"},{"id":"A_1","titulo":"T","texto":"y"}]')
        with self.assertRaisesRegex(pipeline.ErroPipeline, "mesma pasta"):
            pipeline.carregar_prompts(caminho)


# ---------------------------------------------------------------------------
# Identificadores
# ---------------------------------------------------------------------------

class TesteIdentificadores(unittest.TestCase):
    def test_criar_slug(self):
        self.assertEqual(pipeline.criar_slug("qwen3.5:9b"), "qwen3_5_9b")
        self.assertEqual(pipeline.criar_slug("deepseek-r1:8b"), "deepseek_r1_8b")
        self.assertEqual(pipeline.criar_slug("llama3.1:8b"), "llama3_1_8b")
        self.assertEqual(pipeline.criar_slug("PROMPT-001"), "prompt_001")

    def test_id_de_execucao_tem_formato_e_e_unico_no_lote(self):
        ids = [pipeline.gerar_id_execucao("20260101-100000", n) for n in range(1, 51)]
        self.assertEqual(ids[0], "EXEC-20260101-100000-001")
        self.assertEqual(len(set(ids)), 50)

    def test_id_de_lote(self):
        self.assertRegex(pipeline.gerar_id_lote(), r"^\d{8}-\d{6}$")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class TesteMarkdown(TesteComPastaTemporaria):
    def test_cria_pastas_e_arquivo_com_conteudo_esperado(self):
        saida = self.pasta / "outputs"  # a pasta ainda não existe
        caminho_relativo = pipeline.salvar_markdown(resultado_ficticio(), saida)

        self.assertEqual(caminho_relativo, "prompt_001/qwen3_5_9b__EXEC-TESTE-001.md")
        conteudo = (saida / caminho_relativo).read_text(encoding="utf-8")
        for trecho in ("# Resultado da execução", "## Identificação", "EXEC-TESTE-001", "PROMPT-001",
                       "`qwen3.5:9b`", "2026-01-01T10:00:00-03:00", "Status: sucesso", "1.234 s",
                       "temperature: 0", "## Input", "Texto do prompt\ncom duas linhas",
                       "## Output", "Resposta fictícia com acentos: ç ã é", "coração"):
            self.assertIn(trecho, conteudo)
        self.assertNotIn("## Erro", conteudo)

    def test_execucao_com_falha_gera_secao_de_erro(self):
        resultado = resultado_ficticio(status="falha", resposta="", mensagem_erro="Modelo ausente")
        caminho = pipeline.salvar_markdown(resultado, self.pasta)
        conteudo = (self.pasta / caminho).read_text(encoding="utf-8")
        self.assertIn("Status: falha", conteudo)
        self.assertIn("## Erro", conteudo)
        self.assertIn("Modelo ausente", conteudo)
        self.assertNotIn("## Output", conteudo)

    def test_nao_sobrescreve_arquivo_existente(self):
        pipeline.salvar_markdown(resultado_ficticio(), self.pasta)
        with self.assertRaises(FileExistsError):
            pipeline.salvar_markdown(resultado_ficticio(resposta="OUTRA"), self.pasta)
        arquivo = self.pasta / "prompt_001" / "qwen3_5_9b__EXEC-TESTE-001.md"
        self.assertIn("Resposta fictícia", arquivo.read_text(encoding="utf-8"))

    def test_prompt_com_blocos_de_codigo_nao_quebra_a_cerca(self):
        texto = "Veja:\n```python\nprint(1)\n```"
        conteudo = pipeline.montar_markdown(resultado_ficticio(texto_prompt=texto))
        self.assertIn("````\n" + texto + "\n````", conteudo)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TesteCsv(TesteComPastaTemporaria):
    def test_cabecalho_unico_linhas_acumuladas_e_acentos(self):
        caminho = self.pasta / "sub" / "resultados.csv"
        pipeline.registrar_no_csv(resultado_ficticio(id_execucao="EXEC-1"), caminho)
        pipeline.registrar_no_csv(resultado_ficticio(id_execucao="EXEC-2", status="falha",
                                                      mensagem_erro="linha 1\nlinha 2"), caminho)

        linhas = pipeline.ler_resultados_csv(caminho)
        self.assertEqual([l["id_execucao"] for l in linhas], ["EXEC-1", "EXEC-2"])
        self.assertEqual(linhas[0]["titulo_prompt"], "Título de teste com acentuação: coração, ação, você")
        self.assertEqual(linhas[1]["mensagem_erro"], "linha 1 linha 2")
        self.assertNotIn("resposta", linhas[0])  # a resposta fica só no Markdown

        bruto = caminho.read_bytes()
        self.assertEqual(bruto.count(b"\xef\xbb\xbf"), 1)  # BOM só no início
        self.assertEqual(bruto.decode("utf-8-sig").count("id_execucao,id_prompt"), 1)  # um cabeçalho

    def test_colunas_obrigatorias_existem(self):
        for coluna in ("id_execucao", "id_prompt", "titulo_prompt", "modelo", "data_hora", "status",
                       "tempo_execucao_s", "arquivo_markdown", "mensagem_erro"):
            self.assertIn(coluna, pipeline.COLUNAS_CSV)


# ---------------------------------------------------------------------------
# Verificação de modelos
# ---------------------------------------------------------------------------

class TesteModelos(unittest.TestCase):
    def test_listar_modelos_instalados(self):
        servidor = FakeOllama(["fake-a:1b", "llama3.1:latest"])
        self.addCleanup(servidor.parar)
        self.assertEqual(pipeline.listar_modelos_instalados(servidor.url), ["fake-a:1b", "llama3.1:latest"])

    def test_ollama_fora_do_ar(self):
        with self.assertRaises(pipeline.ErroPipeline) as contexto:
            pipeline.listar_modelos_instalados(url_sem_servidor(), timeout_s=2)
        self.assertRegex(str(contexto.exception), "Ollama")

    def test_verificar_modelos_nao_altera_nomes_e_sugere(self):
        instalados = ["llama3.1:latest", "deepseek-r1:8b"]
        disponiveis, ausentes = pipeline.verificar_modelos(
            ["llama3.1:8b", "deepseek-r1:8b", "qwen3.5:9b"], instalados)
        self.assertEqual(disponiveis, ["deepseek-r1:8b"])
        self.assertEqual(list(ausentes), ["llama3.1:8b", "qwen3.5:9b"])
        self.assertEqual(ausentes["llama3.1:8b"], ["llama3.1:latest"])  # sugestão, sem troca silenciosa
        self.assertEqual(ausentes["qwen3.5:9b"], [])


# ---------------------------------------------------------------------------
# Execução de um prompt (contra o servidor falso)
# ---------------------------------------------------------------------------

class TesteExecutarPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servidor = FakeOllama()

    @classmethod
    def tearDownClass(cls):
        cls.servidor.parar()

    def executar(self, modelo, texto="Olá, mundo!", **extra):
        return pipeline.executar_prompt_em_modelo(
            "PROMPT-001", texto, modelo, url=self.servidor.url,
            opcoes={"temperature": 0, "seed": 42}, **extra)

    def test_sucesso_envia_texto_original_e_registra_metadados(self):
        texto = "  Texto original\ncom  espaços e acentos: ção  "
        resultado = self.executar("fake-a:1b", texto)
        self.assertEqual(resultado["status"], "sucesso")
        self.assertEqual(resultado["resposta"], f"Resposta fictícia de fake-a:1b para: {texto}")
        self.assertEqual((resultado["tokens_prompt"], resultado["tokens_gerados"]), (12, 34))
        self.assertEqual(resultado["duracao_ollama_s"], 1.5)
        self.assertEqual(resultado["motivo_termino"], "stop")
        self.assertGreaterEqual(resultado["tempo_execucao_s"], 0)
        self.assertEqual(resultado["mensagem_erro"], "")

        enviado = self.servidor.pedidos[-1]
        self.assertEqual(enviado["prompt"], texto)  # prompt idêntico, sem modificação
        self.assertEqual(enviado["model"], "fake-a:1b")
        self.assertEqual(enviado["options"], {"temperature": 0, "seed": 42})
        self.assertIs(enviado["stream"], False)

    def test_raciocinio_e_guardado_separadamente(self):
        resultado = self.executar("pensa-bem:1b")
        self.assertEqual(resultado["raciocinio"], "raciocínio fictício")

    def test_modelo_inexistente(self):
        resultado = self.executar("inexistente:1b")
        self.assertEqual(resultado["status"], "falha")
        self.assertIn("ollama pull inexistente:1b", resultado["mensagem_erro"])

    def test_resposta_vazia_nao_e_sucesso(self):
        resultado = self.executar("vazio:1b")
        self.assertEqual(resultado["status"], "falha")
        self.assertIn("vazia", resultado["mensagem_erro"])

    def test_resposta_que_nao_e_json(self):
        resultado = self.executar("quebrado:1b")
        self.assertEqual(resultado["status"], "falha")
        self.assertIn("JSON", resultado["mensagem_erro"])

    def test_tempo_limite_excedido(self):
        resultado = self.executar("lento:1b", timeout_s=1)
        self.assertEqual(resultado["status"], "falha")
        self.assertIn("Tempo limite", resultado["mensagem_erro"])

    def test_servico_fora_do_ar_nao_levanta_excecao(self):
        resultado = pipeline.executar_prompt_em_modelo(
            "PROMPT-001", "x", "fake-a:1b", url=url_sem_servidor(), timeout_s=2)
        self.assertEqual(resultado["status"], "falha")
        self.assertIn("conectar", resultado["mensagem_erro"])


# ---------------------------------------------------------------------------
# Coordenação completa (servidor falso)
# ---------------------------------------------------------------------------

class TesteProcessarExperimentos(TesteComPastaTemporaria):
    PROMPTS = [
        {"id": "PROMPT-001", "titulo": "Primeiro", "texto": "Texto 1"},
        {"id": "PROMPT-002", "titulo": "Segundo", "texto": "Texto 2"},
    ]

    def setUp(self):
        super().setUp()
        self.servidor = FakeOllama()
        self.addCleanup(self.servidor.parar)
        self.saida = self.pasta / "outputs"

    def processar(self, modelos, **extra):
        return pipeline.processar_experimentos(
            self.PROMPTS, modelos, self.saida, url=self.servidor.url,
            opcoes={"temperature": 0}, timeout_s=5, **extra)

    def test_falha_em_um_modelo_nao_interrompe_os_demais(self):
        resultados = self.processar(["inexistente:1b", "fake-a:1b", "vazio:1b", "fake-b:1b"],
                                    id_lote="L1")
        self.assertEqual(len(resultados), 8)  # 2 prompts x 4 modelos
        status = [(r["modelo"], r["status"]) for r in resultados[:4]]
        self.assertEqual(status, [("inexistente:1b", "falha"), ("fake-a:1b", "sucesso"),
                                  ("vazio:1b", "falha"), ("fake-b:1b", "sucesso")])

        linhas = pipeline.ler_resultados_csv(self.saida / "resultados.csv")
        self.assertEqual(len(linhas), 8)
        self.assertEqual(len({l["id_execucao"] for l in linhas}), 8)
        for linha in linhas:
            self.assertTrue((self.saida / linha["arquivo_markdown"]).exists())  # md também para falhas
            if linha["status"] == "falha":
                self.assertTrue(linha["mensagem_erro"])

    def test_cada_modelo_recebe_o_mesmo_texto_em_requisicao_independente(self):
        self.processar(["fake-a:1b", "fake-b:1b"], id_lote="L1")
        pedidos = [(p["model"], p["prompt"]) for p in self.servidor.pedidos]
        self.assertEqual(pedidos, [("fake-a:1b", "Texto 1"), ("fake-b:1b", "Texto 1"),
                                   ("fake-a:1b", "Texto 2"), ("fake-b:1b", "Texto 2")])

    def test_novo_experimento_preserva_resultados_anteriores(self):
        self.processar(["fake-a:1b"], id_lote="L1")
        arquivos_antes = {p: p.read_bytes() for p in self.saida.rglob("*.md")}
        self.assertEqual(len(arquivos_antes), 2)

        self.processar(["fake-a:1b"], id_lote="L2")
        linhas = pipeline.ler_resultados_csv(self.saida / "resultados.csv")
        self.assertEqual(len(linhas), 4)
        for caminho, conteudo in arquivos_antes.items():
            self.assertEqual(caminho.read_bytes(), conteudo)  # intactos
        self.assertEqual(len(list(self.saida.rglob("*.md"))), 4)

    def test_colisao_de_id_nao_sobrescreve_e_registra_falha(self):
        self.processar(["fake-a:1b"], id_lote="L1")
        original = {p: p.read_bytes() for p in self.saida.rglob("*.md")}

        resultados = self.processar(["fake-a:1b"], id_lote="L1")  # mesmo lote de propósito
        self.assertTrue(all(r["status"] == "falha" for r in resultados))
        self.assertIn("salvar", resultados[0]["mensagem_erro"])
        for caminho, conteudo in original.items():
            self.assertEqual(caminho.read_bytes(), conteudo)

    def test_csv_bloqueado_nao_perde_linhas(self):
        real = pipeline.registrar_no_csv
        chamadas = {"n": 0}

        def csv_bloqueado_na_primeira_vez(resultado, caminho):
            chamadas["n"] += 1
            if chamadas["n"] == 1:
                raise PermissionError("arquivo aberto no Excel")
            return real(resultado, caminho)

        with mock.patch.object(pipeline, "registrar_no_csv", csv_bloqueado_na_primeira_vez):
            self.processar(["fake-a:1b"], id_lote="L1")
        linhas = pipeline.ler_resultados_csv(self.saida / "resultados.csv")
        self.assertEqual([l["id_prompt"] for l in linhas], ["PROMPT-001", "PROMPT-002"])


class TesteResumo(unittest.TestCase):
    def test_resumo_por_modelo(self):
        linhas = [
            {"modelo": "a", "status": "sucesso", "tempo_execucao_s": "2.0"},
            {"modelo": "a", "status": "sucesso", "tempo_execucao_s": "4.0"},
            {"modelo": "a", "status": "falha", "tempo_execucao_s": "0.1"},
            {"modelo": "b", "status": "falha", "tempo_execucao_s": "0.1"},
        ]
        resumo = {r["modelo"]: r for r in pipeline.resumir_por_modelo(linhas)}
        self.assertEqual((resumo["a"]["sucessos"], resumo["a"]["falhas"], resumo["a"]["tempo_medio_s"]), (2, 1, 3.0))
        self.assertEqual((resumo["b"]["sucessos"], resumo["b"]["tempo_medio_s"]), (0, ""))


if __name__ == "__main__":
    unittest.main()
