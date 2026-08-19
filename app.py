# =========================================================
# 2. CONFIGURAÇÕES GERAIS E MODELOS DE CONTINGÊNCIA
# =========================================================
BASE_CONHECIMENTO_DIR = "base_conhecimento"
ARQUIVOS_SUPORTADOS = (".txt", ".pdf")

MODELO_PRINCIPAL = "gemini-2.5-flash"
MODELO_FALLBACK = "gemini-2.0-flash"

TAMANHO_CHUNK = 1800
SOBREPOSICAO_CHUNK = 250
TOP_CHUNKS = 12

# =========================================================
# 6. GERAÇÃO DE RESPOSTA COM RETRY E FALLBACK AUTOMÁTICO
# =========================================================
def gerar_resposta(pergunta: str, modo_estrito: bool = True):
    try:
        cliente = criar_cliente()
    except Exception as e:
        return {"ok": False, "texto": "", "tempo": 0, "trechos": [], "erro": str(e)}

    trechos = buscar_trechos_na_base(pergunta, TOP_CHUNKS)
    contexto = montar_contexto(trechos)
    prompt_usuario = f"PERGUNTA DO USUÁRIO:\n{pergunta}\n\nBASE LOCAL LOCALIZADA:\n{contexto}"

    inicio = time.time()
    modelos_para_tentar = [MODELO_PRINCIPAL, MODELO_FALLBACK]
    ultimo_erro = ""

    for modelo in modelos_para_tentar:
        for tentativa in range(3):  # Até 3 tentativas por modelo
            try:
                resposta = cliente.models.generate_content(
                    model=modelo,
                    contents=prompt_usuario,
                    config=types.GenerateContentConfig(
                        system_instruction=PROMPT_SISTEMA,
                        temperature=0.0
                    )
                )
                tempo = round(time.time() - inicio, 2)
                texto = resposta.text.strip() if hasattr(resposta, "text") and resposta.text else ""

                if modo_estrito and not trechos:
                    texto = "Não localizei base suficiente para responder com segurança."

                return {
                    "ok": True,
                    "texto": texto if texto else "Não houve resposta textual do modelo.",
                    "tempo": tempo,
                    "trechos": trechos,
                    "modelo_usado": modelo,
                    "erro": ""
                }
            except Exception as e:
                ultimo_erro = str(e)
                # Se for erro 503 (indisponibilidade temporária), aguarda antes de tentar novamente
                if "503" in ultimo_erro or "UNAVAILABLE" in ultimo_erro:
                    time.sleep(1.5 * (tentativa + 1))
                else:
                    break  # Para outros tipos de erro, passa para o próximo modelo

    return {
        "ok": False,
        "texto": "",
        "tempo": round(time.time() - inicio, 2),
        "trechos": trechos,
        "erro": f"Servidores indisponíveis após várias tentativas. Detalhe: {ultimo_erro}"
    }
