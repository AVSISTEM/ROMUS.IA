from app import assinatura_base, catalogo, selecionar, ler, recuperar


def test_it11_width_question_retrieves_context():
    question = "Qual a largura mínima das saídas de emergência conforme a IT nº 11/2025?"
    cat = catalogo(assinatura_base())
    docs = []
    for item in selecionar(question, cat):
        loaded = ler(item["arquivo"], item["tamanho"], item["mtime"])
        if loaded:
            docs.append(loaded)

    assert docs, "A IT 11/2025 não foi localizada na base."
    passages = recuperar(question, docs)
    assert passages, "Nenhuma página relevante foi recuperada."

    text = "\n".join(p["texto"].lower() for p in passages)
    assert "largura" in text, "O contexto recuperado não contém a palavra-chave esperada."
    assert any("it nº 11-25" in p["arquivo"].lower() for p in passages)


def test_decreto_reference_is_selected():
    question = "Qual o número do decreto de segurança contra incêndio?"
    cat = catalogo(assinatura_base())
    docs = selecionar(question, cat)
    assert docs, "O Decreto 69.118/2024 não foi localizado na base."
    assert "decreto" in docs[0]["norm"]
