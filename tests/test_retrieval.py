from app import assinatura_base, catalogo, selecionar, ler, recuperar, grupos, resposta_valida
import re


def carregar(question):
    cat = catalogo(assinatura_base())
    docs = []
    for item in selecionar(question, cat):
        loaded = ler(item["arquivo"], item["tamanho"], item["mtime"])
        if loaded:
            docs.append(loaded)
    return recuperar(question, docs)


def test_it11_width_question_retrieves_context():
    question = "Qual a largura mínima das saídas de emergência conforme a IT nº 11/2025?"
    passages = carregar(question)
    assert passages, "Nenhuma página relevante foi recuperada."
    text = "\n".join(p["texto"].lower() for p in passages)
    assert "largura" in text
    assert any("it nº 11-25" in p["arquivo"].lower() for p in passages)


def test_f11_is_hard_filter_for_retrieval():
    question = "Para uma edificação com população de 100 pessoas, quantas unidades de passagem são necessárias e qual deve ser a largura da saída de emergência? grupo F-11"
    assert grupos(question) == {"F-11"}
    passages = carregar(question)
    assert passages, "Nenhuma página foi recuperada para F-11."
    text = "\n".join(p["texto"].upper() for p in passages)
    assert "F-11" in text, "A página recuperada não contém F-11."
    for p in passages:
        codes = set(re.findall(r"\b[A-Z]{1,3}-\d{1,3}\b", p["texto"].upper()))
        if codes:
            assert "F-11" in codes


def test_wrong_occupancy_answer_is_rejected():
    question = "Grupo F-11: para 100 pessoas, quantas unidades de passagem são necessárias?"
    assert resposta_valida("for E-5, E-6? Wait, let's look at the text", question) is None
    assert resposta_valida("Para o grupo E-5, são necessárias 2 unidades.", question) is None


def test_clean_answer_is_accepted():
    question = "Grupo F-11: qual a largura mínima?"
    assert resposta_valida("A largura mínima é 1,20 m.", question) == "A largura mínima é 1,20 m."


def test_decreto_reference_is_selected():
    question = "Qual o número do decreto de segurança contra incêndio?"
    cat = catalogo(assinatura_base())
    docs = selecionar(question, cat)
    assert docs, "O Decreto 69.118/2024 não foi localizado na base."
    assert "decreto" in docs[0]["norm"]
