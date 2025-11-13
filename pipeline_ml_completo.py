"""
PIPELINE DE MACHINE LEARNING COM PLN E VECTORDB
================================================

Este notebook implementa um pipeline completo de machine learning incluindo:
1. Carregamento de arquivos (PDF, CSV, JSON)
2. Processamento de Linguagem Natural (PLN)
3. Chunking com overlap
4. Embedding de textos
5. Armazenamento em banco de dados vetorial
6. Busca vetorial (Retrieval)

Autor: Pipeline ML
Data: 2024
"""

# ============================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS (executar apenas uma vez)
# ============================================================================
"""
!pip install sentence-transformers
!pip install chromadb
!pip install pdfplumber
!pip install nltk
!pip install numpy
"""

# ============================================================================
# IMPORTS
# ============================================================================
import json
import csv
import re
import numpy as np
import pickle
import os

# ============================================================================
# 1. CARREGAMENTO DE ARQUIVOS
# ============================================================================
print("="*80)
print("ETAPA 1: CARREGAMENTO DE ARQUIVOS")
print("="*80)

def carregar_pdf(caminho_arquivo):
    """Carrega e extrai texto de um arquivo PDF"""
    try:
        import pdfplumber
        texto_completo = []
        with pdfplumber.open(caminho_arquivo) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo.append(texto)
        return "\n".join(texto_completo)
    except ImportError:
        print("⚠ pdfplumber não disponível. Instale com: pip install pdfplumber")
        return ""

def carregar_csv(caminho_arquivo):
    """Carrega e extrai texto de um arquivo CSV"""
    textos = []
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for linha in reader:
            texto = " ".join([str(v) for v in linha.values()])
            textos.append(texto)
    return "\n".join(textos)

def carregar_json(caminho_arquivo):
    """Carrega e extrai texto de um arquivo JSON"""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    textos = []

    def extrair_texto_recursivo(obj):
        if isinstance(obj, dict):
            for valor in obj.values():
                extrair_texto_recursivo(valor)
        elif isinstance(obj, list):
            for item in obj:
                extrair_texto_recursivo(item)
        elif isinstance(obj, str):
            textos.append(obj)

    extrair_texto_recursivo(dados)
    return "\n".join(textos)

def carregar_arquivo(caminho_arquivo):
    """Função universal para carregar arquivos PDF, CSV ou JSON"""
    extensao = caminho_arquivo.split('.')[-1].lower()

    if extensao == 'pdf':
        return carregar_pdf(caminho_arquivo)
    elif extensao == 'csv':
        return carregar_csv(caminho_arquivo)
    elif extensao == 'json':
        return carregar_json(caminho_arquivo)
    else:
        raise ValueError(f"Formato não suportado: {extensao}")

# ============================================================================
# 2. PROCESSAMENTO DE LINGUAGEM NATURAL (PLN)
# ============================================================================
print("\n" + "="*80)
print("ETAPA 2: PROCESSAMENTO DE LINGUAGEM NATURAL")
print("="*80)

# Stopwords em português e inglês
STOPWORDS_PT = {'a', 'o', 'e', 'é', 'de', 'da', 'do', 'em', 'um', 'uma', 'os', 'as', 
                'dos', 'das', 'para', 'com', 'por', 'no', 'na', 'nos', 'nas', 'ao', 
                'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas', 'que', 'se', 
                'não', 'mais', 'como', 'mas', 'foi', 'são', 'ser', 'tem', 'ter'}

STOPWORDS_EN = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had'}

STOPWORDS_ALL = STOPWORDS_PT.union(STOPWORDS_EN)

def limpar_texto(texto):
    """Realiza limpeza básica do texto"""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéêíóôõúçñ\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def tokenizar_simples(texto):
    """Tokeniza o texto (divisão por espaços)"""
    return texto.split()

def remover_stopwords(palavras, stopwords=STOPWORDS_ALL):
    """Remove stopwords em português e inglês"""
    return [palavra for palavra in palavras 
            if palavra not in stopwords and len(palavra) > 2]

def stemizar_simples(palavra):
    """Stemização simplificada em português"""
    sufixos = ['mente', 'ação', 'ador', 'ante', 'ância', 'ência', 'izar',
               'oso', 'osa', 'ivo', 'iva', 'vel', 'eis', 'al', 'ar', 'er', 
               'ir', 'ado', 'ada', 'ido', 'ida', 'ando', 'endo', 'indo']

    for sufixo in sufixos:
        if palavra.endswith(sufixo) and len(palavra) > len(sufixo) + 3:
            return palavra[:-len(sufixo)]
    return palavra

def stemizar_texto(palavras):
    """Aplica stemização nas palavras"""
    return [stemizar_simples(palavra) for palavra in palavras]

def processar_texto_pln(texto):
    """Pipeline completo de PLN"""
    texto_limpo = limpar_texto(texto)
    palavras = tokenizar_simples(texto_limpo)
    palavras_sem_stopwords = remover_stopwords(palavras)
    palavras_stem = stemizar_texto(palavras_sem_stopwords)
    return ' '.join(palavras_stem)

# ============================================================================
# 3. CHUNKING COM OVERLAP
# ============================================================================
print("\n" + "="*80)
print("ETAPA 3: CHUNKING")
print("="*80)

def criar_chunks(texto, tamanho_chunk=150, overlap=30):
    """
    Divide o texto em chunks com overlap

    Args:
        texto: Texto a ser dividido
        tamanho_chunk: Tamanho de cada chunk em palavras
        overlap: Número de palavras que se sobrepõem entre chunks

    Returns:
        Lista de chunks
    """
    palavras = texto.split()
    chunks = []

    inicio = 0
    while inicio < len(palavras):
        fim = min(inicio + tamanho_chunk, len(palavras))
        chunk = ' '.join(palavras[inicio:fim])
        chunks.append(chunk)

        if fim >= len(palavras):
            break

        inicio = fim - overlap

    return chunks

# ============================================================================
# 4. EMBEDDING
# ============================================================================
print("\n" + "="*80)
print("ETAPA 4: EMBEDDING")
print("="*80)

def criar_embeddings(chunks, usar_modelo_real=True):
    """Cria embeddings para os chunks"""
    if usar_modelo_real:
        try:
            from sentence_transformers import SentenceTransformer

            print("Carregando modelo de embedding...")
            modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ Modelo carregado")

            embeddings = modelo.encode(chunks, show_progress_bar=True)
            return embeddings

        except ImportError:
            print("⚠ sentence-transformers não disponível")
            print("Usando embeddings simulados...")
            usar_modelo_real = False

    if not usar_modelo_real:
        # Embeddings simulados para demonstração
        dimensao = 384
        embeddings = np.random.randn(len(chunks), dimensao).astype('float32')
        return embeddings

# ============================================================================
# 5. VECTOR DATABASE
# ============================================================================
print("\n" + "="*80)
print("ETAPA 5: BANCO DE DADOS VETORIAL")
print("="*80)

class VectorDB:
    """Banco de dados vetorial simples"""

    def __init__(self):
        self.embeddings = []
        self.documentos = []
        self.metadados = []
        self.ids = []

    def adicionar(self, embeddings, documentos, metadados, ids):
        """Adiciona documentos ao banco"""
        self.embeddings = np.array(embeddings)
        self.documentos = documentos
        self.metadados = metadados
        self.ids = ids

    def buscar(self, query_embedding, n_resultados=3):
        """Busca documentos similares (similaridade de cosseno)"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        similaridades = np.dot(docs_norm, query_norm)
        indices_top = np.argsort(similaridades)[::-1][:n_resultados]

        resultados = []
        for idx in indices_top:
            resultados.append({
                'id': self.ids[idx],
                'documento': self.documentos[idx],
                'metadados': self.metadados[idx],
                'similaridade': float(similaridades[idx])
            })

        return resultados

    def salvar(self, caminho='vectordb.pkl'):
        """Salva o banco em arquivo"""
        dados = {
            'embeddings': self.embeddings,
            'documentos': self.documentos,
            'metadados': self.metadados,
            'ids': self.ids
        }
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        print(f"✓ Banco salvo em: {caminho}")

    def carregar(self, caminho='vectordb.pkl'):
        """Carrega o banco de arquivo"""
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
        self.embeddings = dados['embeddings']
        self.documentos = dados['documentos']
        self.metadados = dados['metadados']
        self.ids = dados['ids']
        print(f"✓ Banco carregado de: {caminho}")

    def contar(self):
        """Retorna número de documentos"""
        return len(self.documentos)

# ============================================================================
# 6. BUSCA VETORIAL (QUERY)
# ============================================================================
print("\n" + "="*80)
print("ETAPA 6: BUSCA VETORIAL")
print("="*80)

def realizar_busca_vetorial(vector_db, query, processar_query=True, n_resultados=3):
    """
    Realiza busca vetorial

    Args:
        vector_db: Instância do banco de dados vetorial
        query: Texto da query
        processar_query: Se True, aplica PLN na query
        n_resultados: Número de resultados a retornar
    """
    print(f"\nQUERY: {query}")
    print("-" * 70)

    # Processar query (mesmo pipeline do texto)
    if processar_query:
        query_processada = processar_texto_pln(query)
        print(f"Query processada: {query_processada}")
    else:
        query_processada = query

    # Criar embedding da query
    # Nota: Em produção, usar o mesmo modelo dos documentos
    query_embedding = criar_embeddings([query_processada], usar_modelo_real=False)[0]

    # Buscar documentos similares
    resultados = vector_db.buscar(query_embedding, n_resultados=n_resultados)

    print(f"\nResultados: {len(resultados)}")
    for i, resultado in enumerate(resultados, 1):
        print(f"\n[{i}] ID: {resultado['id']}")
        print(f"    Similaridade: {resultado['similaridade']:.4f}")
        print(f"    Documento: {resultado['documento'][:150]}...")

    return resultados

# ============================================================================
# EXEMPLO DE USO COMPLETO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXEMPLO DE USO DO PIPELINE")
    print("="*80)

    # 1. Criar arquivos de teste (você pode pular isso se já tiver arquivos)
    # [Código para criar arquivos de teste aqui]

    # 2. Carregar arquivos
    # texto = carregar_arquivo('seu_arquivo.pdf')
    texto = "Seu texto de exemplo aqui..."

    # 3. Processar com PLN
    texto_processado = processar_texto_pln(texto)
    print(f"\n✓ Texto processado: {len(texto_processado)} caracteres")

    # 4. Criar chunks
    chunks = criar_chunks(texto_processado, tamanho_chunk=100, overlap=20)
    print(f"✓ Criados {len(chunks)} chunks")

    # 5. Criar embeddings
    embeddings = criar_embeddings(chunks, usar_modelo_real=False)
    print(f"✓ Embeddings criados: {embeddings.shape}")

    # 6. Criar e popular banco vetorial
    vector_db = VectorDB()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadados = [{"index": i} for i in range(len(chunks))]
    vector_db.adicionar(embeddings, chunks, metadados, ids)
    print(f"✓ Banco vetorial criado com {vector_db.contar()} documentos")

    # 7. Salvar banco
    vector_db.salvar('meu_vectordb.pkl')

    # 8. Realizar buscas
    queries = [
        "machine learning",
        "processamento de linguagem natural",
        "inteligência artificial"
    ]

    for query in queries:
        realizar_busca_vetorial(vector_db, query, n_resultados=2)

    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)
