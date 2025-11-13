# Pipeline de Machine Learning com PLN e Vector Database

## Descrição
Este projeto implementa um pipeline completo de machine learning para processamento de documentos, incluindo:
- Carregamento de múltiplos formatos (PDF, CSV, JSON)
- Processamento de Linguagem Natural (PLN)
- Chunking com overlap
- Embedding de textos
- Banco de dados vetorial
- Busca semântica

## Instalação

```bash
pip install sentence-transformers
pip install chromadb  # Opcional
pip install pdfplumber
pip install nltk
pip install numpy
```

## Uso Básico

```python
from pipeline_ml_completo import *

# 1. Carregar documento
texto = carregar_arquivo('seu_documento.pdf')

# 2. Processar com PLN
texto_processado = processar_texto_pln(texto)

# 3. Criar chunks
chunks = criar_chunks(texto_processado, tamanho_chunk=100, overlap=20)

# 4. Criar embeddings
embeddings = criar_embeddings(chunks, usar_modelo_real=True)

# 5. Criar banco vetorial
vector_db = VectorDB()
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadados = [{"index": i} for i in range(len(chunks))]
vector_db.adicionar(embeddings, chunks, metadados, ids)

# 6. Salvar banco
vector_db.salvar('meu_banco.pkl')

# 7. Realizar buscas
resultados = realizar_busca_vetorial(vector_db, "sua pergunta aqui", n_resultados=3)
```

## Estrutura do Pipeline

### 1. Carregamento de Arquivos
- **PDF**: Extração de texto com pdfplumber
- **CSV**: Leitura e concatenação de colunas
- **JSON**: Extração recursiva de strings

### 2. Processamento de Linguagem Natural (PLN)
- **Limpeza**: Remoção de caracteres especiais, normalização
- **Tokenização**: Divisão em palavras
- **Stopwords**: Remoção de palavras comuns (PT/EN)
- **Stemização**: Redução à raiz das palavras

### 3. Chunking
- Divisão em pedaços configuráveis
- Overlap para manter contexto
- Parâmetros ajustáveis (tamanho e overlap)

### 4. Embedding
- Suporte para sentence-transformers
- Modelo padrão: paraphrase-multilingual-MiniLM-L12-v2
- Fallback para embeddings simulados

### 5. Vector Database
- Implementação própria com numpy
- Busca por similaridade de cosseno
- Persistência em arquivo pickle
- Alternativa: ChromaDB (requer instalação)

### 6. Busca Vetorial
- Processamento automático de queries
- Ranking por similaridade
- Retorno de metadados e documentos

## Parâmetros Principais

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| tamanho_chunk | 100 | Palavras por chunk |
| overlap | 20 | Palavras de sobreposição |
| n_resultados | 3 | Número de resultados da busca |
| usar_modelo_real | True | Usar modelo real ou embeddings simulados |

## Arquivos Gerados

- `pipeline_ml_completo.py`: Código principal
- `vectordb.pkl`: Banco de dados vetorial salvo
- `resumo_pipeline.csv`: Resumo das etapas

## Observações

- O pipeline processa texto em **português** e **inglês**
- Stemização otimizada para português
- Stopwords configuráveis para ambos os idiomas
- Embeddings podem ser reais ou simulados (para testes)

## Exemplo Completo

Veja o arquivo `pipeline_ml_completo.py` para um exemplo completo de uso.

## Licença

MIT License
