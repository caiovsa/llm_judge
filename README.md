# Atividade — LLM-as-a-Judge

Implementação do pipeline de avaliação automática de respostas médicas usando um LLM como juiz.

## Vídeo de Apresentação

> Link: _a adicionar_

## Dependências

```bash
pip install openai psycopg2-binary python-dotenv pandas scipy streamlit plotly statsmodels
```

## Configuração

Crie um arquivo `.env` na raiz do projeto seguindo o exemplo do `.env.example`

---

## Pipeline completo com Docker Compose

O docker-compose executa todo o pipeline em sequência:

```
db → loader → judge → analysis → dashboard
```

| Serviço    | O que faz                                              |
|------------|--------------------------------------------------------|
| `db`       | PostgreSQL com schema criado via `schema.sql`          |
| `loader`   | Carrega `respostas_atividade_1.json` (K-QA)            |
| `judge`    | Avalia respostas com 3 LLMs juízes                     |
| `analysis` | Calcula médias e concordância entre juízes             |
| `dashboard`| Interface web em http://localhost:8501                 |

### Subir o pipeline completo

```bash
docker compose up
```

Para apagar os dados e recomeçar do zero:

```bash
docker compose down -v && docker compose up
```

Para subir apenas o dashboard (com banco já populado):

```bash
docker compose up dashboard
```

---

## Restaurar a partir do dump

O arquivo `dump.sql` contém um snapshot completo do banco (schema + todos os dados).  
Use esta opção para pular os loaders e ter o banco pronto imediatamente.

### Com Docker

**1. Suba o banco com o dump:**

```bash
docker compose up -d db
```

O PostgreSQL carrega `dump.sql` automaticamente na primeira inicialização (via `docker-entrypoint-initdb.d`).

> Se o volume `pgdata` já existir, o dump não é aplicado novamente. Para forçar:
> ```bash
> docker compose down -v && docker compose up -d db
> ```

**2. Suba o dashboard:**

```bash
docker compose up dashboard
```

Acesse **http://localhost:8501**.

---

### Sem Docker (PostgreSQL local)

**1. Crie o banco e o usuário:**

```sql
CREATE USER llm_user WITH PASSWORD 'llm_pass';
CREATE DATABASE llm_judge OWNER llm_user;
```

**2. Restaure o dump:**

```bash
psql -U llm_user -d llm_judge -f dump.sql
```

**3. Execute a análise:**

```bash
DB_HOST=localhost python src/analysis.py
```

**4. Suba o dashboard:**

```bash
DB_HOST=localhost streamlit run src/dashboard.py
```

---

## Criar o banco manualmente (sem dump)

### Com Docker

```bash
docker compose up db loader
```

### Sem Docker

**1. Crie o banco e o usuário:**

```sql
CREATE USER llm_user WITH PASSWORD 'llm_pass';
CREATE DATABASE llm_judge OWNER llm_user;
```

**2. Crie as tabelas:**

```bash
psql -U llm_user -d llm_judge -f schema.sql
```

**3. Carregue os dados:**

```bash
pip install psycopg2-binary
python load_open_questions.py
```
