# Atividade — LLM-as-a-Judge

Implementação do pipeline de avaliação automática de respostas médicas usando um LLM como juiz.

## Dependências

```bash
pip install openai python-dotenv
```

## Configuração

Crie um arquivo `.env` na raiz do projeto seguindo o exemplo do `.env.example`

---

## Banco de dados

### Opção 1 — Com Docker Compose (recomendado)

Sobe o PostgreSQL e executa a carga automaticamente:

```bash
docker compose up -d
```

O serviço `loader` aguarda o banco ficar disponível, cria as tabelas via `schema.sql` e carrega `respostas_atividade_1.json`.

Para acompanhar os logs da carga:

```bash
docker compose logs -f loader
```

Para derrubar os containers mantendo os dados:

```bash
docker compose down
```

Para derrubar e apagar os dados:

```bash
docker compose down -v
```

---

### Opção 2 — Sem Docker (PostgreSQL local)

**Pré-requisitos:** PostgreSQL instalado e rodando localmente.

**1. Criar o banco e o usuário:**

```sql
CREATE USER llm_user WITH PASSWORD 'llm_pass';
CREATE DATABASE llm_judge OWNER llm_user;
```

**2. Criar as tabelas:**

```bash
psql -U llm_user -d llm_judge -f schema.sql
```

**3. Instalar dependência Python:**

```bash
pip install psycopg2-binary
```

**4. Executar a carga:**

```bash
python load_data.py
```

Por padrão o script conecta em `localhost:5432`. Para usar outro host ou porta, defina as variáveis de ambiente antes de executar:

```bash
DB_HOST=localhost DB_PORT=5432 DB_NAME=llm_judge DB_USER=llm_user DB_PASSWORD=llm_pass python load_data.py
```

---

## Restaurar a partir do dump

O arquivo `dump.sql` contém um snapshot completo do banco (schema + dados).

### Com Docker

**1. Suba apenas o container do banco:**

```bash
docker compose up -d db
```

**2. Restaure o dump:**

```bash
docker exec -i llm_judge_db psql -U llm_user -d llm_judge < dump.sql
```

### Sem Docker (PostgreSQL local)

**1. Crie o banco e o usuário** (se ainda não existirem):

```sql
CREATE USER llm_user WITH PASSWORD 'llm_pass';
CREATE DATABASE llm_judge OWNER llm_user;
```

**2. Restaure o dump:**

```bash
psql -U llm_user -d llm_judge -f dump.sql
```

> Usar o dump é a forma mais rápida de ter o banco pronto — não é necessário rodar `load_data.py` depois.
