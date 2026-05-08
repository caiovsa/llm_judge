CREATE EXTENSION IF NOT EXISTS "pgcrypto";


CREATE TABLE modelos (
    id_modelo    SERIAL PRIMARY KEY,
    nome_modelo  TEXT   NOT NULL,
    versao       TEXT   NOT NULL,
    tipo         TEXT   NOT NULL CHECK (tipo IN ('candidato', 'juiz')),

    UNIQUE (nome_modelo, versao)
);


CREATE TABLE datasets (
    id_dataset   SERIAL PRIMARY KEY,
    nome_dataset TEXT   NOT NULL UNIQUE,
    descricao    TEXT
);


CREATE TABLE perguntas (
    id_pergunta   SERIAL PRIMARY KEY,
    id_dataset    INT    NOT NULL REFERENCES datasets(id_dataset) ON DELETE CASCADE,
    enunciado     TEXT   NOT NULL,
    resposta_ouro TEXT   NOT NULL
);


CREATE TABLE prompts (
    id_prompt  SERIAL PRIMARY KEY,
    tipo       TEXT   NOT NULL CHECK (tipo IN ('candidato', 'juiz')),
    texto      TEXT   NOT NULL
);


CREATE TABLE respostas_atividade_1 (
    id_resposta    SERIAL PRIMARY KEY,
    id_pergunta    INT    NOT NULL REFERENCES perguntas(id_pergunta) ON DELETE CASCADE,
    id_modelo      INT    NOT NULL REFERENCES modelos(id_modelo),
    id_prompt      INT    REFERENCES prompts(id_prompt),
    texto_resposta TEXT   NOT NULL
);


CREATE TABLE avaliacoes_juiz (
    id_avaliacao   SERIAL   PRIMARY KEY,
    id_resposta    INT      NOT NULL REFERENCES respostas_atividade_1(id_resposta) ON DELETE CASCADE,
    id_modelo_juiz INT      NOT NULL REFERENCES modelos(id_modelo),
    id_prompt      INT      NOT NULL REFERENCES prompts(id_prompt),
    nota           SMALLINT NOT NULL CHECK (nota BETWEEN 1 AND 5),
    justificativa  TEXT     NOT NULL
);