# Iris-DM2

Projeto de Graduação em Computação da UFABC sobre a avaliação de biomarcadores oculares associados
ao diabetes mellitus tipo 2 (DM2) em fotografias convencionais da íris.

O PGC investiga três eixos de contribuição:

1. classificação baseada em correlatos de vascularização;
2. classificação baseada no diâmetro da pupila e na espessura da íris;
3. caracterização morfológica da coroa pupilar e iridiana por desvios em relação à circunferência
  ajustada, sulcos, colaretes e proxies de criptas.

O baseline clássico permanece como controle experimental. Arquiteturas profundas, aprendizado por
transferência e descritores cromáticos pertencem à IC2, mantida separadamente no repositório
[`vsedrim/Iris-IC2`](https://github.com/vsedrim/Iris-IC2).

A proposta não valida a iridologia e não deve ser usada para diagnóstico individual ou decisões
clínicas.

## Conteúdo

- [`projeto_pgc.pdf`](projeto_pgc.pdf): versão compilada do projeto.
- [`projeto_pgc.tex`](projeto_pgc.tex): fonte LaTeX do relatório.
- [`imagens/`](imagens/): figuras e respectivas atribuições.
- [`projeto_codigo/`](projeto_codigo/): implementação do protocolo computacional.
- [`projeto_codigo/docs/METHODOLOGY.md`](projeto_codigo/docs/METHODOLOGY.md): rastreabilidade,
  metodologia e limites científicos.
- [`projeto_codigo/docs/BASELINE_RESULTS.md`](projeto_codigo/docs/BASELINE_RESULTS.md): resultados
  exploratórios da linha de base clássica.
- [`projeto_codigo/docs/RECESSO_VASCULAR.md`](projeto_codigo/docs/RECESSO_VASCULAR.md): execução de
  recesso com o proxy RGB de vascularização e plano de validação.

## Execução

O código requer Python 3.12. Em PowerShell:

```powershell
Set-Location .\projeto_codigo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Os comandos de preparação, extração de características e avaliação estão documentados em
[`projeto_codigo/README.md`](projeto_codigo/README.md).

## Dados e limitações

Fotografias RGB permitem estimar o diâmetro pupilar em pixels, a razão entre os diâmetros da pupila e
da íris e a largura anular aparente. Elas não medem a espessura anatômica da íris. O campo opcional
`iris_thickness_mm` deve receber uma medição clínica externa, por exemplo, de AS-OCT ou UBM, e requer
o registro de `ocular_measurement_source`. O código atual aceita e preserva essas colunas opcionais no
manifesto.

Os dados brutos, resultados de execução e ambientes locais não são versionados. A base legada possui
limitações de proveniência, licenciamento, identificação dos participantes e confirmação dos rótulos
clínicos. Ela não deve ser redistribuída nem usada como evidência científica sem que essas questões
sejam resolvidas com o custodiante dos dados.

Os resultados atuais são exploratórios. Uma conclusão científica exige identificadores reais por
pessoa, confirmação clínica de DM2, metadados de aquisição e validação em uma coorte independente.
