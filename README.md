# Iris-DM2

Projeto de Graduação em Computação da UFABC sobre a avaliação de biomarcadores oculares associados
ao diabetes mellitus tipo 2 (DM2) em fotografias convencionais da íris.

O trabalho investiga vascularização, diâmetro pupilar, morfologia e descritores cromáticos por meio
de visão computacional e aprendizado de máquina. A proposta não valida a iridologia e não deve ser
usada para diagnóstico individual ou decisões clínicas.

## Conteúdo

- [`projeto_pgc.pdf`](projeto_pgc.pdf): versão compilada do projeto.
- [`projeto_pgc.tex`](projeto_pgc.tex): fonte LaTeX do relatório.
- [`imagens/`](imagens/): figuras e respectivas atribuições.
- [`projeto_codigo/`](projeto_codigo/): implementação do protocolo computacional.
- [`projeto_codigo/docs/METHODOLOGY.md`](projeto_codigo/docs/METHODOLOGY.md): rastreabilidade,
  metodologia e limites científicos.
- [`projeto_codigo/docs/BASELINE_RESULTS.md`](projeto_codigo/docs/BASELINE_RESULTS.md): resultados
  exploratórios da linha de base clássica.

## Execução

O código requer Python 3.12. Em PowerShell:

```powershell
Set-Location .\projeto_codigo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,deep]"
.\.venv\Scripts\python.exe -m pytest -q
```

Os comandos de preparação, extração de características e avaliação estão documentados em
[`projeto_codigo/README.md`](projeto_codigo/README.md).

## Dados e limitações

Os dados brutos, resultados de execução e ambientes locais não são versionados. A base legada possui
limitações de proveniência, licenciamento, identificação dos participantes e confirmação dos rótulos
clínicos. Ela não deve ser redistribuída nem usada como evidência científica sem que essas questões
sejam resolvidas com o custodiante dos dados.

Os resultados atuais são exploratórios. Uma conclusão científica exige identificadores reais por
pessoa, confirmação clínica de DM2, metadados de aquisição e validação em uma coorte independente.
