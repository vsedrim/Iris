# Resultados exploratórios do PGC

**Status:** execução exploratória concluída  
**Data da execução:** 2026-08-09  
**Base:** `Data.zip`, leiaute `personBase_invert`, grupos de linha não verificados e deduplicação por
cópia exata

## Protocolo

A execução utilizou 123 imagens normalizadas únicas da íris: 66 controles reportados e 57 amostras
reportadas como diabetes. Foram usados cinco classificadores, três sementes e cinco *folds* por
semente. Os modelos foram ajustados somente com a condição original; as perturbações fotométricas
foram aplicadas apenas ao teste.

A configuração gerada registra hashes, versões dos pacotes, parâmetros e limitações da base. Nenhum
grupo sintético de linha cruzou os *folds*, mas os IDs reais das pessoas não estão disponíveis. Esse
controle não comprova isolamento por indivíduo. A execução também registra autorizações separadas
para identidade e rótulos clínicos não verificados.

## Baseline clássico

Os resultados abaixo preservam os controles clássicos pertinentes ao PGC na condição original.

| Modelo | Características | Acurácia | F1 | Sensibilidade | Especificidade |
|---|---|---:|---:|---:|---:|
| AdaBoost | Clássicas | 0,840 | 0,821 | 0,795 | 0,879 |
| *Random Forest* | Clássicas | 0,819 | 0,797 | 0,784 | 0,849 |

O melhor resultado desse baseline usou somente descritores clássicos. Ele constitui um controle para
os eixos de contribuição, não uma estimativa de desempenho clínico em DM2.

## Morfologia

Na execução exploratória do grupo `morphology`, a regressão logística obteve acurácia 0,789, F1
0,781, sensibilidade 0,813 e especificidade 0,769 na condição original.

Esse grupo inclui o contorno angular do colarete em $r(\theta)$, desvios RMS, padrão e máximo,
segundo harmônico, curvatura, contagem e profundidade de sulcos e proxies de criptas. O resultado não
valida essas medidas como descritores anatômicos: o contorno, os sulcos e as criptas ainda precisam
ser comparados a anotações humanas representativas.

## Vascularização

No recorte vascular do recesso, a regressão logística foi o melhor modelo na condição original, com
acurácia 0,7613 ± 0,0693. As condições de contraste reduziram a acurácia e evidenciaram instabilidade
fotométrica do proxy atual.

O protocolo, as demais métricas, os cinco cenários fotométricos e o plano de validação estão em
[RECESSO_VASCULAR.md](RECESSO_VASCULAR.md). O descritor é um proxy de *vesselness* em RGB, não uma
medida por AS-OCTA.

## Interpretação

Estes resultados substituem o uso direto da acurácia histórica próxima de 0,92 porque o arquivo
legado continha 73 linhas duplicadas entre 196 linhas e a avaliação antiga não demonstrava separação
por pessoa. A análise atual compara diabetes reportado e controle reportado; ela não confirma o
subtipo DM2. IDs reais, verificação clínica dos rótulos, metadados de aquisição e uma coorte
independente permanecem indisponíveis.

Os CSVs e demais resultados gerados são registros locais de execução e não são commitados no
repositório versionado.