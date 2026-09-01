# Relatório de recesso: vascularização

**Status:** execução exploratória concluída; validação vascular e clínica pendente  
**Base:** 123 imagens únicas da base legada, com IDs e rótulos clínicos não verificados

## Objetivo

Avaliar se descritores computacionais de estruturas lineares em fotografias RGB da íris apresentam
sinal útil para classificação no arquivo legado e determinar quais validações ainda são necessárias
antes de qualquer interpretação vascular ou clínica.

O descritor avaliado é um proxy RGB de *vesselness*. Ele não mede vasos por AS-OCTA e não possui
equivalência clínica demonstrada com densidade ou fluxo vascular da íris.

## Status e protocolo

A execução utilizou 123 imagens únicas após a remoção de duplicatas exatas da base legada. Os grupos
de linha e os rótulos fornecidos pela base não permitem verificar a identidade real das pessoas nem
confirmar que a classe positiva representa especificamente DM2.

O protocolo usou:

- o grupo de características `vascular`, extraído das imagens RGB normalizadas;
- descritores de *vesselness* de Frangi, máscara e esqueleto do proxy, ramificações e contraste entre
  os canais vermelho e verde;
- regressão logística, SVM, *Random Forest*, MLP e AdaBoost;
- três sementes e cinco *folds* por semente;
- treino com as características da condição original e avaliação separada nas condições original,
  `brightness_low`, `brightness_high`, `contrast_low` e `contrast_high`.

Os limiares atuais do proxy não foram validados contra anotações humanas de vasos. A execução mede o
comportamento do descritor implementado, não a qualidade de uma segmentação vascular.

## Resultados

A regressão logística foi o melhor modelo na condição original. A tabela apresenta média e desvio
padrão entre as 15 avaliações de cada condição.

| Condição | Acurácia | F1 | Sensibilidade | Especificidade |
|---|---:|---:|---:|---:|
| Original | 0,7613 ± 0,0693 | 0,7453 ± 0,0800 | 0,7641 ± 0,1089 | 0,7582 ± 0,0740 |
| `brightness_low` | 0,7371 ± 0,0596 | 0,7608 ± 0,0505 | 0,9005 ± 0,0740 | 0,5960 ± 0,1098 |
| `brightness_high` | 0,7477 ± 0,0686 | 0,6656 ± 0,1060 | 0,5545 ± 0,1172 | 0,9143 ± 0,0699 |
| `contrast_low` | 0,6013 ± 0,0701 | 0,6984 ± 0,0382 | 0,9889 ± 0,0293 | 0,2689 ± 0,1358 |
| `contrast_high` | 0,6281 ± 0,0697 | 0,3597 ± 0,1932 | 0,2475 ± 0,1569 | 0,9542 ± 0,0565 |

## Interpretação limitada

O resultado original mostra que o proxy RGB produziu separação estatística exploratória sob os
rótulos legados. Ele não demonstra associação com DM2, generalização por indivíduo nem medição da
vasculatura da íris. Essas conclusões permanecem bloqueadas pela falta de IDs reais, pela ausência de
verificação dos rótulos clínicos e pela inexistência de *ground truth* vascular.

A queda de acurácia nas duas condições de contraste indica sensibilidade fotométrica do fluxo atual.
Ela não identifica, por si só, se o modelo utiliza vasos, textura, pigmentação ou artefatos de
aquisição. As condições de brilho produziram acurácias mais próximas da condição original, mas isso
também não valida o descritor como medida vascular.

## Plano executável do recesso

1. **Auditar visualmente o proxy atual.** Selecionar amostras representativas, sobrepor resposta de
   *vesselness*, máscara e esqueleto à imagem e registrar falsos positivos em pigmentação, reflexos,
   pálpebras e cílios.
2. **Construir o *ground truth*.** Definir um protocolo de anotação de vasos visíveis em RGB, anotar
   um conjunto reservado e registrar concordâncias e divergências entre anotadores. Manter imagens
   sem vasos discerníveis como casos válidos, sem fabricar traçados.
3. **Comparar detectores.** Implementar e comparar Frangi isolado, Frangi após CLAHE, transformação
   *black-hat*, filtros de Gabor e detectores de linhas. Executar todos sobre as mesmas imagens e com
   as mesmas divisões.
4. **Controlar oclusões e reflexos.** Criar máscaras explícitas para reflexos especulares, pálpebras e
   cílios. Medir a diferença entre resultados com e sem cada máscara.
5. **Impedir vazamento de calibração.** Ajustar limiares, escalas e demais parâmetros usando somente
   os dados de treino de cada *fold*. Congelar esses valores antes de processar o respectivo teste.
6. **Medir estabilidade fotométrica.** Repetir a avaliação com as cinco condições já definidas e
   incluir uma análise por amostra da estabilidade da máscara, do esqueleto e da predição.
7. **Executar ablações.** Avaliar separadamente *vesselness*, densidade de máscara, densidade de
   esqueleto, ramificações e contraste vermelho-verde. Comparar cada bloco com o conjunto combinado.
8. **Adicionar controles.** Criar um teste sintético com linhas de espessura, contraste, curvatura e
   ruído conhecidos. Avaliar também o conjunto anotado, mantido fora da calibração final.
9. **Repetir após a auditoria clínica.** Reexecutar a classificação somente quando os rótulos e os
   IDs reais estiverem disponíveis, preservando todas as imagens da mesma pessoa no mesmo *fold*.

## Critérios de saída

O eixo vascular estará pronto para relato científico somente quando:

- a proveniência, a identidade por pessoa e os rótulos clínicos estiverem documentados;
- o detector e seus limiares forem definidos sem acesso aos dados de teste;
- o teste sintético demonstrar o comportamento esperado para estruturas lineares conhecidas;
- o conjunto anotado permitir medir acertos e erros do proxy vascular;
- as máscaras de oclusão e reflexo forem auditadas;
- a ablação identificar quais componentes sustentam o resultado;
- a estabilidade fotométrica for relatada sob um critério definido antes da avaliação final.

Se esses critérios não forem atendidos, o resultado deve permanecer descrito como classificação por
um proxy RGB de estruturas lineares, sem interpretação vascular ou clínica.

## Origem local da execução

Os valores acima foram obtidos dos CSVs locais em
`PGC/projeto_codigo/results/recesso-vascular/evaluation/`, em especial `summary.csv`,
`metrics_by_fold.csv`, `predictions.csv` e `fold_assignments.csv`. Esses resultados gerados servem como
registro da execução local e não são commitados no repositório versionado.