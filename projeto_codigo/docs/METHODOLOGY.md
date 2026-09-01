# Metodologia e rastreabilidade

**Status:** protocolo de pesquisa implementado; validações clínica e morfológica pendentes  
**Escopo:** estudo exploratório de associações com DM2 em fotografias convencionais da íris

## Rastreabilidade

| Requisito do PGC | Implementação | Evidência produzida |
|---|---|---|
| Segmentação da íris e da pupila | `iris_dm2.preprocessing.segment_iris` | Matriz geométrica e relatório de exclusões |
| Normalização *rubber sheet* | `iris_dm2.preprocessing.rubber_sheet_normalize` | Matrizes RGB com 201 amostras radiais por 720 angulares |
| Baseline de intensidade, LBP e Haralick | Grupo `classic` | Colunas nomeadas em `features_<variant>.npz` |
| Viabilidade de vascularização | Grupo `vascular` | *Vesselness* de Frangi, máscara, esqueleto, ramificações e proxies vermelho-verde |
| Morfologia do colarete | Grupo `morphology` | $r(\theta)$, desvios RMS/padrão/máximo, segundo harmônico, curvatura e sulcos |
| Proxies de criptas | Grupo `morphology` | Densidade e raio de *blobs* na íris normalizada |
| Geometria da pupila e da íris | Matriz `geometry` | Diâmetros em pixels, razões, circularidade, desvios radiais e largura anular aparente |
| Medidas clínicas opcionais | `pupil_diameter_mm` e `iris_thickness_mm` | Valores externos e `ocular_measurement_source` preservados nos dados preparados |
| Validação por pessoa | `StratifiedGroupKFold` em `iris_dm2.evaluation` | IDs reais exigidos no manifesto e asserção de não sobreposição durante a execução |
| Modelos clássicos | `iris_dm2.evaluation.build_model` | Métricas por *fold* e predições de cinco classificadores |
| Acurácia, sensibilidade, especificidade, precisão e F1 | `classification_metrics` | Média e desvio padrão nos CSVs de resumo |
| Estabilidade entre sementes | Repetição dos *folds* agrupados | Semente em cada métrica, predição e atribuição |
| Estabilidade fotométrica | Cinco variantes determinísticas | Variante em cada métrica e predição |
| Contexto de execução | Hashes, versões, parâmetros e divisões exatas | `metadata.json` e configuração da execução |
| Qualidade da segmentação | Contenção, largura anular, limites da imagem e limiar de qualidade | Amostras rejeitadas em `exclusions.json` |
| Proveniência por amostra | Caminho, SHA-256, fonte do diagnóstico e hash do manifesto | `manifest.csv` e `metadata.json` preparados |

## Protocolo do conjunto de dados

A entrada preferencial é um manifesto de imagens brutas com `person_id`, rótulo clínico binário,
lateralidade, `diagnosis_source` e `diagnosis_verified`. A preparação preserva esses valores, o caminho
resolvido, o hash SHA-256 da imagem e o hash do manifesto. Os dois olhos de uma pessoa recebem o mesmo
grupo e não podem cruzar a fronteira entre treino e teste. Qualquer autorização provisória para a base
legada aparece no comando e na configuração da execução.

O manifesto também aceita `pupil_diameter_mm` e `iris_thickness_mm` como colunas opcionais. Esses
valores não são inferidos da fotografia. Quando ao menos uma das colunas está presente,
`ocular_measurement_source` é obrigatório, e todas as medidas devem ser positivas. Uma fonte adequada
para `iris_thickness_mm` é uma modalidade clínica externa, como AS-OCT ou UBM.

As fotografias RGB permitem obter `pupil_diameter_px`, `pupil_iris_diameter_ratio` e largura anular
aparente em pixels ou normalizada. Sem escala de aquisição, o diâmetro permanece uma medida relativa.
Mesmo com escala, a fotografia frontal não mede a espessura anatômica da íris.

O arquivo legado não contém identificadores nem fotografias brutas. O conversor atribui IDs sintéticos
estáveis por linha e remove duplicatas exatas de pixels. Isso controla apenas vazamento por cópia
exata. Duas linhas não idênticas ainda podem pertencer à mesma pessoa porque o mapeamento de origem
não está disponível. A preparação e a avaliação exigem autorização explícita, e os metadados gerados
marcam agrupamento e classe positiva como não verificados.

O IEEE retratou a publicação associada ao arquivo. O DOI original é
`10.1109/ICBME.2018.8703564`, e o aviso de retratação é
`10.1109/ICBME45317.2018.10207763`. O arquivo não constitui evidência científica e não deve ser
redistribuído até que proveniência, licença, consentimento e implicações da retratação sejam
esclarecidos.

## Protocolo de avaliação

Os modelos clássicos estimam filtragem por variância e escalonamento somente com o treino de cada
*fold*. Nenhuma transformação é ajustada sobre o conjunto completo. Os modelos usam hiperparâmetros
fixos e declarados. A MLP não cria uma divisão interna por amostra, de modo que os *folds* externos de
teste não participam da seleção do modelo.

Os metadados canônicos ficam dentro do NPZ e alteram seu SHA-256. O arquivo JSON lateral é comparado
ao conteúdo incorporado para impedir que uma alteração somente no arquivo lateral contorne os
bloqueios. Nos testes de estabilidade, os modelos treinam com características da condição original;
as variantes fotométricas nomeadas são usadas somente no *fold* de teste.

A segmentação é rejeitada quando a qualidade fica abaixo do limiar, o contorno da pupila deixa o
contorno da íris, a largura anular angular colapsa ou uma fronteira sai da imagem. Esses controles
verificam consistência geométrica, não acurácia clínica da segmentação.

## Protocolo morfológico

O grupo `morphology` calcula o gradiente radial na íris normalizada, localiza um contorno candidato do
colarete e o representa como $r(\theta)$. Após suavização angular periódica, a implementação calcula o
raio normalizado, os desvios RMS, padrão e máximo absoluto, o segundo harmônico e a curvatura RMS. Os
sulcos correspondem a mínimos locais do contorno com proeminência definida pelo próprio sinal; são
registrados sua contagem e seus valores médio e máximo de profundidade relativa.

A implementação também calcula a força da borda do colarete e usa detecção de *blobs* escuros para
produzir proxies de densidade e raio de criptas. Esses descritores ainda precisam ser comparados a
contornos, sulcos e criptas anotados por pessoas qualificadas. Até essa validação, devem ser tratados
como proxies de morfologia da imagem.

## Limites de interpretação

- O *vesselness* RGB é um proxy computacional. Ele não é AS-OCTA e não possui equivalência clínica
  demonstrada.
- As medidas de colarete, sulcos e criptas na faixa normalizada são proxies de imagem e exigem
  validação contra anotações humanas.
- O diâmetro pupilar é relativo, salvo quando a aquisição fornece escala e iluminação controladas.
- A largura anular em RGB é aparente e não representa espessura anatômica.
- `iris_thickness_mm` deve vir de medição clínica externa e identificada pela fonte.
- O arquivo legado não permite características geométricas porque a normalização descartou contornos
  e raios necessários.
- IDs sintéticos da base legada não estabelecem isolamento por pessoa.
- A classe positiva legada significa diabetes reportado; o subtipo DM2 não pode ser verificado.
- A remoção de duplicatas exatas reduz a base efetiva de 196 linhas para 123 imagens únicas.
- A validação em uma coorte independente permanece necessária antes de alegações de generalização.
- Comparações entre famílias de características e modelos são exploratórias e exigem controle de
  múltiplas comparações.
- As saídas representam métricas de pesquisa em nível de coorte, não diagnósticos individuais.

## Requisitos antes do relato científico

Antes de relatar conclusões científicas, é necessário obter e documentar:

1. proveniência e licença do arquivo de imagens;
2. confirmação de que a classe positiva representa especificamente DM2;
3. identificadores reais de participantes ou agrupamento fornecido pelo custodiante;
4. metadados de aquisição, inclusive iluminação, câmera e escala quando aplicável;
5. fonte clínica de qualquer diâmetro em milímetros ou espessura anatômica;
6. anotações humanas para validar colarete, sulcos, criptas e proxies vasculares;
7. coorte externa independente ou declaração explícita de sua ausência;
8. plano estatístico para intervalos de confiança e correção de múltiplas comparações.