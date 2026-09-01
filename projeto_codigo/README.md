# Código de pesquisa do PGC Iris-DM2

Este diretório contém o protocolo computacional descrito em `../projeto_pgc.tex`. O PGC avalia
associações exploratórias entre imagens da íris e rótulos reportados de diabetes mellitus tipo 2
(DM2). O código não implementa a iridologia e não se destina a diagnóstico individual ou decisão
clínica.

O escopo documentado do PGC inclui:

- conversão restrita da base legada em Python 2 para NPZ, sem permitir *pickle* nos experimentos;
- segmentação da pupila e da íris em imagens brutas, seguida de normalização *rubber sheet*;
- descritores clássicos de intensidade, LBP e Haralick como baseline de controle;
- proxies RGB de vascularização;
- geometria da pupila e da íris quando os contornos de segmentação estão disponíveis;
- medidas clínicas opcionais de diâmetro pupilar e espessura anatômica da íris;
- morfologia do colarete em $r(\theta)$, desvios do contorno, harmônico, curvatura, sulcos e proxies
  de criptas;
- validação cruzada estratificada e agrupada, com divisões compartilhadas entre modelos;
- regressão logística, SVM, *Random Forest*, MLP e AdaBoost como classificadores clássicos;
- predições por *fold*, cinco métricas, médias, desvios padrão, sementes e testes de estabilidade
  fotométrica.

Consulte [docs/METHODOLOGY.md](docs/METHODOLOGY.md) para a rastreabilidade e os limites científicos,
[docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) para a linha de base clássica e
[docs/RECESSO_VASCULAR.md](docs/RECESSO_VASCULAR.md) para a execução do recesso sobre o eixo vascular.

## Integridade da base legada

`Data.zip` contém 108 linhas de controle e 88 linhas reportadas como diabetes em
`personBase_invert`. A preparação identificou 73 duplicatas exatas e reteve 123 imagens únicas: 66
controles reportados e 57 amostras reportadas como diabetes. As duplicatas são removidas de forma
determinística e registradas em `metadata.json`. Resultados do código legado precisam ser recalculados
antes de qualquer comparação.

O arquivo contém somente faixas de íris já normalizadas. Ele não contém fotografias brutas, IDs reais
de participantes, lateralidade, contornos da pupila e da íris ou metadados de aquisição. Portanto,
não permite reconstruir diâmetro pupilar, largura anular nem morfologia das bordas. IDs sintéticos por
linha não comprovam identidade, e a classe positiva não pode ser confirmada como DM2.

## Instalação

O projeto requer Python 3.12. A partir deste diretório, no PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Fluxo da base legada

Prepare a base NPZ deduplicada:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 prepare `
  --archive .\Data.zip `
  --layout personBase_invert `
  --output .\data\legacy `
  --acknowledge-unverified-legacy-metadata
```

Extraia os grupos do PGC e as variantes fotométricas:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 extract `
  --dataset .\data\legacy\dataset.npz `
  --output .\data\legacy\features `
  --feature-groups classic vascular morphology `
  --variants original brightness_low brightness_high contrast_low contrast_high
```

Execute a avaliação clássica:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 evaluate `
  --dataset .\data\legacy\dataset.npz `
  --features .\data\legacy\features `
  --output .\results\classical `
  --models logistic_regression svm random_forest mlp adaboost `
  --feature-sets all classic vascular morphology `
  --variants original brightness_low brightness_high contrast_low contrast_high `
  --seeds 42 123 2025 `
  --folds 5 `
  --allow-unverified-identity `
  --allow-unverified-labels
```

As opções `--allow-unverified-identity` e `--allow-unverified-labels` autorizam somente a análise
provisória da base legada. Elas não transformam IDs sintéticos em identidades reais nem verificam o
diagnóstico.

## Fluxo de imagens brutas

Crie um manifesto CSV com uma linha por fotografia. As três últimas colunas do exemplo são opcionais
e representam medições clínicas externas:

```csv
sample_id,image_path,person_id,label,eye,diagnosis_source,diagnosis_verified,pupil_diameter_mm,iris_thickness_mm,ocular_measurement_source
sample-0001,images/person-001-left.jpg,person-001,0,L,clinical_record,true,4.2,0.48,AS-OCT
sample-0002,images/person-001-right.jpg,person-001,0,R,clinical_record,true,4.1,0.47,AS-OCT
sample-0003,images/person-002-left.jpg,person-002,1,L,clinical_record,true,3.8,0.42,UBM
```

`label` deve ser `0` para controle ou `1` para DM2. `person_id` é obrigatório e define os grupos da
validação cruzada. `diagnosis_source` registra a origem do rótulo clínico, e `diagnosis_verified`
registra se o projeto concluiu sua verificação externa. O software preserva esses campos, mas não
autentica prontuários.

A fotografia RGB permite calcular `pupil_diameter_px`, a razão entre os diâmetros da pupila e da íris
e a largura anular aparente em pixels ou em forma normalizada. Ela não mede a espessura anatômica da
íris. `iris_thickness_mm` deve vir de uma medição clínica externa, por exemplo, AS-OCT ou UBM.
`pupil_diameter_mm` também é uma coluna clínica opcional. Quando qualquer uma dessas colunas estiver
presente, `ocular_measurement_source` torna-se obrigatório. O código valida valores positivos e
preserva os nomes e as fontes no conjunto preparado.

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 preprocess `
  --manifest .\raw_manifest.csv `
  --output .\data\raw
```

O comando grava imagens normalizadas, características geométricas, manifesto, metadados e
`exclusions.json`. O manifesto de saída preserva o caminho de origem, o hash SHA-256 e a fonte do
diagnóstico. As segmentações devem satisfazer os controles de qualidade mínima, contenção dos
contornos, largura anular e limites da imagem. Falhas são registradas e excluídas sem substituição
silenciosa.

## Morfologia implementada

Sobre a íris normalizada, o código estima o contorno angular do colarete como uma função
$r(\theta)$. A implementação suaviza o contorno e calcula:

- raio normalizado do colarete;
- desvio RMS, desvio padrão e desvio máximo absoluto;
- segundo harmônico do contorno;
- curvatura RMS;
- contagem, profundidade média e profundidade máxima de sulcos;
- força da borda do colarete;
- densidade e raio de *blobs* como proxies de criptas;
- energias e bordas nas direções radial e angular.

Essas grandezas são proxies de imagem. A localização do colarete, a identificação de sulcos e os
proxies de criptas ainda exigem validação contra anotações humanas representativas.

## Saídas

| Arquivo | Finalidade |
|---|---|
| `dataset.npz` | Imagens, rótulos, grupos, proveniência, geometria opcional e metadados canônicos |
| `manifest.csv` | Amostra, pessoa, classe, olho, fonte do diagnóstico, caminho e hash da origem |
| `metadata.json` | Hashes, deduplicação, premissas e limitações |
| `features_<variant>.npz` | Matriz de características, nomes, famílias, amostras e perturbação |
| `metrics_by_fold.csv` | Métricas clássicas por modelo, conjunto, semente, *fold* e variante |
| `predictions.csv` | Predições por amostra para auditoria |
| `fold_assignments.csv` | Alocação agrupada no teste para cada semente |
| `summary.csv` | Média e desvio padrão entre *folds* e sementes |
| `run_config.json` | Versões, hashes, parâmetros e limitações da execução |

Resultados de execução são gerados localmente e não são versionados.

## Validação

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Os testes cobrem conversão restrita, remoção de duplicatas, segmentação, normalização *rubber sheet*,
esquemas de características, variantes fotométricas, isolamento dos grupos e métricas obrigatórias.

## Dataset Provenance and Retraction

The bundled archive is associated with the following retracted work and is retained only for local
provenance and implementation compatibility checks:

```bibtex
@inproceedings{iridology-icbme2018,
  author    = {Parsa Moradi and Naghme Nazer and Amirhosein Khasahmadi and Hoda Mohammadzadeh and Hasan Khojasteh Jafari},
  title     = {Discovering Informative Regions in Iris Images to Predict Diabetes},
  booktitle = {2018 25th National and 3rd International Iranian Conference on Biomedical Engineering},
  year      = {2018},
  doi       = {10.1109/ICBME.2018.8703564},
  note      = {Retracted. See DOI 10.1109/ICBME45317.2018.10207763}
}
```

IEEE marks the original article as `Retracted: Discovering Informative Regions in Iris Images to
Predict Diabetes`. The formal notice is `Retraction Notice: Discovering Informative Regions in Iris
Images to Predict Diabetes`, DOI `10.1109/ICBME45317.2018.10207763`.

The legacy README reports 88 diabetic and 108 control cases acquired under ophthalmologist supervision
at Farabi Hospital. The archive does not provide enough metadata to independently verify DM2 subtype,
identity, consent, licensing, or acquisition protocol. Do not redistribute it or use it as scientific
evidence without resolving those issues and the retraction with the data custodian.

Legacy ZIP members are checked for encryption, uncompressed size, compression ratio, and declared
size before restricted NumPy-only deserialization. Static opcode inspection is advisory; the runtime
allowlist is the enforcement boundary.

Canonical metadata is embedded in `dataset.npz`, so it contributes to the dataset hash. A human-readable
`metadata.json` sidecar is generated from the same object and must match the embedded copy when present.
