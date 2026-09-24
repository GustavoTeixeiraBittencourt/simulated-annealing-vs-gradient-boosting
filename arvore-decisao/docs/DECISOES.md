# Registro de decisões — Módulo Árvore de Decisão

Formato: decisão tomada, alternativas consideradas, por que escolhemos o que escolhemos, e o impacto esperado. Tudo aqui foi verificado empiricamente antes de virar decisão — nada foi assumido.

---

## 0. Mudança de dataset (contexto)

**Decisão:** abandonar o dataset Credit Card Fraud (Kaggle) e usar o German Credit / `credit-g` (OpenML).

**Alternativas consideradas:** manter o Credit Card Fraud, que era o plano original do README da raiz do repositório.

**Por quê:** o Credit Card Fraud tem atributos anonimizados por PCA (`V1`–`V28`), o que torna qualquer regra da árvore ininterpretável — impossível de confrontar com conhecimento de domínio, como o enunciado exige. Além disso, é extremamente desbalanceado (~0,17% de fraudes): uma árvore sem nenhuma divisão já ultrapassaria 99% de acurácia, tornando a "meta de acurácia" do enunciado trivial e sem sentido pedagógico.

**Impacto esperado:** o German Credit tem atributos legíveis (histórico de crédito, saldo em conta, duração e valor do empréstimo etc.) e desbalanceamento moderado (70/30), permitindo cumprir as exigências de explicabilidade e de meta de acurácia não trivial.

---

## 1. Implementação do C4.5

**Decisão:** usar a biblioteca `chefboost` (opção *a* do enunciado) para o C4.5, com um wrapper (`src/arvore/c45.py`) que corrige duas incompatibilidades de ambiente. Mantemos também uma árvore CART com `criterion="entropy"` do scikit-learn como referência, deixando explícito que ela **não é C4.5** (é CART: divisões sempre binárias, sem razão de ganho, com poda por complexidade de custo diferente da poda do C4.5 original).

**Alternativas consideradas:** (a) `chefboost`; (b) C4.5 didático implementado do zero.

**Verificação feita (antes de decidir):**
- Instalação: `pip install chefboost` funciona normalmente (venv descartável, Python 3.12, chefboost 0.0.19).
- Uso de razão de ganho de verdade: conferido no código-fonte (`chefboost/training/Training.py`), que calcula `split_info` (soma de `-p*log2(p)` sobre a distribuição de valores do atributo) e divide `gain / split_info` — exatamente a definição de razão de ganho do C4.5, não apenas ganho de informação renomeado.
- Validação contra o exemplo clássico Play Tennis (14 instâncias): a raiz escolhida foi `Outlook`, como esperado nos livros-texto.
- Teste no dataset real (`credit-g`, 800 linhas de treino): treina em menos de 1 segundo, faz corte por limiar em atributos numéricos corretamente, chega a ~72-76% de acurácia dependendo da profundidade — resultado plausível.

**Duas armadilhas de ambiente encontradas e corrigidas no wrapper:**
1. **Dtype de texto do pandas moderno.** A chefboost decide se uma coluna é numérica ou categórica checando `dtype == object`. Pandas recentes (testamos com pandas 3.0.6) usam por padrão um dtype `string`/`category` diferente do `object` clássico para colunas de texto, e a chefboost quebra com `TypeError: Cannot perform reduction 'std' with string dtype` ao tentar calcular estatísticas sobre essas colunas como se fossem numéricas. **Correção:** convertemos explicitamente só as colunas de texto (nunca as numéricas!) para `dtype=object` antes de treinar (`dados.preparar_para_c45`). Convertê-las também nas colunas numéricas quebra a detecção de atributo contínuo da chefboost (ela passa a tratar `credit_amount` como categórico, criando um ramo por valor exato — testamos e isso de fato aconteceu, gerando uma árvore com centenas de ramos e 0% de acerto em valores nunca vistos).
2. **Import dinâmico fora de notebook.** A chefboost salva as regras aprendidas em `outputs/rules/rules.py` (relativo ao diretório de trabalho atual) e as importa de volta via `importlib.util.find_spec`, que exige que esse diretório esteja no `sys.path`. Isso já acontece automaticamente em notebooks/Colab, mas não ao rodar um script `.py` puro — nesse caso é preciso `sys.path.insert(0, os.getcwd())` antes de chamar `chef.fit`. O wrapper faz isso automaticamente.

**Limitação documentada (não escondida):** a chefboost só expõe `max_depth` como controle de complexidade para o C4.5 — não há `min_samples_leaf` nem uma poda pós-treino de verdade (o "pruning condition" citado no código-fonte é, na prática, apenas a parada por profundidade máxima, não a poda pessimista por erro do C4.5 original de Quinlan). Por isso a busca da árvore mais enxuta do C4.5 (`busca_enxuta.buscar_arvore_enxuta_c45`) é 1-D (só profundidade), enquanto a do CART é 2-D (profundidade × mínimo de amostras por folha). Isso é uma diferença real entre "C4.5 de uma biblioteca pronta" e "CART do scikit-learn com controle fino de hiperparâmetros" que vale a pena citar na comparação final.

**Impacto esperado:** conseguimos usar uma implementação de C4.5 genuína (com razão de ganho, ramos multi-valorados e limiares numéricos) sem reimplementá-la, cumprindo o enunciado ("não é necessário implementar os algoritmos do zero"), ao custo de um wrapper de ~15 linhas para lidar com as duas incompatibilidades acima. As métricas de divisão (entropia, ganho, razão de ganho) foram implementadas do zero em `metricas_divisao.py` de qualquer forma, pois são exigidas para a explicação passo a passo do enunciado e são testadas independentemente da chefboost.

---

## 2. Meta de acurácia

**Decisão:** meta de **71%**.

**Baselines medidas primeiro (antes de fixar a meta):**
- Classe majoritária ("good") no treino: **70,0%**.
- Árvore de profundidade 1 (uma única pergunta), acurácia média de validação cruzada no treino: **70,0%** — ou seja, a melhor pergunta única isolada não supera, na prática, a linha de base trivial.

**Verificação exploratória do teto alcançável:** antes de fixar a meta definitiva, rodamos uma checagem rápida do teto de acurácia de validação cruzada (5 dobras) de um CART com `criterion="entropy"`, variando profundidade (1 a 8, e sem limite) e mínimo de amostras por folha (1, 5, 10, 20, 40). O melhor resultado observado foi **72,0%** (profundidade 4, mínimo de 20 amostras por folha).

**Por quê 71% e não outro valor:** uma meta "redonda" e mais ambiciosa (ex.: 75% ou 80%) exigiria uma árvore mais profunda — e portanto menos enxuta — só para ser alcançada, ou simplesmente seria inatingível por uma única árvore neste dataset (o teto empírico é ~72%). 71% fica acima das duas baselines (não é trivial de bater) e dentro do que uma árvore enxuta consegue de fato entregar, deixando ainda uma margem pequena de escolha real entre configurações na busca da árvore mais enxuta (várias combinações de profundidade/folhas atingem entre 71% e 72%).

**Impacto esperado:** a meta é desafiadora o suficiente para forçar uma escolha de complexidade não trivial, mas realista o suficiente para não nos obrigar a relaxar o critério de "árvore enxuta" depois de ver os resultados.

---

## 3. Protocolo de avaliação

**Decisão:** divisão treino/teste estratificada 80/20 com semente fixa (`config.SEMENTE_PRINCIPAL = 42`). A escolha de hiperparâmetros (profundidade, mínimo de amostras por folha) usa validação cruzada de 5 dobras **somente no treino**. O conjunto de teste é usado uma única vez, no final, para reportar o resultado da configuração já escolhida.

**Por que não escolher hiperparâmetros olhando o teste (explicação simples):** se testássemos várias profundidades e mínimos de amostras e escolhêssemos a combinação que acerta mais no conjunto de teste, estaríamos "espiando a prova antes de responder" — encontraríamos, por tentativa e erro, a configuração que por acaso acerta mais aquelas 200 linhas específicas, não necessariamente a que generaliza melhor para clientes novos. O número final pareceria ótimo, mas seria uma ilusão criada pela própria busca. Usar validação cruzada no treino simula "dados nunca vistos" sem gastar o conjunto de teste, que fica reservado para uma medição honesta no final.

**Impacto esperado:** o número de acurácia de teste reportado no notebook é uma estimativa razoável de desempenho em dados novos, não um número inflado pela própria busca de hiperparâmetros.

---

## 4. Busca da árvore mais enxuta

**Decisão:** grade de busca para o CART: profundidade máxima em `{1, 2, 3, 4, 5, 6, 7, 8, sem limite}` × mínimo de amostras por folha em `{1, 5, 10, 20, 40}`. Para o C4.5 (chefboost), grade 1-D só de profundidade (ver limitação da biblioteca no item 1). Critério de escolha: entre as configurações que atingem a meta na validação cruzada, escolher a de **menor número de folhas**; em caso de empate, a de **menor profundidade**.

**Por quê esse critério de desempate:** o enunciado pede a árvore "mais enxuta", e número de folhas é a medida mais direta de quantas regras distintas a árvore realmente usa (duas árvores de mesma profundidade podem ter números de folhas bem diferentes). Profundidade como segundo critério desempata os casos remanescentes e também limita o comprimento máximo de qualquer regra individual.

**Impacto esperado:** a árvore final não é a mais acurada possível, e sim a mais simples entre as que já atingem o que definimos como "bom o suficiente" — coerente com a meta da modalidade ("obter boa taxa de acertos sem produzir uma árvore desnecessariamente complexa").

**Achado que não escondemos: o C4.5 (chefboost) nunca atinge a meta neste dataset.** Rodamos a busca de profundidade do C4.5 de 1 a 10 e a acurácia de validação cruzada **piora monotonicamente** conforme a árvore cresce:

| Profundidade | Acurácia CV (treino) | Nº de folhas |
|---:|---:|---:|
| 1 | 70,3% | 3 |
| 2 | 69,1% | 7 |
| 3 | 69,3% | 12 |
| 4 | 68,6% | 18 |
| 5 | 67,9% | 27 |

O melhor resultado do C4.5 (profundidade 1, 70,3%) fica abaixo até da baseline da classe majoritária (70,0% + arredondamento) e não atinge a meta de 71%. A causa mais provável (coerente com o que já sabíamos do item 1): a chefboost não faz poda de verdade, e cada divisão categórica cria um ramo por categoria — com atributos de muitas categorias (`purpose` tem 10, `personal_status` tem 4, etc.) isso fragmenta rapidamente o conjunto de treino em subgrupos pequenos e ruidosos, sobreajustando já em profundidades rasas. O CART, por comparação, controla isso com `min_samples_leaf` (que o C4.5 desta biblioteca não tem) e com cortes sempre binários, que fragmentam os dados mais devagar.

**Como lidamos com isso:** `busca_enxuta.buscar_arvore_enxuta_c45` não lança exceção nesse caso — devolve o candidato de melhor acurácia entre os testados (profundidade 1), marcado com `atinge_meta=False`. No notebook, isso é reportado como um resultado real da comparação, não escondido: **a árvore mais enxuta que atinge a meta, neste dataset, vem do CART, não do C4.5**. Isso não invalida o C4.5 como algoritmo — é uma limitação desta implementação específica (sem poda), que vale mais a pena expor do que mascarar trocando de meta ou de biblioteca a essa altura.

---

## 5. Pré-processamento

**Decisão:** para o C4.5 (chefboost), manter os atributos categóricos nativos (sem codificação) e os numéricos com seu dtype original, deixando a própria biblioteca decidir o limiar de corte nos numéricos. Para o CART do scikit-learn, codificar os atributos categóricos com **one-hot encoding** (`pd.get_dummies`), não codificação ordinal.

**Alternativas consideradas para o CART:** codificação ordinal (0, 1, 2, ... por categoria).

**Por quê one-hot e não ordinal:** o CART do scikit-learn só faz cortes binários numéricos (`atributo <= valor`). Com codificação ordinal, um corte como "`checking_status <= 1.5`" implicaria uma ordem entre categorias (`<0` < `0<=X<200` < `>=200` < `no checking`, por exemplo) que não existe de fato nos dados — a regra ficaria tecnicamente correta, mas sem significado de domínio, e portanto ininterpretável. Com one-hot, o mesmo corte vira "`checking_status_no_checking <= 0.5`", que se traduz diretamente para "o cliente **não** tem conta corrente" — uma afirmação categórica legível, sem inventar uma ordem inexistente.

**Rótulos compreensíveis:** o dataset `credit-g` do OpenML já decodifica os atributos originais do UCI (que vêm como códigos abreviados tipo `A11`, `A12`, ...) para strings descritivas em inglês (ex.: `checking_status = "no checking"`, `purpose = "new car"`). Ainda assim, construímos um dicionário de tradução para português (`src/arvore/regras.py::TRADUCAO_ATRIBUTOS` e `TRADUCAO_CATEGORIAS`) para que as regras finais no notebook fiquem em português claro, não em inglês técnico.

**Impacto esperado:** as regras extraídas de ambas as árvores continuam legíveis por um humano, sem introduzir relações numéricas artificiais entre categorias.

---

## 6. Ensemble (Gradient Boosting)

**Decisão:** `GradientBoostingClassifier` do scikit-learn com `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, mesma semente, mesmos dados (versão one-hot, igual ao CART) e mesma divisão treino/teste da árvore única.

**Por quê essa configuração e não uma busca de hiperparâmetros:** o objetivo desta modalidade é comparar a **estratégia** de ensemble com a de árvore única (conforme o enunciado), não extrair o último ponto percentual de acurácia do Gradient Boosting. `max_depth=3` mantém cada árvore do ensemble rasa (o padrão do boosting: muitas árvores fracas, não uma árvore forte), e `n_estimators=100`/`learning_rate=0.1` são os valores-padrão do scikit-learn, amplamente documentados e um ponto de partida razoável.

**Impacto esperado:** comparação justa (mesmos dados, mesma divisão, mesma semente) entre uma árvore única enxuta e um ensemble não otimizado ao extremo — o que já é suficiente para discutir o compromisso entre desempenho e interpretabilidade.

---

## 7. Limitações da acurácia

**Decisão:** reportar, além da acurácia, a matriz de confusão e os dois tipos de erro separadamente (falso negativo = mau pagador aprovado; falso positivo = bom pagador recusado), destacando que eles têm custos diferentes.

**Achado empírico que reforça essa decisão:** a árvore CART mais enxuta escolhida pela busca (profundidade 3, 8 folhas) acertou apenas **4 de 60** maus pagadores reais no conjunto de teste, mesmo tendo 70,5% de acurácia geral — a acurácia alta esconde que o modelo praticamente sempre prevê "bom pagador". Isso ilustra concretamente, com nosso próprio resultado, por que acurácia sozinha é enganosa neste problema: um modelo que erra sistematicamente na classe mais cara de errar (aprovar crédito a quem não vai pagar) ainda parece "bom" pela acurácia.

**Sobre a matriz de custo original do dataset:** o German Credit (UCI) define uma matriz de custo em que classificar um mau pagador como bom custa 5× mais que o contrário — não a usamos para treinar (o enunciado não pede otimização de custo, só que a limitação seja reconhecida), mas ela reforça por que os falsos negativos aqui importam mais que a acurácia agregada.

**Impacto esperado:** o relatório final não trata a acurácia como a única medida de sucesso, e discute explicitamente essa limitação em vez de escondê-la.
