# Contexto do problema e histórico de desenvolvimento — Busca Local (EVRP / GR17)

Este documento reconstrói, em ordem cronológica, as decisões e descobertas que levaram à versão atual do módulo de Busca Local. Serve como material de apoio para a apresentação/defesa — explica *por que* cada escolha foi feita, não só *o que* foi feito.

---

## 1. O requisito do edital

A disciplina exige, para a modalidade Busca Local, a aplicação de **três algoritmos** ao mesmo problema de otimização:

1. **Hill Climbing com reinício aleatório** (obrigatório)
2. **Simulated Annealing** (obrigatório)
3. Uma **variante aprimorada** de um dos dois, pesquisada pela equipe (não vista em sala)

A comparação final precisa incluir múltiplas execuções (por causa da aleatoriedade), estatísticas de melhor/pior/média, e discussão de mínimos locais, platôs e dependência do estado inicial.

## 2. Escolha do problema: de TSP puro a EVRP

A ideia inicial era um TSP simples de roteamento de entregas, usando o dataset clássico **GR17** (TSPLIB, Groetschel) — 17 cidades, matriz de distância real, tour ótimo conhecido de custo 2085. Esse dataset foi escolhido por já ser uma fonte reconhecida na literatura, com matriz completa e gabarito de solução.

A equipe decidiu, no entanto, migrar para **EVRP (Electric Vehicle Routing Problem)**, adicionando a restrição de autonomia de bateria e a necessidade de recarga em pontos específicos do trajeto. Essa mudança tornou o problema mais realista (rotas de entrega com veículos elétricos), mas também mais complexo, já que o GR17 original não tem essa estrutura.

## 3. Busca por um dataset EVRP pronto — e por que foi descartada

Antes de adaptar o GR17, dois datasets EVRP prontos foram avaliados:

- **Schneider, Stenger & Goeke (2014)**, *"The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations"* (Transportation Science 48(4)) — o dataset E-VRPTW mais usado na literatura, com instâncias pequenas de 5/10/15 clientes. Descartado porque inclui **janelas de tempo** por cliente, uma restrição extra fora do escopo definido para o módulo.
- **EC-TSP dataset (Mendeley)**, de Gialos & Zeimpekis (2023) — dataset de Electric Capacitated TSP com dados reais de entregas na Grécia. Descartado por vir em formato Excel com múltiplas dimensões (capacidade, tempos de serviço) que exigiriam parsing desproporcional ao prazo disponível, além de não ser TSP puro.

Com autorização da professora para adaptar um dataset existente (documentando o processo), a equipe manteve o **GR17** e o adaptou manualmente para EVRP.

## 4. O mapeamento manual do GR17 para EVRP

Sem alterar **nenhuma distância** da matriz original, os 17 nós foram reclassificados em três papéis:

| Papel | Nós (1-indexado) |
|---|---|
| Depósito | 1 |
| Estações de recarga | 3, 9, 14 |
| Clientes | 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17 |

As estações foram posicionadas para se distribuir ao longo do tour ótimo de referência, evitando concentração num só trecho do percurso.

Parâmetros do modelo de bateria, definidos e justificados pela equipe:

- **Capacidade da bateria**: 800 unidades (mesma escala da matriz de distância). Justificativa: a maior aresta do grafo é 745 (2↔16), então 800 garante que qualquer trecho isolado seja sempre viável com bateria cheia.
- **Consumo**: 1:1 com a distância percorrida (sem fatores físicos — simplificação assumida e declarada).
- **Recarga**: sempre total (100%) ao passar por depósito ou estação.

A distância total do tour ótimo (~2085) é bem maior que a capacidade (800), forçando ao menos duas recargas ao longo do percurso — garantindo que a restrição de bateria seja de fato relevante, e não um TSP disfarçado.

## 5. O primeiro bug crítico: a penalidade em degrau

Ao implementar o Hill Climbing, a equipe descobriu que a recarga só ocorre ao **chegar** a um depósito/estação — não ao passar por um cliente. Um cliente "encaixado" entre duas paradas de recarga consome a **soma** das duas pernas de uma única carga, não cada perna isoladamente. O próprio tour ótimo original mostrou-se inviável sob essa regra (trecho 3→11→10→2→5→9 acumula 1118 de distância sem recarga intermediária, acima da capacidade de 800).

Além disso, a função de penalidade original — que somava um valor fixo enorme (`INFEASIBLE_PENALTY = 1.000.000`) mais o índice do trecho quebrado — tratava **toda** rota inviável como aproximadamente igual em custo, não importa se ela estivesse "quase viável" ou "muito inviável". Isso é uma função em **degrau**, sem gradiente: o Hill Climbing (que só aceita vizinhos melhores-ou-iguais) não tinha como "sentir" que uma modificação o aproximava da viabilidade, porque corrigir uma rota geralmente exige inserir **duas** estações em sequência ao redor do mesmo cliente — e cada inserção isolada, antes da segunda, só piorava o custo sem sinalizar progresso.

**Evidência do problema**: em testes com 200 reinícios × 5000 iterações (mais de 1 milhão de avaliações), **0 de 20 execuções** encontraram uma solução viável.

**A correção**: substituir a penalidade em degrau por uma **proporcional ao déficit total de bateria acumulado** ao longo de toda a rota (`compute_total_battery_deficit` + `INFEASIBLE_PENALTY_BASE = 1000`). Isso deu à busca um gradiente real — uma rota que estoura a bateria por 5 unidades passou a ter custo visivelmente menor que uma que estoura por 500.

**Resultado imediato da correção**: nas mesmas condições de teste, o Hill Climbing passou de 0/20 para **20/20 execuções viáveis**. Esse episódio tornou-se um dos pontos de análise crítica mais fortes do trabalho — mostra que a formulação da função objetivo importa tanto quanto o algoritmo de busca escolhido.

## 6. Implementação do Hill Climbing e do Simulated Annealing

Com a penalidade corrigida, foram implementados:

- **Hill Climbing com reinício aleatório** (`hill_climbing.py`): aceita vizinhos melhores-ou-iguais, reinicia após um número de iterações sem melhora estrita (estagnação). Depois foi criada `hill_climbing_fixed_budget`, que roda sob **orçamento fixo de avaliações** (não critério de estagnação) — usada quando é preciso comparar HC e SA de forma justa sob o mesmo número total de avaliações. A justificativa registrada para essa segunda versão: o critério de estagnação da função original "estoura" um pouco o orçamento nominal a cada reinício, e múltiplos reinícios amplificam essa variação, o que tornaria a comparação com o SA (execução única e contínua) injusta.
- **Simulated Annealing** (`simulated_annealing.py`): resfriamento geométrico (`T = T0 * cooling_rate^iteração`), aceitação de vizinhos piores com probabilidade `exp(-Δ/T)`.
- **Vizinhança** (`neighborhood.py`): três operações — troca de dois nós (`swap`), inserção de estação (`insert_station`), remoção de estação (`remove_station`) — reusadas pelos três algoritmos, garantindo que a comparação seja justa (mesma vizinhança, só muda a regra de aceitação).

## 7. Comparação inicial (n=10) e o alerta de amostra pequena

Um primeiro piloto comparou HC (sob orçamento equivalente ao SA, 20.000 avaliações) e SA puro em 10 execuções pareadas pela mesma seed (mesma rota inicial nos dois algoritmos — a única variável isolada era a regra de aceitação). Resultado:

| Métrica | Hill Climbing | Simulated Annealing |
|---|---|---|
| Execuções viáveis | 10/10 | 10/10 |
| Iteração média da 1ª viável | 562,7 (dp 603,1) | 773,5 (dp 212,4) |
| Custo final: mín / máx / média | 2610,0 / 3026,0 / 2834,9 | 2459,0 / 2890,0 / 2664,5 |

Teste de Wilcoxon pareado: **p≈0,027**, sugerindo vantagem estatisticamente significativa do SA (custo final ~6% menor em média).

Esse resultado foi tratado com cautela desde o início — n=10 é uma amostra pequena e sensível a outliers (o desvio padrão do HC, 603, era quase do tamanho da própria média, sinal de distribuição assimétrica puxada por duas execuções de convergência tardia). A decisão foi repetir o experimento com **n=30** antes de tirar qualquer conclusão.

## 8. O resultado com n=30: o efeito desaparece

Com n=30 (mais poder estatístico), o resultado mudou completamente: **p≈0,49** — sem diferença estatisticamente significativa entre HC e SA. As medianas de custo final ficaram praticamente empatadas (2680,0 vs. 2679,5).

Essa mudança confirmou que o resultado de n=10 era **ruído amostral**, não um efeito real — uma lição prática sobre por que tamanho de amostra importa antes de generalizar qualquer conclusão. Nessa mesma rodada, apareceram as duas primeiras execuções inviáveis dentro do orçamento oficial: seed 2037 no HC e seed 2055 no SA.

## 9. O segundo bug crítico: o schedule de resfriamento mal calibrado

Investigando por que HC e SA empataram, a equipe descobriu que o `cooling_rate=0.995` (fixo, escolhido sem cálculo) fazia a temperatura do SA cair abaixo de 1.0 já na iteração **1700 de 20.000** (8,5% do orçamento) — ou seja, o SA passava **~90% do orçamento total se comportando essencialmente como um Hill Climbing guloso**, sem nunca mais aceitar pioras.

Isso significava que o "empate" entre HC e SA não provava que Simulated Annealing não trazia vantagem — provava que a versão de SA testada colapsava em HC cedo demais.

**A correção**: uma fórmula analítica (`sa_utils.compute_cooling_rate`) que deriva o `cooling_rate` a partir de `T0`, `T_mínima` e `número de iterações`, garantindo que o resfriamento consuma o orçamento inteiro:

```
cooling_rate = (T_mínima / T0) ** (1 / max_iterações)
```

Com isso, `cooling_rate` passou de 0,995 (fixo) para ≈0,9995742 (calibrado para T₀=5000, T_mín=1, 20.000 iterações) — a nova temperatura só cruza o limiar na última iteração (100% do orçamento), contra 8,5% do schedule antigo.

**Resultado da recalibração**: com o schedule corrigido, o SA passou a superar o HC de forma clara e estatisticamente muito significativa (**p≈0,000055**), com desvio padrão de custo final caindo de 209 para 52 (muito mais consistente) e 58× mais aceitações de piora ao longo da execução (de 845 para 49.214 no agregado das 30 execuções).

## 10. A variante aprimorada: Simulated Annealing com reaquecimento

A variante escolhida foi **SA com reaquecimento**: quando a melhor solução global fica estagnada por um número de iterações (`stagnation_limit`), a temperatura é multiplicada por um fator (`reheat_factor`), permitindo à busca reabrir a exploração.

### Primeira tentativa: parâmetros exploratórios (stagnation_limit=500, reheat_factor=3.0)

Sob o schedule de resfriamento **antigo** (mal calibrado), o reaquecimento parecia "inerte" — sem efeito prático, porque a temperatura já estava perto de zero quando a estagnação disparava.

Ao trocar para o schedule **corrigido**, essa mesma combinação de parâmetros **quebrou o algoritmo**: como a temperatura ainda estava alta (na casa dos milhares) quando a estagnação disparava, multiplicá-la por 3 fazia `exp(-Δ/T)` disparar para perto de 1 para qualquer piora, transformando a busca em um passeio essencialmente aleatório que destruía o progresso já feito. Resultado: apenas **1 em 30 execuções** encontrou uma solução viável, com custo médio mais que o dobro do SA puro (5173 vs. 2536).

### Recalibração: stagnation_limit=1000, reheat_factor=1.3

Reduzir a frequência de reaquecimento e a intensidade do "empurrão" de temperatura recuperou o comportamento do algoritmo: **30 de 30 execuções** voltaram a encontrar viabilidade.

**Mas o SA com reaquecimento não superou o SA puro** — o teste de Wilcoxon entre os dois deu **p≈0,028** (significativo, mas o efeito é bem mais modesto que os outros dois pares testados). A hipótese levantada: num problema pequeno (17 nós) com um schedule de resfriamento já bem calibrado para consumir o orçamento inteiro, pode já haver tempo suficiente de exploração probabilística sem precisar de perturbações extras.

**A lição central deste episódio**: os parâmetros de uma variante (`reheat_factor`, `stagnation_limit`) não podem ser avaliados isoladamente — eles estão implicitamente acoplados a outras escolhas de design (aqui, o schedule de resfriamento do algoritmo base). O mesmo valor de `reheat_factor` pode ser inofensivo sob um schedule e catastrófico sob outro.

## 11. Resultado final da comparação de três vias (n=30)

| Algoritmo | Viabilidade | Custo médio | Custo mediano | Desvio padrão |
|---|---|---|---|---|
| Hill Climbing (orçamento fixo) | 29/30 | 2750,1 | 2680,0 | 215,3 |
| Simulated Annealing (schedule calibrado) | 30/30 | 2536,3 | 2530,0 | 51,9 |
| SA com reaquecimento (calibrado) | 30/30 | 2596,6 | 2556,0 | 125,7 |

Testes de Wilcoxon pareados:
- HC vs. SA puro: **p≈0,000055** (SA vence com folga)
- HC vs. SA-reaquecido: **p≈0,000489** (SA-reaquecido também vence o HC)
- SA puro vs. SA-reaquecido: **p≈0,028** (SA puro vence, mas com efeito modesto)

Esta é a **única** comparação usada para conclusões no notebook — os CSVs de etapas anteriores (schedule antigo, reaquecimento não calibrado) foram mantidos apenas como registro histórico do processo de calibração.

## 12. Investigação de casos de cauda longa

Duas seeds específicas — 2037 no HC, 2055 no SA — não encontraram viabilidade dentro do orçamento oficial de 20.000 avaliações. Ao rodar essas duas seeds com 5× o orçamento (100.000 avaliações, com o schedule recalibrado para esse novo orçamento), ambas convergiram (HC na iteração 44.026, SA na iteração 1.552), confirmando que não são "becos sem saída" do espaço de busca, mas sim casos de forte dependência do estado inicial — algo que o edital pede para ser discutido explicitamente.

## 13. Consolidação: notebook, testes e documentação

Todo o trabalho foi consolidado em:

- **`src/busca_local/`**: pacote Python de produção com 9 módulos (dataset, mapeamento EVRP, problema/função objetivo, vizinhança, Hill Climbing, Simulated Annealing, SA com reaquecimento, utilitário de calibração de temperatura, infraestrutura de experimentos).
- **`tests/`**: 37 testes automatizados cobrindo desde a validação do dataset até as propriedades estatísticas dos algoritmos (ex.: o melhor custo nunca piora ao longo da execução).
- **`notebooks/simulated_annealing_evrp.ipynb`**: a entrega principal — contextualização do EVRP, definição formal do problema, execução e discussão de cada algoritmo, comparação final de três vias com Wilcoxon, limitações e declaração de uso de IA generativa.
- **`resultados/`**: cinco CSVs — quatro históricos (etapas de calibração, mantidos para rastreabilidade) e um final (`comparacao_tres_vias_v2_reaquecimento_calibrado.csv`), única fonte de dados usada nas conclusões do notebook.
- **`docs/DECISOES.md`**: histórico de decisões de modelagem, o episódio da penalidade em degrau, e instruções de reprodução do ambiente.
- **`docs/RESUMO_ESTRUTURA.md`**: mapa completo de todos os arquivos do módulo (pastas, módulos, scripts, CSVs, testes), com a distinção explícita entre resultados históricos e finais.

A reprodutibilidade foi validada duas vezes de forma rigorosa: uma vez para os pacotes básicos (numpy, pytest, scipy) e outra, mais tarde, para o ambiente completo do notebook (pandas, matplotlib, jupyter) — em ambos os casos, destruindo o ambiente virtual e recriando do zero a partir apenas do `requirements.txt`, sem reaproveitar nada já instalado.

## 14. O que ainda falta (no momento da defesa intermediária)

- Vídeo de demonstração (1–3 min)
- Slides da apresentação oral
- Relatório final integrando as duas modalidades (Árvore de Decisão + Busca Local)
- Preenchimento dos placeholders do notebook: nomes da equipe, extensão da contribuição de IA generativa, confirmação da referência bibliográfica de curso (Russell & Norvig ou outra usada em sala)
- Limpeza de arquivos de ambiente local (venv, cache) antes do push para o GitHub

---

## Pontos fortes para destacar na defesa

1. **Dois bugs reais encontrados e corrigidos com evidência empírica** (penalidade em degrau, schedule de resfriamento mal calibrado) — não são hipóteses, são resultados medidos antes e depois da correção, com números exatos de antes/depois.
2. **Rigor estatístico**: a mudança de conclusão entre n=10 (p≈0,027) e n=30 (p≈0,49) é, por si só, uma demonstração prática de por que tamanho de amostra importa — e a equipe documentou isso em vez de esconder.
3. **Honestidade sobre a variante não superar o algoritmo base**: em vez de forçar uma conclusão de que o reaquecimento "funcionou", o resultado modesto (p≈0,028, efeito pequeno) foi reportado com a qualificação adequada, e a jornada de calibração (inerte → quebrou → recalibrado) virou parte da análise crítica.
4. **Transparência sobre a adaptação do dataset**: nenhuma distância foi inventada; só os papéis dos nós existentes foram reatribuídos, e isso está documentado com justificativa nó a nó.
5. **Reprodutibilidade validada de verdade**: ambiente recriado do zero (não só "funciona na minha máquina") duas vezes ao longo do projeto.

## Perguntas prováveis na arguição (e onde estão as respostas)

- "Por que EVRP e não TSP puro?" → Seção 2.
- "Por que esse dataset, e por que vocês mexeram nele?" → Seções 3 e 4; nota de integridade em `DECISOES.md` ("nenhuma distância foi alterada").
- "Como vocês decidiram a capacidade da bateria?" → Seção 4 (maior aresta = 745, capacidade = 800).
- "O Hill Climbing encontra o ótimo global?" → Não necessariamente — Seção 11 mostra que HC tem custo médio e variância piores que o SA; discutir mínimos locais e dependência do estado inicial (Seção 7 e 12).
- "O reaquecimento valeu a pena?" → Não, estatisticamente (Seção 10) — e essa é uma resposta esperada e defensável, não uma falha do trabalho.
- "Quantas execuções vocês rodaram e por quê?" → Seção 7 e 8 (a evolução de n=10 para n=30 e por quê).
- "Todos os algoritmos usam a mesma função de custo?" → Sim, `evaluate()` é única e compartilhada pelos três (ver `RESUMO_ESTRUTURA.md`, Seção 3).
