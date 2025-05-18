---
title: Relatório de Fundamentos de Inteligência Artificial
subtitle: PL3
lang: pt-PT
toc: true
author:
  - Alexandre Fonseca, nº 2022223121, uc2022223121@student.uc.pt
  - David Carvalheiro, nº 2022220112, uc2022220112@student.uc.pt
  - Luís Góis,         nº 2018280716, uc2018280716@student.uc.pt
date: \today
---

# Lunar Lander

Ficheiro: `src/main.py`

## Sistema Base

Neste trabalho, desenvolvemos um sistema baseado em redes neuronais, integradas num ambiente evolutivo.

Cada rede possui:

- 8 parâmetros de entrada, correspondentes à observação do ambiente;
- 12 neurónios em camadas escondidas;
- 2 valores de saída, utilizados para controlar as ações da nave.

Cada indivíduo da população evolutiva contém duas componentes principais:

- Genótipo: corresponde aos pesos da rede neuronal, representando o “ADN” do indivíduo;
- Fitness: valor numérico que reflete o desempenho da rede após simulação, calculado com base em critérios definidos (distância à base de aterragem, velocidade, estabilidade, etc.).
- O processo evolutivo decorre da seguinte forma:
  - Inicialização da população com redes neuronais com pesos aleatórios;
  - Avaliação de cada rede através de simulações no ambiente Lunar Lander;
  - Seleção dos melhores indivíduos com base no fitness;
  - Cruzamento (crossover) e mutação para gerar novos indivíduos (filhos);
  - Substituição dos indivíduos antigos por novos;

Repetição do processo por várias gerações.

No final, o melhor indivíduo (aquele com maior fitness) é guardado num ficheiro para análise posterior ou reutilização.


## Funções implementadas

```python
!include`snippetStart="def objective_function", snippetEnd="):", includeSnippetDelimiters=True` src/alexandre/main.py
```

Esta função avalia a **aptidão (fitness)** de cada indivíduo com base na **observação
final** do seu estado, após uma simulação no ambiente do Lunar Lander.
A observação é representada por um vetor com 8 componentes que descrevem a
posição, velocidade, orientação e estado de contacto da nave:

A função utiliza um **sistema de penalizações e recompensas**, cujo objetivo é guiar o algoritmo
evolutivo a favorecer indivíduos que exibam comportamentos desejáveis,
nomeadamente:

- Permanecer perto do centro da zona de aterragem.
- Apresentar velocidade vertical baixas.
- Manter a nave estável (sem inclinação excessiva).
- Estabelecer contacto com ambas as pernas (pontos por cada perna a fazer contacto).

A função incorpora um sistema de **recompensas condicionais** em níveis
hierárquicos: um indivíduo que aterre perto do centro recebe mais pontos do que
se este aterrasse numa região mais afastada se este estiver perto o suficiente
do centro, este começa a valorizar a sua velocidade vertical, que será
recompensado caso mantenha um valor baixo, se este for baixo o suficiente, é
acrescentado um novo parâmetro, neste caso o ângulo da nave relativamente ao
chão, que será recompensado caso os valores sejam pequenos, e assim por diante.
Funcionando com base num sistema de fitness cumulativo, consoante o indivíduo
se aproxime do objetivo final(aterragem bem sucedida). Este sistema progressivo
ajuda a guiar as redes para melhorias graduais ao longo das gerações,
priorizando um problema no sistema de cada vez, alterando essas bases para
adaptar aos novos valores, daí ser importante a segregação dos níveis.

O valor final de fitness é a soma das recompensas obtidas menos as
penalizações.

```python
!include`snippetStart="def parent_selection_tournament", snippetEnd="):", includeSnippetDelimiters=True` src/alexandre/main.py
```

Optou-se pelo método de **seleção por torneio** devido à sua simplicidade e
eficácia na introdução de pressão seletiva. Este algoritmo seleciona k
indivíduos aleatoriamente (neste caso, `k = 5`) da população e escolhe o melhor
entre eles, com base no valor de **fitness**. Este processo é repetido até obter o
número desejado de progenitores para o operador de crossover. Este método evita
o problema do escalonamento de fitness e permite uma boa diversidade genética,
mantendo uma seleção estável e controlada.


```python
!include`snippetStart="def crossover", snippetEnd="):", includeSnippetDelimiters=True` src/alexandre/main.py
```

O operador de **crossover** implementado baseia-se na técnica de **dois pontos
(two-point crossover)**. O genótipo de cada progenitor (vetor de pesos da rede
neuronal) é dividido em três partes:

- A primeira e última partes são herdadas de **p1**;
- A parte intermédia é herdada de **p2**.

Esta abordagem, representada simbolicamente por `p1 + p2 + p1`, permite uma
recombinação equilibrada dos genes dos dois progenitores, promovendo uma maior
diversidade nos descendentes e explorando melhor o espaço de soluções.


```python
!include`snippetStart="def mutation", snippetEnd="):", includeSnippetDelimiters=True` src/alexandre/main.py
```

A função de mutação implementa uma estratégia **gene a gene**. Cada gene (peso da
rede) tem uma certa **probabilidade de mutação** (`PROB_MUTATION`). Se for
selecionado, o valor do gene é modificado adicionando um valor aleatório obtido
de uma distribuição normal com média 0 e desvio padrão pré-definido.

Esta forma de mutação permite alterações suaves nos indivíduos, favorecendo a
exploração local do espaço de soluções e evitando perturbações drásticas que
possam comprometer indivíduos de alto desempenho.

## Resultados

Após a criação do sistema base, e de alguns testes de funcionalidade
irrelevantes, começamos a pôr o sistema em funcionamento, os resultados obtidos
foram os seguintes:

### Sem Vento

| Experiência | Taxas de sucesso (%) | Médias (%) | Desvio padrão |
| ---         | ---                  | ---        | ---           |
| 1           | 17, 35, 0, 3, 2      | 11.4       | 14.81         |
| 2           | 79, 76, 2, 0, 13     | 34.0       | 35.79         |
| 3           | 11, 79, 75, 0, 5     | 34.0       | 35.30         |
| 4           | 48, 16, 8, 51, 54    | 35.4       | 17.24         |
| 5           | 38, 4, 9, 1, 1       | 10.6       | 14.009        |
| 6           | 0, 1, 0, 8, 37       | 9.2        | 14.218        |
| 7           | 13, 2, 75, 6, 0      | 19.2       | 28.252        |
| 8           | 98, 1, 4, 57, 11     | 34.2       | 37.796        |

Table: Distribuição das Taxas de Sucesso sem vento

Como podemos observar, os valores das médias demonstram ser relativamente
baixos, porém, os desvios padrão destas demonstram ser, em todos menos um caso,
maior que a própria média. Isto pode indicar que o nosso método é extremamente
volátil, e errático, porém, também foi capaz de criar soluções extremamente
eficazes em várias ocasiões. Esta divisão pode originar de sistemas que
apresentaram dificuldade em ultrapassar uma certa etapa, logo igualmente
incapazes de melhorar noutra área.

### Com Vento

| Experiência | Taxas de Sucesso(%) | Médias(%) | Desvios Padrão |
| ---         | ---                 | ---       | ---            |
| 1           | 5, 6, 4, 0, 7       | 4.4       | 2,42           |
| 2           | 3, 1, 5, 6, 15      | 6.0       | 4, 82          |
| 3           | 21, 2, 9, 2, 19     | 10.6      | 8.11           |
| 4           | 5, 1, 12, 23, 10    | 10.2      | 7,46           |
| 5           | 0, 1, 0, 5, 4       | 2.0       | 2,098          |
| 6           | 0, 2, 0, 0, 6       | 1.6       | 2,332          |
| 7           | 4, 2, 13, 2, 14     | 7.0       | 5,367          |
| 8           | 54, 10, 14, 1, 82   | 32,2      | 30,831         |

Table: Distribuição das Taxas de Sucesso com vento

Similarmente à tabela 1, a tabela 2 demonstra ser incrivelmente erradica, tendo
desvios padrões superiores à própria média, porém, esta contêm valores
incrivelmente baixos, comparado com os itens da tabela 1 (exceto na experiência
8, que contém valores que se aproximam aos melhores da tabela 1), o que seria
de esperar, considerando o acréscimo de dificuldade que o vento propõe.


## Conclusão

Os resultados demonstram que o algoritmo genético foi capaz de encontrar
soluções para o problema da aterragem bem-sucedida, mas de forma bastante
instável. Mesmo na ausência de vento, as médias de sucesso revelaram-se
geralmente baixas, e os desvios padrão elevados indicam um comportamento
imprevisível: algumas execuções registaram desempenhos notáveis, enquanto
outras falharam por completo. Com a introdução de vento, o desempenho global
deteriorou-se, como seria de esperar devido ao aumento da complexidade do
ambiente. Ainda assim, a **experiência 8** destacou-se em ambos os cenários,
sugerindo que uma combinação de **taxas elevadas de crossover e mutação aliadas
ao elitismo** favorece a obtenção de melhores resultados.

Melhorias futuras poderão incluir o aumento do tamanho da população, número de
gerações e experiências, a refinação da função de fitness, e a reestruturação
do sistema evolutivo e dos seus mecanismos de seleção e variação.

Estes resultados refletem bem as propriedades típicas dos algoritmos genéticos:
são ferramentas poderosas para a exploração de soluções em espaços complexos,
mas altamente sensíveis à configuração dos seus parâmetros. A ocorrência de
soluções ótimas rodeadas por médias fracas evidencia o potencial do método, mas
também reforça a necessidade de um **maior controlo da diversidade populacional
e dos operadores evolutivos**, a fim de alcançar **maior consistência e
fiabilidade** nos resultados, minimizando a dependência de fatores aleatórios.

No geral, este projeto evidencia o **potencial dos algoritmos genéticos como
uma abordagem robusta e flexível** para resolver problemas desafiantes de
controlo, mesmo em ambientes dinâmicos e ruidosos como o do Lunar Lander.
