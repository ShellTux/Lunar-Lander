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

Para este trabalho, estamos a usar um sistema de redes neuronais com 8
parametros iniciais, 12 camadas escondidos, e dois valores de output, que
complementamos com um ambiente evolucionario, que permitirá às redes neuronais
melhorarem num espaço de varias gerações. Cada rede neuronal contem uma
componente genotype, que contem os pesos para as redes neuronais, e uma
componente fitness, que, gerada atravez de um conjunto de diversos atributos,
contem o valor usado para identificar o nivel de sucesso da rede em si. Após
as redes serem criadas, estas são testadas num simulador, e após um certo
numero de simulações, são criados um certo numero de filhos, e atravez de um
metodo de seleção geral, são escolhidos as novas redes neuronais, e o sistema
repete-se por mais algumas gerações, até o final, onde irá colocar o melhor
resultado num ficheiro final

### Sistema de seleção (geral)

Para escolher as redes que passam para a proxima geração, primeiro são
acrescentados um certo numero dos melhores valores da seleção anterior (este
espaço é opcional), depois do qual, seram acrescentados os novos valores, sendo
estes crossovers, mutações, ou replicas do valores.

### Sistema de seleção (individual)

Para Selecionar os elementos desejados, o metodo usado para obter um elemento
aleatório é atravez do metodo torneiro. Por cada processo de seleção, onde são
selecionados 3 elementos aleatórios, e o maior destes é removido da lista de
elementos, depois de todos serem selecionados estes são reintroduzidos à lista
de elementos. Possiveis alternativas a este metodo poderam incluir, uma roleta
que escolhe valores aleatóriamente, com a confição que valores mais bem
succedidos contêm uma maior chance de ser escolhidos.

### Sistema de crossover

A forma de crossover utilizada, se ativada, cria um novo elemento a partir de
dois elementos diferentes e um crossover point, o crossover point é um valor
aleatório entre 0 e o tamanho do genotype, que é usado para dividir o genotype
dos dois elementos obtidos, obtendo a primeira parte do primeiro elemento, e a
segunda parte do segundo elemento. Outras possibilidades incluem a separação em
varios pontos diferentes aleatórios, alternando o elemento que usa para ir
buscar o genotype, ou usando um numero aleatório entre 0 e 1, e com base nesse,
acresce ao genotype do novo objeto o valor do genotype correpondente ao valor
aleatório, repetindo até ter o genotype completo.

### Sistema de mutação

Caso um objeto se mostre eligivel para mutação, se o valor de fitness for
inferior a um certo threshold, dois dos valores do genotype são modificados,
caso contrário, apenas um dos valores são alterados, o(s) valor(es) é/são
alterado(s) baseado num valor aleatório obtido a partir de uma distribuição
normal. Uma possivel alteração seria aumentar o numero de valores que poderiam
ser alterados, ou alterar a taxa de alteração da função normal com base da
fitness do elemento a ser mutado.

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

| Experiência | Crossover | Mutation | Elite Size | Média | Desvio Padrão |
| ---         | ---       | ---      | ---        | ---   | ---           |
| 1           | .5        | .008     | 0          | 4.4   | 2.42          |
| 2           | .5        | .05      | 0          | 6.0   | 4.82          |
| 3           | .9        | .008     | 0          | 10.6  | 8.11          |
| 4           | .9        | .05      | 0          | 10.2  | 7.47          |

Table: Distribuição das Taxas de Sucesso com vento


## Conclusão

Neste projeto, implementamos um algoritmo genético (GA) para resolver o
problema `Lunar Lander`, demonstrando a eficácia dos métodos evolutivos nas
tarefas de aprendizagem. Ao evoluir uma população de redes neurais por meio de
seleção, crossover e mutação, a nossa abordagem melhorou gradualmente o
desempenho da aterragem.

Os resultados destacam a adaptabilidade do GA na otimização de políticas de
controle para ambientes complexos, mesmo com recompensas esparsas. Embora o
algoritmo exigisse ajuste cuidadoso de hiperparâmetros, como tamanho da
população, taxa de mutação e critérios de condicionamento físico, ele se
mostrou capaz de alcançar aterragens estáveis.

O GA é mais adequado para problemas de otimização complexos, não diferenciáveis
ou ruidosos, onde os métodos baseados em gradiente lutam, como ajustar
hiperparâmetros, redes neurais em evolução ou resolver tarefas de controle com
recompensas esparsas. Eles se destacam em explorar espaços de pesquisa de alta
dimensão sem ficar presos no Optima local, tornando-os ideais para a otimização
de caixas negras. No entanto, eles podem ser computacionalmente caros, por isso
são mais práticos quando paralelizáveis ou quando gradientes exatos não estão
disponíveis. Evite o GA para problemas simples, suaves ou convexos, onde
existem métodos mais rápidos (por exemplo, SGD) e sabemos o erro.

No geral, este projeto ressalta o potencial dos algoritmos genéticos como uma
ferramenta versátil para resolver problemas desafiadores de controle.
