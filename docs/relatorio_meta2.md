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

## Sistema Base

Para este trabalho, estamos a usar um sistema de redes neuronais com 8 parametros iniciais, 12 camadas escondidos, e dois valores de output, que complementamos com um ambiente evolucionario, que permitirá às redes neuronais melhorarem num espaço de varias gerações.
Cada rede neuronal contem uma componente genotype, que contem os pesos para as redes neuronais, e uma componente fitness, que, gerada atravez de um conjunto de diversos atributos, contem o valor usado para identificar o nivel de successo da rede em si. Após as redes serem criadas, estas são testadas num simulador, e após um certo numero de simulações, são criados um certo numero de filhos, e atravez de um metodo de seleção geral, são escolhidos as novas redes neuronais, e o sistema repete-se por mais algumas gerações, até o final, onde irá colocar o melhor resultado num ficheiro final

### Sistema de seleção (geral)

Para escolher as redes que passam para a proxima geração, primeiro são acrescentados um certo numero dos melhores valores da seleção anterior (este espaço é opcional), depois do qual, seram acrescentados os novos valores, sendo estes crossovers, mutações, ou replicas do valores.

### Sistema de seleção (individual)

Para Selecionar os elementos desejados, o metodo usado para obter um elemento aleatório é atravez do metodo torneiro. Por cada processo de seleção, onde são selecionados 3 elementos aleatórios, e o maior destes é removido da lista de elementos, depois de todos serem selecionados estes são reintroduzidos à lista de elementos. Possiveis alternativas a este metodo poderam incluir, uma roleta que escolhe valores aleatóriamente, com a confição que valores mais bem succedidos contêm uma maior chance de ser escolhidos.

### Sistema de crossover

A forma de crossover utilizada, se ativada, cria um novo elemento a partir de dois elementos diferentes e um crossover point, o crossover point é um valor aleatório entre 0 e o tamanho do genotype, que é usado para dividir o genotype dos dois elementos obtidos, obtendo a primeira parte do primeiro elemento, e a segunda parte do segundo elemento. Outras possibilidades incluem a separação em varios pontos diferentes aleatórios, alternando o elemento que usa para ir buscar o genotype, ou usando um numero aleatório entre 0 e 1, e com base nesse, acresce ao genotype do novo objeto o valor do genotype correpondente ao valor aleatório, repetindo até ter o genotype completo.

### Sistema de mutação

Caso um objeto se mostre eligivel para mutação, se o valor de fitness for inferior a um certo threshold, dois dos valores do genotype são modificados, caso contrário, apenas um dos valores são alterados, o(s) valor(es) é/são alterado(s) baseado num valor aleatório obtido a partir de uma distribuição normal. Uma possivel alteração seria aumentar o numero de valores que poderiam ser alterados, ou alterar a taxa de alteração da função normal com base da fitness do elemento a ser mutado.

## Resultados

Por defenir

## Conclusão

Por defenir
