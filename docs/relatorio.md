# Fundamentos de Inteligência Artificial

---
title: Relatório de Fundamentos de Inteligência Artificial
subtitle: PL3
lang: pt-PT
author:
  - Alexandre Fonseca, nº 2022223121, uc2022223121@student.uc.pt
  - David Carvalheiro, nº 2022220112, uc2022220112@student.uc.pt
  - Luís Pedro de Sousa Oliveira Góis, nº 2018280716, uc2018280716@student.uc.pt
date: \today
---

## Modelação e desenvolvimento do Sistema de Produções

### Definição das Perceções e Ações

Perceções:

1. x: Posição horizontal da nave em relação ao centro (plataforma de aterragem).
   Negativa à esquerda; positiva à direita.
2. y: Posição vertical da nave em relação ao solo.
3. vx: Velocidade horizontal da nave. Negativa quando se move para a esquerda;
positiva quando se move para a direita.
4. vy: Velocidade vertical da nave. Negativa quando desce; positiva quando sobe.
5. theta: Orientação da nave. Negativa quando está inclinada para a direita;
positiva quando está inclinada para a esquerda.
6. vel_ang: Velocidade angular (mudança no ângulo). Negativa quando roda no sentido
horário; positiva quando roda no sentido anti-horário.
7. left_contact: Booleano (1 ou 0): Indica se a perna esquerda está em contato com o solo.
8. right_contact: Booleano (1 ou 0): Indica se a perna direita está em contato com o solo.

Ações:

1. Ativar o motor principal
2. Ativar um dos motores secundários

### Modelação do comportamento da nave através de um sistema de produções.

Sistema de produções:

```python
!include`snippetStart="def get_actions", snippetEnd="return action_np", includeSnippetDelimiters=True` src/main.py
```

### Sistema de Gerações:

Para obter o melhor valor possivel para o nosso sistema de produções, nós decidimos criar um sistema que conseguisse gerar esses valores automaticamente, e com uma elevada taxa de sucesso.
Inicialmente, considerámos um sistema que testava varias combinações de valores pré-defenidos, porem, este sistema demonstrava-se demasiado restristivo, ou demasiado pesado e demorado, estes motivos levaram-nos a criar um sistema que, gera 10 conjuntos valores iniciais aleatórios, e testa as taxas de sucessos dos mesmos, com estes valores, os 3 mais bem sucedidos são selecionados e a partir destes, são criados 3 novos valores com ligeiras mutações, estas mutações iram depender da taxa de sucesso do respetivos valores, e o 10 elemento é gerado a partir da média dos 3 valores obtidos, esta nova geração é por sua vez testada, e o processo repete-se no total de 30 gerações.
Durante as 30 gerações, os valores também são verificados, para ver qual destes teve a melhor taxa de sucesso, que será indicada no final.

## Resultados

  Resultados obtidos para 1000 episódios: 
  -> sem vento: ~73%
  -> com vento: ~43%

## Conclusão

O sistema desenvolvido demonstrou a eficácia dos agentes reativos baseados em regras para a aterragem lunar, alcançando uma taxa de sucesso de ~73% sem vento e ~43% com vento. A otimização através de gerações permitiu a melhoria progressiva das regras de decisão, garantindo um desempenho sólido em condições ideais.
No entanto, a abordagem puramente reativa tem limitações, especialmente em ambientes dinâmicos. O uso de memória poderia tornar o agente mais robusto, permitindo-lhe armazenar e utilizar informações passadas para melhorar a tomada de decisão.