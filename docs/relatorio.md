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

Com base nas percepções e ações definidas, podemos começar com a criação de uma
base a partir da qual desenvolvemos o nosso sistema de produções.

Para começar, queremos que a nave se localize sobre a zona de aterragem, logo,
desejamos que se este estiver fora dessa região, que esteja em direção à mesma,
para isso podemos ativar um dos motores secundários de forma a que este esteja
suficientemente inclinado para que o motor principal seja suficiente para levar
a nave para dentro da área referida anteriormente, porém, não queremos que
esteja demasiado inclinada para a mesma, caso contrário, arriscamos que esta
perca controlo e não consiga aterrar com as pernas, por isso, caso esta fique
demasiado inclinada, devemos ativar o motor secundário para inverter a rotação
de forma a que volte a estar mais inclinada.

Agora que estamos dentro da zona de aterragem, devemos certificar que a nave
está o mais direita possível para a aterragem, usando os motores secundários
para a manter relativamente direita, também desejamos que a aterragem em si não
seja demasiado brusca, logo se a queda for demasiado alta, ativamos o motor
principal, para controlar a velocidade de queda.

Por último, mas não menos importante, a nave deve mover-se de forma descendente
o tempo todo. Com as bases do sistema criadas, o sistema de produções final
deverá ter o seguinte aspeto:

```python
!include`snippetStart="def get_actions", snippetEnd="return action_np", includeSnippetDelimiters=True` src/main.py
```

## Resultados

Resultados obtidos para 1000 episódios:

- sem vento: ~73%
- com vento: ~43%

## Conclusão

O sistema desenvolvido demonstrou a eficácia dos agentes reativos baseados em
regras para a aterragem lunar, alcançando uma taxa de sucesso de ~73% sem vento
e ~43% com vento. A otimização através de gerações permitiu a melhoria
progressiva das regras de decisão, garantindo um desempenho sólido em condições
ideais. No entanto, a abordagem puramente reativa tem limitações, especialmente
em ambientes dinâmicos. O uso de memória poderia tornar o agente mais robusto,
permitindo-lhe armazenar e utilizar informações passadas para melhorar a tomada
de decisão.
