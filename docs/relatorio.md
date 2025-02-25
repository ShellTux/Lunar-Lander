# Fundamentos de Inteligência Artificial

---
title: Relatório de Fundamentos de Inteligência Artificial
subtitle: PL3
lang: pt-PT
author:
  - Alexandre Fonseca
  - David Carvalheiro
  - Luís Pedro de Sousa Oliveira Góis, nº 2018280716, uc2018280716@student.uc.pt
date: \today
---

## Modelação e desenvolvimento do Sistema de Produções

### Definição das Perceções e Ações

Perceções:

1. Posição horizontal da nave em relação ao centro (plataforma de aterragem).
   Negativa à esquerda; positiva à direita.
2. Posição vertical da nave em relação ao solo.
3. Velocidade horizontal da nave. Negativa quando se move para a esquerda;
positiva quando se move para a direita.
4. Velocidade vertical da nave. Negativa quando desce; positiva quando sobe.
5. Orientação da nave. Negativa quando está inclinada para a direita;
positiva quando está inclinada para a esquerda.
6. Velocidade angular (mudança no ângulo). Negativa quando roda no sentido
horário; positiva quando roda no sentido anti-horário.
7. Booleano (1 ou 0): Indica se a perna esquerda está em contato com o solo.
8. Booleano (1 ou 0): Indica se a perna direita está em contato com o solo.

Ações:

1. Ativar o motor principal
2. Ativar o motor secundário

### Modelação do comportamento da nave através de um sistema de produções.
