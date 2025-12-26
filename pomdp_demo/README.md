# Demonstração de POMDP - Grid World com Neblina

Projeto educacional para demonstrar o funcionamento de um POMDP (Partially Observable Markov Decision Process) na prática.

## O que é POMDP?

Um POMDP é um modelo matemático para tomada de decisão sequencial em ambientes onde:
- O agente **não observa diretamente** o estado do ambiente
- As observações são **ruidosas ou parciais**
- O agente deve manter uma **crença (belief)** sobre o estado real
- A política ótima depende da distribuição de probabilidade sobre estados

## O Problema

Neste exemplo, um agente deve navegar em um grid 5x5 para alcançar um objetivo, mas:
- 🌫️ Há "neblina" - as observações têm 70% de chance de estar incorretas por padrão na demonstração (ajustável)
- 🧱 Existem obstáculos no caminho
- 🎯 O agente deve usar o histórico de observações para inferir sua posição real

## Componentes

### 1. Ambiente (`environment.py`)
- Grid World 5x5
- Estados: posições (x, y)
- Ações: cima, baixo, esquerda, direita
- Observações: posição observada (pode ser ruidosa)
- Ruído de transição: chance de escorregar para estado vizinho (slip)
- Modelo: T(s,a,s'), O(s',a,o), R(s,a)

### 2. Solver (`pomdp_solver.py`)
- Algoritmo: Value Iteration para POMDP
- Computa política ótima sobre belief states
- Usa alpha vectors para representar value function
- Atualiza belief usando filtro Bayesiano

### 3. Visualizador (`visualizer.py`)
- Visualização em tempo real com Pygame
- Mostra:
  - Grid com obstáculos e objetivo
  - Posição real do agente (círculo azul)
  - Observação recebida (círculo amarelo se incorreta, ciano se correta)
  - Belief state (heatmap vermelho no grid + barras laterais)
  - Informações (ação, reward, etc.)

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

## Uso

```bash
# Executar demonstração
python main.py
```

O script irá:
1. Criar o ambiente Grid World
2. Treinar o solver POMDP
3. Executar um episódio completo
4. Visualizar em tempo real
5. Salvar vídeo em `pomdp_demonstration.mp4`

## Interpretação da Visualização

### Grid
- ⬜ **Branco**: Células vazias (navegáveis)
- ⬛ **Cinza**: Obstáculos
- 🟩 **Verde**: Objetivo (META)

### Agente e Observações
- 🔵 **Círculo azul**: Posição real do agente
- 🟡 **Círculo amarelo (outline)**: Observação incorreta
- 🔷 **Círculo ciano (outline)**: Observação correta

### Belief State
- 🟥 **Heatmap vermelho**: Probabilidade de estar em cada célula
  - Mais intenso = maior probabilidade
- **Painel lateral**: Top 10 estados mais prováveis com barras

### Painel de Informações
- **Step**: Número do passo atual
- **Ação**: Última ação executada
- **Observação**: Posição observada
- **Reward**: Recompensa do último passo
- **Total Reward**: Recompensa acumulada
- **Status**: Em execução ou concluído

## Conceitos Demonstrados

### 1. Belief State
O agente mantém uma distribuição de probabilidade sobre sua posição real:
```
belief(s) = P(estado = s | histórico de observações)
```

### 2. Atualização de Belief (Filtro Bayesiano)
Após executar ação `a` e observar `o`:
```
belief'(s') ∝ O(s',a,o) × Σₛ T(s,a,s') × belief(s)
```

### 3. Seleção de Ação
A política ótima seleciona ações baseadas no belief, não no estado:
```
π*(b) = argmax_a Q*(b, a)
```

### 4. Value Iteration para POMDP
- Computa value function sobre belief space
- Usa alpha vectors para representação compacta
- Converge para política ótima

## Parâmetros Ajustáveis

Em `main.py`, você pode modificar:
- `grid_size`: Tamanho do grid (default atual: 7 para a demonstração; ambiente aceita qualquer valor)
- `observation_noise`: Probabilidade de observação incorreta (default do script: 0.7; default do ambiente: 0.2 se instanciado diretamente)
- `transition_noise`: Probabilidade de escorregar para outro estado (default do script: 0.2; default do ambiente: 0.2)
- `n_iterations`: Iterações de treinamento (default: 30)
- `gamma`: Fator de desconto (default: 0.95)
- `max_steps`: Máximo de passos por episódio (default: 50)

Em `environment.py`:
- `obstacles`: Lista de posições com obstáculos
- `start_pos`: Posição inicial
- `goal_pos`: Posição objetivo

## Arquitetura do Código

```
pomdp_demo/
├── environment.py      # Ambiente Grid World POMDP
├── pomdp_solver.py     # Solver com Value Iteration
├── visualizer.py       # Visualização com Pygame
├── main.py             # Script principal
├── requirements.txt    # Dependências
└── README.md           # Este arquivo
```