import numpy as np
from typing import Tuple, List, Optional


class GridWorldPOMDP:
    """
    Grid World com observabilidade parcial
    - Estados: posições (x, y) no grid
    - Ações: 0=cima, 1=direita, 2=baixo, 3=esquerda
    - Observações: posição observada (pode ter erro devido à neblina)
    """

    def __init__(
        self,
        size: int = 5,
        observation_noise: float = 0.2,
        transition_noise: float = 0.2,
    ):
        """
        Args:
            size: Tamanho do grid (size x size)
            observation_noise: Probabilidade de observação incorreta
            transition_noise: Probabilidade de escorregar para estado alternativo
        """
        self.size = size
        self.observation_noise = max(0.0, min(1.0, observation_noise))
        self.transition_noise = max(0.0, min(1.0, transition_noise))

        # Define posições especiais
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.obstacles = [
            (2, 2),
            (2, 3),
            (3, 2),
            (6, 4),
            (5, 4),
            (3, 6),
        ]  # Posições com obstáculos

        # Estados e ações
        self.states = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if (x, y) not in self.obstacles
        ]
        self.n_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        self.actions = ["up", "right", "down", "left"]
        self.n_actions = len(self.actions)

        # Observações são as mesmas que os estados
        self.observations = self.states.copy()
        self.n_observations = len(self.observations)
        self.obs_to_idx = {o: i for i, o in enumerate(self.observations)}

        # Estado atual
        self.current_state = self.start_pos

        # Construir modelo POMDP
        self._build_transition_model()
        self._build_observation_model()
        self._build_reward_model()

    def _build_transition_model(self):
        """Constrói matriz de transição T(s, a, s') com ruído de transição"""
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                desired_state = self._get_next_state(state, action)
                desired_idx = self.state_to_idx[desired_state]

                # Outros destinos possíveis (ficar parado + vizinhos válidos)
                slip_targets = set(self._get_neighbors(state) + [state])
                if desired_state in slip_targets:
                    slip_targets.remove(desired_state)

                if self.transition_noise > 0 and slip_targets:
                    base_prob = max(0.0, 1.0 - self.transition_noise)
                    slip_prob = self.transition_noise / len(slip_targets)

                    self.T[s_idx, a_idx, desired_idx] = base_prob
                    for t_state in slip_targets:
                        t_idx = self.state_to_idx[t_state]
                        self.T[s_idx, a_idx, t_idx] += slip_prob
                else:
                    # Transição determinística
                    self.T[s_idx, a_idx, desired_idx] = 1.0

    def _build_observation_model(self):
        """Constrói matriz de observação O(s', a, o)"""
        self.O = np.zeros((self.n_states, self.n_actions, self.n_observations))

        for s_idx, state in enumerate(self.states):
            for a_idx in range(self.n_actions):
                # Observação correta com prob (1 - noise)
                correct_obs_idx = self.obs_to_idx[state]
                self.O[s_idx, a_idx, correct_obs_idx] = 1.0 - self.observation_noise

                # Distribuir noise entre observações vizinhas
                neighbors = self._get_neighbors(state)
                if neighbors:
                    noise_per_neighbor = self.observation_noise / len(neighbors)
                    for neighbor in neighbors:
                        if neighbor in self.obs_to_idx:
                            neighbor_idx = self.obs_to_idx[neighbor]
                            self.O[s_idx, a_idx, neighbor_idx] = noise_per_neighbor
                else:
                    # Se não tem vizinhos, toda probabilidade vai para observação correta
                    self.O[s_idx, a_idx, correct_obs_idx] = 1.0

    def _build_reward_model(self):
        """Constrói função de recompensa R(s, a)"""
        self.R = np.zeros((self.n_states, self.n_actions))

        for s_idx, state in enumerate(self.states):
            for a_idx in range(self.n_actions):
                if state == self.goal_pos:
                    self.R[s_idx, a_idx] = 100.0  # Recompensa por alcançar objetivo
                else:
                    self.R[s_idx, a_idx] = -1.0  # Custo de movimento

    def _get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Retorna próximo estado dada ação"""
        x, y = state

        if action == "up":
            next_state = (x, max(0, y - 1))
        elif action == "down":
            next_state = (x, min(self.size - 1, y + 1))
        elif action == "left":
            next_state = (max(0, x - 1), y)
        elif action == "right":
            next_state = (min(self.size - 1, x + 1), y)
        else:
            next_state = state

        # Se próximo estado é obstáculo, fica no mesmo lugar
        if next_state in self.obstacles:
            return state
        return next_state

    def _get_neighbors(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Retorna estados vizinhos válidos"""
        x, y = state
        neighbors = []

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.size
                and 0 <= ny < self.size
                and (nx, ny) not in self.obstacles
            ):
                neighbors.append((nx, ny))

        return neighbors

    def reset(self) -> Tuple[int, int]:
        """Reseta ambiente ao estado inicial"""
        self.current_state = self.start_pos
        return self.current_state

    def step(
        self, action_idx: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int], float, bool]:
        """
        Executa ação no ambiente

        Returns:
            next_state: Próximo estado verdadeiro
            observation: Observação (pode ser ruidosa)
            reward: Recompensa recebida
            done: Se chegou ao objetivo
        """
        s_idx = self.state_to_idx[self.current_state]

        # Amostra próximo estado a partir de T(s,a,·)
        transition_probs = self.T[s_idx, action_idx]
        if transition_probs.sum() == 0:
            # Segurança: se algo deu errado na construção, mantém determinístico
            action = self.actions[action_idx]
            next_state = self._get_next_state(self.current_state, action)
            next_idx = self.state_to_idx[next_state]
        else:
            next_idx = np.random.choice(self.n_states, p=transition_probs)
            next_state = self.states[next_idx]

        # Gera observação ruidosa
        if np.random.random() < self.observation_noise:
            # Observação incorreta - escolhe vizinho aleatório
            neighbors = self._get_neighbors(next_state)
            if neighbors:
                observation = neighbors[np.random.randint(len(neighbors))]
            else:
                observation = next_state
        else:
            # Observação correta
            observation = next_state

        # Atualiza estado
        self.current_state = next_state
        done = next_state == self.goal_pos

        # Recompensa baseada no estado de destino
        reward = self.R[next_idx, action_idx]

        return next_state, observation, reward, done

    def get_initial_belief(self) -> np.ndarray:
        """Retorna belief inicial (certeza de estar no start_pos)"""
        belief = np.zeros(self.n_states)
        belief[self.state_to_idx[self.start_pos]] = 1.0
        return belief
