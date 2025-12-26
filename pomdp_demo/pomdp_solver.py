import numpy as np
from typing import Tuple, List, Optional, Union
from environment import GridWorldPOMDP


class POMDPSolver:
    """
    Solver para POMDP usando Value Iteration simplificado
    Usa representação de belief state e computa política ótima
    """

    def __init__(self, env: GridWorldPOMDP, gamma: float = 0.95):
        """
        Args:
            env: Ambiente POMDP
            gamma: Fator de desconto
        """
        self.env = env
        self.gamma = gamma

        # Parâmetros do ambiente
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_observations = env.n_observations

        # Matrizes do modelo POMDP
        self.T = env.T  # Transição: T(s, a, s')
        self.O = env.O  # Observação: O(s', a, o)
        self.R = env.R  # Recompensa: R(s, a)

        # Value function sobre beliefs
        self.V = None  # Será array de vetores alpha
        self.policy = {}  # Mapeamento de belief para ação

    def solve(self, n_iterations: int = 20, sample_beliefs: int = 100):
        """
        Resolve POMDP usando Value Iteration

        Args:
            n_iterations: Número de iterações
            sample_beliefs: Número de belief states para amostrar
        """
        print(f"Resolvendo POMDP com {n_iterations} iterações...")

        # Gera conjunto de belief states para avaliar
        belief_points = self._sample_beliefs(sample_beliefs)

        # Inicializa value function
        # Cada alpha vector tem dimensão n_states
        alpha_vectors = [np.zeros(self.n_states)]
        actions = [0]  # Ação associada a cada alpha vector

        for iteration in range(n_iterations):
            print(f"Iteração {iteration + 1}/{n_iterations}")

            new_alpha_vectors = []
            new_actions = []

            # Para cada ação, computa novos alpha vectors
            for a in range(self.n_actions):
                # Backup de valor para ação a
                alpha_a = self._backup_action(a, alpha_vectors)
                new_alpha_vectors.append(alpha_a)
                new_actions.append(a)

            # Pruning: mantém apenas alpha vectors úteis
            alpha_vectors, actions = self._prune_alpha_vectors(
                new_alpha_vectors, new_actions, belief_points
            )

            print(f"  Alpha vectors: {len(alpha_vectors)}")

        self.V = alpha_vectors
        self.actions_map = actions

        print(f"Solução encontrada com {len(alpha_vectors)} alpha vectors")

    def _sample_beliefs(self, n_samples: int) -> List[np.ndarray]:
        """Amostra belief states uniformemente"""
        beliefs = []

        # Adiciona belief inicial
        beliefs.append(self.env.get_initial_belief())

        # Adiciona beliefs uniforme e aleatórios
        for _ in range(n_samples - 1):
            belief = np.random.dirichlet(np.ones(self.n_states))
            beliefs.append(belief)

        return beliefs

    def _backup_action(
        self, action: int, alpha_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Computa backup para uma ação específica
        Retorna novo alpha vector
        """
        # Recompensa imediata para ação
        r_a = self.R[:, action]

        # Computa valor esperado futuro para cada observação
        alpha_sum = np.zeros(self.n_states)

        for obs in range(self.n_observations):
            # Encontra melhor alpha vector para esta observação
            best_alpha = self._best_alpha_for_obs(action, obs, alpha_vectors)
            alpha_sum += best_alpha

        # Combina recompensa imediata com valor futuro
        new_alpha = r_a + self.gamma * alpha_sum

        return new_alpha

    def _best_alpha_for_obs(
        self, action: int, obs: int, alpha_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Encontra melhor alpha vector para observação específica
        Computa: sum_s' T(s,a,s') * O(s',a,o) * alpha(s')
        """
        if not alpha_vectors:
            return np.zeros(self.n_states)

        # Para cada alpha vector, computa valor esperado
        alpha_obs_list = []

        for alpha in alpha_vectors:
            # sum_s' T(s,a,s') * O(s',a,o) * alpha(s')
            alpha_obs = np.zeros(self.n_states)

            for s in range(self.n_states):
                for s_next in range(self.n_states):
                    alpha_obs[s] += (
                        self.T[s, action, s_next]
                        * self.O[s_next, action, obs]
                        * alpha[s_next]
                    )

            alpha_obs_list.append(alpha_obs)

        # Retorna alpha que maximiza valor esperado
        if not alpha_obs_list:
            return np.zeros(self.n_states)

        # Usa o alpha que maximiza o valor esperado (element-wise)
        alpha_stack = np.stack(alpha_obs_list, axis=0)
        return np.max(alpha_stack, axis=0)

    def _prune_alpha_vectors(
        self,
        alpha_vectors: List[np.ndarray],
        actions: List[int],
        belief_points: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Remove alpha vectors dominados
        Mantém apenas os que são ótimos para algum belief point
        """
        if not alpha_vectors:
            return [np.zeros(self.n_states)], [0]

        # Para cada belief point, encontra melhor alpha
        best_alphas_idx = set()

        for belief in belief_points:
            values = [np.dot(belief, alpha) for alpha in alpha_vectors]
            best_idx = np.argmax(values)
            best_alphas_idx.add(best_idx)

        # Mantém apenas alphas úteis
        pruned_alphas = [alpha_vectors[i] for i in sorted(best_alphas_idx)]
        pruned_actions = [actions[i] for i in sorted(best_alphas_idx)]

        return pruned_alphas, pruned_actions

    def get_action(self, belief: np.ndarray) -> int:
        """
        Retorna melhor ação para belief state atual

        Args:
            belief: Distribuição de probabilidade sobre estados

        Returns:
            action_idx: Índice da melhor ação
        """
        if self.V is None:
            raise ValueError("Solver não foi treinado. Execute solve() primeiro.")

        # Encontra alpha vector com maior valor para este belief
        values = [np.dot(belief, alpha) for alpha in self.V]
        best_idx = np.argmax(values)

        return self.actions_map[best_idx]

    def get_q_values(self, belief: np.ndarray) -> List[float]:
        """
        Retorna valor esperado por ação (belief dot alpha) para cada ação
        Usa o melhor alpha associado a cada ação
        """
        if self.V is None:
            raise ValueError("Solver não foi treinado. Execute solve() primeiro.")

        q_values = []
        for a in range(self.n_actions):
            # Filtra alphas dessa ação
            action_alphas = [
                alpha for alpha, act in zip(self.V, self.actions_map) if act == a
            ]
            if not action_alphas:
                q_values.append(0.0)
                continue

            values = [np.dot(belief, alpha) for alpha in action_alphas]
            q_values.append(float(np.max(values)))

        return q_values

    def update_belief(
        self,
        belief: np.ndarray,
        action: int,
        observation: Tuple[int, int],
        return_trace: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
        """
        Atualiza belief usando Bayes após observar ação e observação

        belief'(s') ∝ O(s',a,o) * sum_s T(s,a,s') * belief(s)

        Args:
            belief: Belief atual
            action: Ação executada
            observation: Observação recebida
            return_trace: Se True, retorna detalhes do cálculo por estado

        Returns:
            new_belief: Belief atualizado
            trace (opcional): Lista com detalhes por estado
        """
        obs_idx = self.env.obs_to_idx[observation]
        new_belief = np.zeros(self.n_states)
        trace = [] if return_trace else None

        for s_next in range(self.n_states):
            # P(o | s', a)
            prob_obs = self.O[s_next, action, obs_idx]

            # sum_s P(s' | s, a) * b(s)
            prob_transition = 0.0
            for s in range(self.n_states):
                prob_transition += self.T[s, action, s_next] * belief[s]

            unnormalized = prob_obs * prob_transition
            new_belief[s_next] = unnormalized

            if return_trace:
                trace.append(
                    {
                        "state": self.env.states[s_next],
                        "prob_obs": prob_obs,
                        "prob_transition": prob_transition,
                        "unnormalized": unnormalized,
                    }
                )

        # Normaliza
        belief_sum = np.sum(new_belief)
        if belief_sum > 0:
            new_belief /= belief_sum
            if return_trace:
                for item in trace:
                    item["normalized"] = item["unnormalized"] / belief_sum
        else:
            # Se normalização falhar, mantém belief uniforme
            new_belief = np.ones(self.n_states) / self.n_states
            if return_trace:
                uniform = 1.0 / self.n_states
                for item in trace:
                    item["normalized"] = uniform

        if return_trace:
            return new_belief, trace
        return new_belief

    def get_value(self, belief: np.ndarray) -> float:
        """Retorna valor esperado para belief state"""
        if self.V is None:
            return 0.0

        values = [np.dot(belief, alpha) for alpha in self.V]
        return np.max(values)
