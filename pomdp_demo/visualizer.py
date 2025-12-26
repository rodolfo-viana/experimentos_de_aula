import pygame
import numpy as np
from typing import Tuple, Optional, List
from environment import GridWorldPOMDP


class POMDPVisualizer:
    """Visualiza execução do POMDP em tempo real"""

    def __init__(self, env: GridWorldPOMDP, cell_size: int = 100):
        """
        Args:
            env: Ambiente POMDP
            cell_size: Tamanho de cada célula em pixels
        """
        self.env = env
        self.cell_size = cell_size

        # Dimensões da janela
        self.grid_width = env.size * cell_size
        self.grid_height = env.size * cell_size
        self.info_width = 420  # Largura do painel lateral
        self.width = self.grid_width + self.info_width  # Grid + painel lateral
        self.height = self.grid_height

        # Cores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.CYAN = (0, 255, 255)
        self.FOG = (30, 30, 30, 210)  # cor/alpha da neblina

        # Inicializa pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("POMDP Grid World - Demonstração")
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 36)

        self.clock = pygame.time.Clock()
        self.closed = False
        self.last_observation: Optional[Tuple[int, int]] = None

    def _get_visible_cells(
        self, observation: Optional[Tuple[int, int]]
    ) -> Optional[set]:
        """
        Retorna conjunto de células visíveis dado a observação atual.
        Inclui a célula observada e vizinhos imediatos (raio-1).
        """
        if observation is None:
            return None

        x, y = observation
        visible = set()
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.env.size and 0 <= ny < self.env.size:
                if (nx, ny) not in self.env.obstacles:
                    visible.add((nx, ny))
        return visible

    def _process_events(self):
        """Processa eventos do pygame para manter a janela responsiva"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def draw_grid(self):
        """Desenha o grid do ambiente"""
        # Preenche fundo
        self.screen.fill(self.WHITE)

        # Desenha células
        for x in range(self.env.size):
            for y in range(self.env.size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Cor da célula
                if (x, y) == self.env.goal_pos:
                    color = self.GREEN
                elif (x, y) in self.env.obstacles:
                    color = self.GRAY
                else:
                    color = self.WHITE

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 2)

                # Label para objetivo
                if (x, y) == self.env.goal_pos:
                    text = self.font.render("META", True, self.BLACK)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

    def draw_fog(self, visible_cells: Optional[set]):
        """Desenha neblina sobre células não visíveis"""
        if visible_cells is None:
            return

        fog_surface = pygame.Surface(
            (self.grid_width, self.grid_height), pygame.SRCALPHA
        )
        fog_surface.fill(self.FOG)

        # Recorta áreas visíveis (deixa transparente)
        for x, y in visible_cells:
            rect = pygame.Rect(
                x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(fog_surface, (0, 0, 0, 0), rect)

        self.screen.blit(fog_surface, (0, 0))

    def draw_agent(self, position: Tuple[int, int], color: Tuple[int, int, int] = None):
        """Desenha o agente em uma posição"""
        if color is None:
            color = self.BLUE

        x, y = position
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        pygame.draw.circle(
            self.screen, color, (center_x, center_y), self.cell_size // 3
        )

    def draw_observation(
        self, position: Tuple[int, int], true_position: Tuple[int, int]
    ):
        """Desenha observação recebida (pode ser diferente da posição real)"""
        x, y = position
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2

        # Se observação é diferente da posição real, desenha em amarelo
        if position != true_position:
            pygame.draw.circle(
                self.screen, self.YELLOW, (center_x, center_y), self.cell_size // 4, 3
            )
        else:
            # Usa um anel maior para não ser encoberto pelo agente
            pygame.draw.circle(
                self.screen, self.CYAN, (center_x, center_y), self.cell_size // 2 - 6, 4
            )

    def draw_belief(self, belief: np.ndarray):
        """Desenha belief state como heatmap no grid e barras laterais"""
        # Heatmap no grid
        for idx, prob in enumerate(belief):
            if prob > 0.01:  # Só desenha se probabilidade significativa
                state = self.env.states[idx]
                x, y = state

                # Intensidade da cor baseada na probabilidade
                alpha = int(255 * prob)
                overlay = pygame.Surface((self.cell_size, self.cell_size))
                overlay.set_alpha(alpha)
                overlay.fill(self.RED)

                self.screen.blit(overlay, (x * self.cell_size, y * self.cell_size))

                # Texto com probabilidade
                if prob > 0.03:  # Limiar menor para mostrar mais probabilidades
                    text = self.font_small.render(f"{prob:.2f}", True, self.BLACK)
                    text_rect = text.get_rect(
                        center=(
                            x * self.cell_size + self.cell_size // 2,
                            y * self.cell_size + self.cell_size // 2,
                        )
                    )
                    self.screen.blit(text, text_rect)

        # Painel lateral com barras
        panel_x = self.grid_width + 20
        panel_y = 20
        bar_width = 200
        bar_height = 15

        title = self.font_large.render("Belief State", True, self.BLACK)
        self.screen.blit(title, (panel_x, panel_y))
        panel_y += 50

        # Ordena estados por probabilidade
        sorted_indices = np.argsort(belief)[::-1][:10]  # Top 10

        for idx in sorted_indices:
            prob = belief[idx]
            if prob < 0.01:
                break

            state = self.env.states[idx]

            # Desenha barra
            bar_length = int(bar_width * prob)
            pygame.draw.rect(
                self.screen, self.RED, (panel_x, panel_y, bar_length, bar_height)
            )
            pygame.draw.rect(
                self.screen, self.BLACK, (panel_x, panel_y, bar_width, bar_height), 2
            )

            # Label
            label = self.font_small.render(f"{state}: {prob:.3f}", True, self.BLACK)
            self.screen.blit(label, (panel_x + bar_width + 10, panel_y))

            panel_y += bar_height + 5

    def draw_info(
        self,
        step: int,
        action: Optional[str],
        observation: Optional[Tuple[int, int]],
        reward: float,
        total_reward: float,
        done: bool,
        belief: Optional[np.ndarray] = None,
        q_values: Optional[List[float]] = None,
        belief_trace: Optional[List[dict]] = None,
    ):
        """Desenha painel de informações"""
        # Calcula entropia do belief (medida de incerteza)
        uncertainty_text = "N/A"
        if belief is not None:
            # Entropia: -sum(p * log(p))
            entropy = -np.sum(belief * np.log2(belief + 1e-10))
            # Normaliza pela entropia máxima
            max_entropy = np.log2(len(belief))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            uncertainty_text = f"{normalized_entropy:.2f} ({entropy:.2f} bits)"

        # Informações no painel lateral direito
        panel_x = self.grid_width
        panel_y = 0
        panel_w = self.info_width
        panel_h = self.grid_height

        # Fundo do painel
        pygame.draw.rect(
            self.screen, self.LIGHT_GRAY, (panel_x, panel_y, panel_w, panel_h)
        )

        info_x = panel_x + 10
        info_y = panel_y + 12

        # Cabeçalho
        title = self.font_large.render("Informações", True, self.BLACK)
        self.screen.blit(title, (info_x, info_y))
        y_offset = info_y + 34

        # Seção: Estado
        self.screen.blit(
            self.font.render("Estado", True, self.BLACK), (info_x, y_offset)
        )
        y_offset += 26
        status_text = f"Status: {'CONCLUÍDO!' if done else 'Em execução'}"
        state_lines = [
            f"Step: {step}",
            status_text,
            f"Ação: {action if action else 'N/A'}",
            f"Observação: {observation if observation else 'N/A'}",
            f"Reward: {reward:.1f}",
            f"Total Reward: {total_reward:.1f}",
            f"Incerteza: {uncertainty_text}",
        ]
        for line in state_lines:
            if "CONCLUÍDO" in line:
                text = self.font.render(line, True, self.GREEN)
            else:
                text = self.font.render(line, True, self.BLACK)
            self.screen.blit(text, (info_x, y_offset))
            y_offset += 24

        y_offset += 8

        # Seção: Q valores
        self.screen.blit(
            self.font.render("Q (belief · alpha)", True, self.BLACK), (info_x, y_offset)
        )
        y_offset += 24
        if q_values is not None:
            for i, name in enumerate(self.env.actions):
                val = q_values[i] if i < len(q_values) else 0.0
                text = self.font_small.render(f"{name}: {val:.2f}", True, self.BLACK)
                self.screen.blit(text, (info_x, y_offset))
                y_offset += 20
        else:
            self.screen.blit(
                self.font_small.render("N/A", True, self.BLACK), (info_x, y_offset)
            )
            y_offset += 20

        y_offset += 10

        # Seção: Atualização de belief
        if belief_trace:
            self.screen.blit(
                self.font.render("Atualização Belief (top 3)", True, self.BLACK),
                (info_x, y_offset),
            )
            y_offset += 22

            ordered = sorted(
                belief_trace, key=lambda x: x.get("normalized", 0), reverse=True
            )[:3]

            for item in ordered:
                state = item["state"]
                obs_prob = item["prob_obs"]
                trans = item["prob_transition"]
                norm = item.get("normalized", 0.0)
                text = self.font_small.render(
                    f"{state} obs={obs_prob:.2f} trans={trans:.2f} -> {norm:.2f}",
                    True,
                    self.BLACK,
                )
                self.screen.blit(text, (info_x, y_offset))
                y_offset += 20

        # Legenda
        legend_x = info_x
        legend_y = panel_y + panel_h - 130

        legend_items = [
            ("Agente (real)", self.BLUE),
            ("Observação correta", self.CYAN),
            ("Observação ruidosa", self.YELLOW),
            ("Belief (vermelho)", self.RED),
        ]

        self.screen.blit(
            self.font.render("Legenda:", True, self.BLACK), (legend_x, legend_y)
        )
        legend_y += 30

        for label, color in legend_items:
            pygame.draw.circle(self.screen, color, (legend_x + 10, legend_y + 10), 8)
            text = self.font_small.render(label, True, self.BLACK)
            self.screen.blit(text, (legend_x + 30, legend_y))
            legend_y += 25

    def render(
        self,
        agent_pos: Tuple[int, int],
        belief: np.ndarray,
        step: int,
        action: Optional[str] = None,
        observation: Optional[Tuple[int, int]] = None,
        reward: float = 0.0,
        total_reward: float = 0.0,
        done: bool = False,
        q_values: Optional[List[float]] = None,
        belief_trace: Optional[List[dict]] = None,
    ):
        """
        Renderiza frame completo

        Args:
            agent_pos: Posição real do agente
            belief: Belief state atual
            step: Número do step
            action: Última ação executada
            observation: Última observação recebida
            reward: Reward do último step
            total_reward: Reward acumulado
            done: Se episódio terminou
            q_values: Valores esperados por ação no belief atual
            belief_trace: Detalhes do cálculo de update de belief
        """
        if self.closed:
            return

        self._process_events()
        if self.closed:
            return

        # Atualiza última observação (mantém visão caso não haja nova)
        if observation is not None:
            self.last_observation = observation

        visible_cells = self._get_visible_cells(self.last_observation)

        # Desenha componentes
        self.draw_grid()
        self.draw_belief(belief)

        # Desenha agente
        self.draw_agent(agent_pos)

        # Desenha observação depois do agente para o anel ficar visível
        if observation:
            self.draw_observation(observation, agent_pos)

        # Aplica neblina para células não visíveis
        self.draw_fog(visible_cells)

        # Informações
        self.draw_info(
            step,
            action,
            observation,
            reward,
            total_reward,
            done,
            belief,
            q_values=q_values,
            belief_trace=belief_trace,
        )

        # Atualiza display
        pygame.display.flip()
        self.clock.tick(2)  # 2 FPS para visualização

    def get_frame(self) -> Optional[np.ndarray]:
        """Captura frame atual como array numpy (para salvar vídeo)"""
        if self.closed:
            return None

        # Converte surface do pygame para array numpy
        frame = pygame.surfarray.array3d(self.screen)
        # Pygame usa (width, height, 3), OpenCV usa (height, width, 3)
        frame = np.transpose(frame, (1, 0, 2))
        # Pygame usa RGB, OpenCV usa BGR
        frame = frame[:, :, ::-1]
        return frame

    def close(self):
        """Fecha visualizador"""
        if self.closed:
            return
        self.closed = True
        pygame.quit()
