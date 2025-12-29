import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import imageio
import os

# ==============================================================================
# REDE NEURAL Q-NETWORK
# ==============================================================================


class QNetwork(nn.Module):
    """
    Rede Neural Profunda para aproximar a função Q(s, a)

    Arquitetura:
    - Camada de entrada: 4 neurônios (observações do ambiente)
    - Camadas ocultas: 2x 128 neurônios com ativação ReLU
    - Camada de saída: 2 neurônios (valor-Q para cada ação)
    """

    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.layers(x)


# ==============================================================================
# BUFFER DE EXPERIÊNCIAS
# ==============================================================================


class ReplayBuffer:
    """
    Buffer circular para armazenar transições (s, a, r, s', done)

    Implementa Experience Replay para:
    - Quebrar correlação temporal entre experiências consecutivas
    - Reutilizar experiências passadas para maior eficiência amostral
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, observation, action, reward, next_observation, terminated):
        """Adiciona uma transição ao buffer"""
        self.buffer.append((observation, action, reward, next_observation, terminated))

    def sample_batch(self, batch_size):
        """Retorna batch aleatório de transições"""
        transitions = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, terms = zip(*transitions)
        return obs, acts, rews, next_obs, terms

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# AGENTE DQN
# ==============================================================================


class DQNAgent:
    """
    Agente que implementa o algoritmo Deep Q-Network

    Componentes principais:
    - Q-network: rede que está sendo treinada
    - Q-target: rede alvo atualizada periodicamente para estabilidade
    - Replay buffer: memória de experiências passadas
    - Política epsilon-greedy: balanceamento exploração/exploração
    """

    def __init__(self, observation_dim, action_dim, config):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hiperparâmetros
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.target_sync_frequency = config["target_sync_frequency"]
        self.warmup_episodes = config["warmup_episodes"]

        # Redes neurais
        self.q_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.q_target = QNetwork(observation_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        # Otimizador e buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(config["buffer_capacity"])

        self.current_episode = 0

    def choose_action(self, observation, training=True):
        """
        Seleciona ação usando política epsilon-greedy

        Durante treinamento:
        - Com prob. epsilon: ação aleatória (EXPLORAÇÃO)
        - Com prob. 1-epsilon: melhor ação segundo Q-network (EXPLORAÇÃO)

        Args:
            observation: estado atual do ambiente
            training: se True, usa epsilon-greedy; se False, sempre greedy

        Returns:
            Índice da ação selecionada
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax().item()

    def store_transition(
        self, observation, action, reward, next_observation, terminated
    ):
        """Armazena transição no replay buffer"""
        self.replay_buffer.store(
            observation, action, reward, next_observation, terminated
        )

    def update_networks(self):
        """
        Atualiza Q-network usando mini-batch do replay buffer

        Passos do algoritmo:
        1. Amostra batch de transições do buffer
        2. Calcula Q(s,a) atual para ações tomadas
        3. Calcula Q-alvo usando equação de Bellman e Q-target
        4. Minimiza diferença (loss) via gradiente descendente
        5. Atualiza pesos da Q-network

        Returns:
            Valor da loss (ou None se buffer insuficiente)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Amostragem de experiências
        obs, acts, rews, next_obs, terms = self.replay_buffer.sample_batch(
            self.batch_size
        )

        # Conversão para tensores
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        acts_tensor = torch.LongTensor(acts).unsqueeze(1).to(self.device)
        rews_tensor = torch.FloatTensor(rews).unsqueeze(1).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
        terms_tensor = torch.FloatTensor(terms).unsqueeze(1).to(self.device)

        # Q-values para ações tomadas
        q_current = self.q_network(obs_tensor).gather(1, acts_tensor)

        # Cálculo do Q-alvo usando equação de Bellman
        with torch.no_grad():
            q_next_max = self.q_target(next_obs_tensor).max(1)[0].unsqueeze(1)
            # Q(s,a) = r + γ * max_a' Q(s',a') se não terminal
            q_target_value = rews_tensor + self.gamma * q_next_max * (1 - terms_tensor)

        # Loss e otimização
        loss = nn.MSELoss()(q_current, q_target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        """Sincroniza Q-target com Q-network (cópia de pesos)"""
        self.q_target.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduz epsilon gradualmente (menos exploração ao longo do tempo)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def should_train(self):
        """Verifica se agente já passou do período de warm-up"""
        return self.current_episode >= self.warmup_episodes

    def save_model(self, filepath):
        """Salva pesos da Q-network"""
        torch.save(self.q_network.state_dict(), filepath)

    def load_model(self, filepath):
        """Carrega pesos para Q-network"""
        self.q_network.load_state_dict(torch.load(filepath))
        self.q_network.eval()


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================


def record_episode_video(agent, env_name, episode_num, episode_reward):
    """
    Grava vídeo HD de um episódio completo

    Configurações otimizadas para YouTube:
    - 60 FPS para suavidade
    - Codec H.264 (compatibilidade universal)
    - CRF 18 (qualidade quase sem perdas)
    """
    env_render = gym.make(env_name, render_mode="rgb_array")
    observation, _ = env_render.reset()
    video_frames = []
    terminated = False

    # Executa episódio sem exploração
    while not terminated:
        frame = env_render.render()
        video_frames.append(frame)

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action = agent.q_network(obs_tensor).argmax().item()

        observation, _, term, trunc, _ = env_render.step(action)
        terminated = term or trunc

    # Adiciona frames finais para visualização do estado terminal
    for _ in range(30):
        video_frames.append(frame)

    env_render.close()

    # Salva vídeo em alta qualidade
    os.makedirs("videos", exist_ok=True)
    video_filename = (
        f"videos/episode_{episode_num:04d}_reward_{int(episode_reward)}.mp4"
    )
    imageio.mimsave(
        video_filename,
        video_frames,
        fps=60,
        codec="libx264",
        quality=10,
        pixelformat="yuv420p",
        output_params=["-crf", "18"],
    )
    print(f"Vídeo gravado: {video_filename}")


# ==============================================================================
# TREINAMENTO
# ==============================================================================


def train_dqn_agent():
    """Função principal de treinamento"""

    # Configurações
    config = {
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "buffer_capacity": 100000,
        "warmup_episodes": 10,
        "target_sync_frequency": 5,
        "num_episodes": 500,
        "video_frequency": 25,
    }

    # Inicialização
    env = gym.make("CartPole-v1")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(observation_dim, action_dim, config)

    print("TREINAMENTO DQN - CARTPOLE-V1")
    print(f"Total de episódios: {config['num_episodes']}")
    print(f"Sincronização Q-target: a cada {config['target_sync_frequency']} episódios")
    print(f"Gravação de vídeos: episódio 1 + múltiplos de {config['video_frequency']}")
    print(f"Período warm-up: {config['warmup_episodes']} episódios")
    print(f"Capacidade do buffer: {config['buffer_capacity']}")
    print("\n")

    # Loop de episódios
    for episode_idx in range(config["num_episodes"]):
        agent.current_episode = episode_idx

        # Reset do ambiente
        observation, _ = env.reset()
        episode_return = 0
        terminated = False

        # Loop de um episódio
        while not terminated:
            # Seleciona e executa ação
            action = agent.choose_action(observation)
            next_observation, reward, term, trunc, _ = env.step(action)
            terminated = term or trunc

            # Armazena transição
            agent.store_transition(
                observation, action, reward, next_observation, terminated
            )

            # Treina (se passou do warm-up)
            if agent.should_train():
                loss = agent.update_networks()

            observation = next_observation
            episode_return += reward

        # Atualiza epsilon
        agent.decay_epsilon()

        # Sincroniza Q-target periodicamente
        if episode_idx % config["target_sync_frequency"] == 0:
            agent.sync_target_network()

        # Log de progresso
        print(
            f"Ep {episode_idx+1:3d}/{config['num_episodes']} | "
            f"Return: {episode_return:5.0f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer):5d}"
        )

        # Grava vídeo periodicamente
        if episode_idx == 0 or (episode_idx + 1) % config["video_frequency"] == 0:
            record_episode_video(agent, "CartPole-v1", episode_idx + 1, episode_return)

    env.close()

    # Salva modelo final
    model_path = "dqn_cartpole_trained.pth"
    agent.save_model(model_path)

    print("\n")
    print("Treinamento finalizado!")
    print(f"Modelo salvo em: {model_path}")
    print(f"Vídeos disponíveis em: videos/")

    return agent


# ==============================================================================
# DEMONSTRAÇÃO DO AGENTE TREINADO
# ==============================================================================


def demonstrate_agent(agent):
    """Executa demonstração visual do agente treinado"""
    print("\nExecutando demonstração visual...")

    env_demo = gym.make("CartPole-v1", render_mode="human")
    observation, _ = env_demo.reset()
    terminated = False
    total_return = 0

    while not terminated:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action = agent.q_network(obs_tensor).argmax().item()

        observation, reward, term, trunc, _ = env_demo.step(action)
        terminated = term or trunc
        total_return += reward

    env_demo.close()
    print(f"Retorno obtido na demonstração: {total_return}")


if __name__ == "__main__":
    trained_agent = train_dqn_agent()
    demonstrate_agent(trained_agent)
