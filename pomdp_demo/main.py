import numpy as np
import cv2
import os
from environment import GridWorldPOMDP
from pomdp_solver import POMDPSolver
from visualizer import POMDPVisualizer


def run_episode(
    env: GridWorldPOMDP,
    solver: POMDPSolver,
    visualizer: POMDPVisualizer,
    max_steps: int = 50,
    save_video: bool = True,
    video_path: str = "pomdp_demo.mp4",
):
    """
    Executa um episódio completo no ambiente POMDP

    Args:
        env: Ambiente
        solver: Solver POMDP treinado
        visualizer: Visualizador
        max_steps: Número máximo de passos
        save_video: Se deve salvar vídeo
        video_path: Caminho para salvar vídeo
    """
    TARGET_W, TARGET_H = 1920, 1080
    TARGET_FPS = 30
    STEP_HOLD_FRAMES = 10  # mantém cada frame por ~0.33s (10/30)
    END_HOLD_SECONDS = 2

    def fit_to_1080p(frame):
        """Redimensiona e adiciona padding para 1920x1080."""
        h, w, _ = frame.shape
        scale = min(TARGET_W / w, TARGET_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=frame.dtype)
        offset_x = (TARGET_W - new_w) // 2
        offset_y = (TARGET_H - new_h) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        return canvas

    # Inicializa ambiente
    state = env.reset()
    belief = env.get_initial_belief()

    # Variáveis de controle
    done = False
    step = 0
    total_reward = 0.0
    frames = []

    # Visualiza estado inicial
    visualizer.render(
        agent_pos=state, belief=belief, step=step, total_reward=total_reward
    )

    if save_video:
        frame = visualizer.get_frame()
        if frame is not None:
            padded = fit_to_1080p(frame)
            frames.extend([padded] * STEP_HOLD_FRAMES)

    print("\n" + "=" * 60)
    print("Iniciando execução do episódio")
    print("=" * 60)

    while not done and step < max_steps:
        # Valores esperados por ação (para visualização)
        q_values = solver.get_q_values(belief)

        # Seleciona ação baseada no belief atual
        action_idx = int(np.argmax(q_values))
        action_name = env.actions[action_idx]

        print(f"\nStep {step + 1}:")
        print(f"  Belief (top 3): ", end="")
        top_beliefs = np.argsort(belief)[-3:][::-1]
        for idx in top_beliefs:
            if belief[idx] > 0.01:
                print(f"{env.states[idx]}:{belief[idx]:.3f} ", end="")
        print()
        print(f"  Ação escolhida: {action_name}")
        print(
            f"  Q valores: "
            + " ".join(
                f"{env.actions[i]}:{q_values[i]:.2f}" for i in range(len(q_values))
            )
        )

        # Executa ação
        next_state, observation, reward, done = env.step(action_idx)

        print(f"  Estado real: {next_state}")
        print(f"  Observação: {observation}")
        print(f"  Reward: {reward:.1f}")

        # Atualiza belief
        belief, belief_trace = solver.update_belief(
            belief, action_idx, observation, return_trace=True
        )

        # Atualiza recompensa total
        total_reward += reward
        step += 1

        # Visualiza
        visualizer.render(
            agent_pos=next_state,
            belief=belief,
            step=step,
            action=action_name,
            observation=observation,
            reward=reward,
            total_reward=total_reward,
            done=done,
            q_values=q_values,
            belief_trace=belief_trace,
        )

        if visualizer.closed:
            print("Visualização encerrada pelo usuário.")
            break

        if save_video:
            frame = visualizer.get_frame()
            if frame is not None:
                padded = fit_to_1080p(frame)
                frames.extend([padded] * STEP_HOLD_FRAMES)

        state = next_state

    # Mensagem final
    print("\n" + "=" * 60)
    if done:
        print("SUCESSO! Objetivo alcançado!")
    else:
        print(f"Episódio terminou após {max_steps} passos")
    print(f"Reward total: {total_reward:.1f}")
    print("=" * 60 + "\n")

    # Salva vídeo
    if save_video and frames:
        print(f"Salvando vídeo em {video_path}...")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (width, height))

        for frame in frames:
            out.write(frame)

        # Adiciona frames extras no final para pausar
        for _ in range(int(TARGET_FPS * END_HOLD_SECONDS)):
            out.write(frames[-1])

        out.release()
        print(f"Vídeo salvo com sucesso! ({len(frames)} frames)")

    return total_reward, step, done


def main():
    """Função principal"""
    print("=" * 60)
    print("DEMONSTRAÇÃO DE POMDP - Grid World com Neblina")
    print("=" * 60)

    # Parâmetros
    grid_size = 7  # Grid maior para a demonstração
    observation_noise = 0.7  # 70% de chance de observação incorreta (mais incerteza!)
    transition_noise = 0.2  # 20% de chance de escorregar para outro estado
    n_iterations = 100
    max_steps = 50

    print(f"\nParâmetros:")
    print(f"  Tamanho do grid: {grid_size}x{grid_size}")
    print(f"  Ruído de observação: {observation_noise*100:.0f}%")
    print(f"  Ruído de transição: {transition_noise*100:.0f}%")
    print(f"  Iterações de treinamento: {n_iterations}")

    # Cria ambiente
    print("\nCriando ambiente...")
    env = GridWorldPOMDP(
        size=grid_size,
        observation_noise=observation_noise,
        transition_noise=transition_noise,
    )

    print(f"  Estados: {env.n_states}")
    print(f"  Ações: {env.n_actions} {env.actions}")
    print(f"  Observações: {env.n_observations}")
    print(f"  Início: {env.start_pos}")
    print(f"  Objetivo: {env.goal_pos}")
    print(f"  Obstáculos: {env.obstacles}")

    # Cria solver
    print("\nCriando solver POMDP...")
    solver = POMDPSolver(env, gamma=0.95)

    # Treina solver
    print("\nTreinando solver...")
    solver.solve(n_iterations=n_iterations, sample_beliefs=100)

    # Cria visualizador
    print("\nCriando visualizador...")
    visualizer = POMDPVisualizer(env, cell_size=100)

    # Executa episódio
    print("\nExecutando episódio...")
    video_path = os.path.join(os.path.dirname(__file__), "pomdp_demonstration.mp4")

    total_reward, steps, success = run_episode(
        env,
        solver,
        visualizer,
        max_steps=max_steps,
        save_video=True,
        video_path=video_path,
    )

    # Mantém janela aberta por alguns segundos
    print("\nMantendo visualização por 5 segundos...")
    import time

    time.sleep(5)

    # Fecha visualizador
    visualizer.close()

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DA DEMONSTRAÇÃO")
    print("=" * 60)
    print(f"Total de passos: {steps}")
    print(f"Reward total: {total_reward:.1f}")
    print(f"Sucesso: {'Sim' if success else 'Não'}")
    print(f"Vídeo salvo em: {video_path}")
    print("\nDemonstração concluída!")
    print("=" * 60)


if __name__ == "__main__":
    main()
