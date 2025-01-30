from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

from calculate_metric import get_stats_for_state

# Before creating your figure:
sns.set_theme(style="dark", context="talk")  # or "darkgrid", "ticks", etc.

from config.mappo_config import MAPPOConfig
from .target_mpe_env import TargetMPEEnvironment, MPEState

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MPEVisualizer(object):
    def __init__(
            self,
            env: TargetMPEEnvironment,
            state_seq: MPEState,
            config: MAPPOConfig,
            reward_seq=None,
    ):
        self.ax = None
        self.fig = None
        self.env = env

        self.interval = 200
        self.state_seq = state_seq
        self.reward_seq = reward_seq

        self.entity_artists = []
        self.visibility_circle = []
        self.step_counter = None
        self.collision_counter_text = None
        self.death_counter_text = None
        self.collision_counter = 0

        self.config = config

        self.labels = []

        self.init_render()

    def animate(
            self,
            save_filename: Optional[str] = None,
            view: bool = True,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.state_seq.step.shape[0],
            blit=False,
            interval=self.interval,
            cache_frame_data=False,
        )
        # Save the animation to a gif
        if save_filename is not None:
            ani.save(save_filename)

        if view:
            plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle

        env_index = 0
        state = self.state_seq._replace(
            entity_positions=self.state_seq.entity_positions[0][env_index],
            step=self.state_seq.step[0][env_index],
            agent_visibility_radius=self.state_seq.agent_visibility_radius[0][
                env_index
            ],
        )
        self.collision_counter = 0

        # state = self.state_seq[0]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

        sns.despine(ax=self.ax)

        ax_lim = self.config.env_config.env_kwargs.entities_initial_coord_radius[0] + 2

        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])

        palette = sns.color_palette("tab10", n_colors=self.env.num_entities)

        fig_size_inches = self.fig.get_size_inches()
        dpi = self.fig.dpi
        width_pixels = fig_size_inches[0] * dpi
        fontsize = width_pixels / (75 * ax_lim)
        ordered_color = []
        for i in range(self.env.num_agents):
            color = palette[i % len(palette)]
            circle = Circle(
                state.entity_positions[i],  # type: ignore
                self.env.entity_radius[i],
                color=color,
                ec=color,
                lw=1.0,
                zorder=2,
            )
            ordered_color.append(color)
            self.ax.add_patch(circle)
            self.entity_artists.append(circle)

            visibility_circle = Circle(
                state.entity_positions[i],  # type: ignore
                state.agent_visibility_radius[i],  # type: ignore
                color=color,
                ec="lightgray",
                alpha=0.05,
                linestyle=":",
                lw=1.0,
                zorder=3,
            )
            self.ax.add_patch(visibility_circle)
            self.visibility_circle.append(visibility_circle)

            label = f"A {i}"

            circle_text = self.ax.text(
                state.entity_positions[i][0].item(),
                state.entity_positions[i][1].item(),
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
                zorder=2,
            )

            self.labels.append(circle_text)

        for i in range(self.env.num_agents, self.env.num_entities):
            circle = Circle(
                state.entity_positions[i],  # type: ignore
                self.env.entity_radius[i],
                color=ordered_color[i - self.env.num_agents],
                ec=ordered_color[i - self.env.num_agents],
                lw=1.0,
                zorder=1,
            )
            self.ax.add_patch(circle)
            self.entity_artists.append(circle)

            label = f"T {i - self.env.num_agents}"

            circle_text = self.ax.text(
                state.entity_positions[i][0].item(),
                state.entity_positions[i][1].item(),
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
                zorder=1,
            )

            self.labels.append(circle_text)
        fontsize = 10
        text_spacing = ax_lim / 10

        self.step_counter = self.ax.text(
            -(ax_lim - 0.05),
            (ax_lim - 0.05),
            f"Step: {state.step}",
            va="top",
            fontsize=fontsize,
        )

        num_collisions, num_agent_died = get_stats_for_state(self.env, state)
        self.collision_counter += num_collisions
        self.collision_counter_text = self.ax.text(
            -(ax_lim - 0.05),
            (ax_lim - text_spacing - 0.05),
            f"Collisions: {num_collisions}",
            va="top",
            fontsize=fontsize,
        )
        self.death_counter_text = self.ax.text(
            -(ax_lim - 0.05),
            (ax_lim - 2 * text_spacing - 0.05),
            f"Num deaths: {num_agent_died}",
            va="top",
            fontsize=fontsize,
        )

    def update(self, frame):
        env_index = 0
        state = self.state_seq._replace(
            entity_positions=self.state_seq.entity_positions[frame][env_index],
            step=self.state_seq.step[frame][env_index],
        )
        for i, c in enumerate(self.entity_artists):
            c.center = state.entity_positions[i]
        for i, c in enumerate(self.visibility_circle):
            c.center = state.entity_positions[i]
        for i, l in enumerate(self.labels):
            l.set_position(state.entity_positions[i])

        if frame == 0:
            self.collision_counter = 0

        num_collisions, num_agent_died = get_stats_for_state(self.env, state)
        self.collision_counter += num_collisions

        self.step_counter.set_text(f"Step: {state.step}")
        self.collision_counter_text.set_text(f"Collisions: {self.collision_counter}")
        self.death_counter_text.set_text(f"Num deaths: {num_agent_died}")

        return self.entity_artists + [self.step_counter]
