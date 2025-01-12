from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Before creating your figure:
sns.set_theme(style="dark", context="talk")  # or "darkgrid", "ticks", etc.


from config.mappo_config import MAPPOConfig
from .target_mpe_env import TargetMPEEnvironment, MPEState

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MPEVisualizer(object):
    def __init__(
        self,
        env: TargetMPEEnvironment,
        state_seq: list[MPEState],
        config: MAPPOConfig,
        reward_seq=None,
    ):
        self.ax = None
        self.fig = None
        self.env = env

        self.interval = 200
        self.state_seq = state_seq
        self.reward_seq = reward_seq

        self.comm_active = not np.all(self.env.is_agent_silent)
        self.entity_artists = []
        self.visibility_circle = []
        self.step_counter = None

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
            frames=len(self.state_seq),
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_filename is not None:
            ani.save(save_filename)

        if view:
            plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle

        state = self.state_seq[0]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

        sns.despine(ax=self.ax)

        ax_lim = self.config.env_config.EnvKwArgs.entities_initial_coord_radius + 1

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
                self.env.neighborhood_radius[i],
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

        self.step_counter = self.ax.text(
            -(ax_lim - 0.05), (ax_lim - 0.05), f"Step: {state.step}", va="top"
        )

    def update(self, frame):
        state = self.state_seq[frame]
        for i, c in enumerate(self.entity_artists):
            c.center = state.entity_positions[i]
        for i, c in enumerate(self.visibility_circle):
            c.center = state.entity_positions[i]
        for i, l in enumerate(self.labels):
            l.set_position(state.entity_positions[i])
        self.step_counter.set_text(f"Step: {state.step}")

        return self.entity_artists + [self.step_counter]
