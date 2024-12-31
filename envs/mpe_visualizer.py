from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .mpe_env import TargetMPEEnvironment, MPEState

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MPEVisualizer(object):
    def __init__(
        self,
        env: TargetMPEEnvironment,
        state_seq: list[MPEState],
        reward_seq=None,
    ):
        self.ax = None
        self.fig = None
        self.env = env

        self.interval = 100
        self.state_seq = state_seq
        self.reward_seq = reward_seq

        self.comm_active = not np.all(self.env.is_agent_silent)
        print("Comm active? ", self.comm_active)
        self.entity_artists = []
        self.step_counter = None

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

        ax_lim = 2
        self.ax.set_xlim([-ax_lim, ax_lim])
        self.ax.set_ylim([-ax_lim, ax_lim])

        for i in range(self.env.num_entities):
            c = Circle(
                state.entity_positions[i],  # type: ignore
                self.env.entity_radius[i],
                color=np.array(self.env.color[i]) / 255,
            )
            self.ax.add_patch(c)
            self.entity_artists.append(c)

        self.step_counter = self.ax.text(-1.95, 1.95, f"Step: {state.step}", va="top")

    def update(self, frame):
        state = self.state_seq[frame]
        for i, c in enumerate(self.entity_artists):
            c.center = state.entity_positions[i]
        self.step_counter.set_text(f"Step: {state.step}")

        return self.entity_artists + [self.step_counter]
