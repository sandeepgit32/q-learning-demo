import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time # Added import

# Page Config
st.set_page_config(page_title="Q-Learning Demo", layout="wide")
st.title("🤖 Q-Learning Demo")
st.write("---")

# --- Default Hyperparameter Values ---
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_EPSILON = 0.1

# --- Fixed Environment Parameters ---
grid_size = 3
goal_x, goal_y = 2, 2  # goal position
actions_map = {0: "⬆️", 1: "⬇️", 2: "⬅️", 3: "➡️"}

# --- Session State Init ---
if "Q" not in st.session_state:
    st.session_state.Q = np.zeros((grid_size, grid_size, 4), dtype=float) # Explicitly float
if "state" not in st.session_state:
    st.session_state.state = (0, 0)
if "done" not in st.session_state:
    st.session_state.done = False
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "last_reward" not in st.session_state:
    st.session_state.last_reward = None
if "updated_cell" not in st.session_state:
    st.session_state.updated_cell = None
if "old_q_value" not in st.session_state:
    st.session_state.old_q_value = None
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "path_history" not in st.session_state:
    st.session_state.path_history = [(0,0)]
if "autoplay_active" not in st.session_state:
    st.session_state.autoplay_active = False
if "autoplay_interval" not in st.session_state:
    st.session_state.autoplay_interval = 1.5
if "autoplay_was_active_when_goal_reached" not in st.session_state:
    st.session_state.autoplay_was_active_when_goal_reached = False

if "alpha" not in st.session_state:
    st.session_state.alpha = DEFAULT_ALPHA
if "gamma" not in st.session_state:
    st.session_state.gamma = DEFAULT_GAMMA
if "epsilon" not in st.session_state:
    st.session_state.epsilon = DEFAULT_EPSILON
if "hyperparams_just_changed_warning" not in st.session_state:
    st.session_state.hyperparams_just_changed_warning = False

if "reward_table" not in st.session_state:
    st.session_state.reward_table = np.full((grid_size, grid_size), -1.0, dtype=float)
    st.session_state.reward_table[goal_x, goal_y] = 10.0
if "previous_reward_table" not in st.session_state:
    if "reward_table" in st.session_state:
        st.session_state.previous_reward_table = st.session_state.reward_table.copy()
    else:
        temp_reward_table = np.full((grid_size, grid_size), -1.0, dtype=float)
        temp_reward_table[goal_x, goal_y] = 10.0
        st.session_state.previous_reward_table = temp_reward_table.copy()

if "rewards_just_changed_warning" not in st.session_state:
    st.session_state.rewards_just_changed_warning = False

# Page navigation state
page_options = ["Simulation", "Reward Table"]
if "current_page" not in st.session_state:
    st.session_state.current_page = page_options[0]


# --- Helper function to reset simulation state (core parts) ---
def reset_simulation_core_state(reset_q_table: bool = True): # Modified signature
    if reset_q_table:
        st.session_state.Q = np.zeros((grid_size, grid_size, 4), dtype=float) # Explicitly float
    st.session_state.state = (0, 0)
    st.session_state.done = False
    st.session_state.last_action = None
    st.session_state.last_reward = None
    st.session_state.updated_cell = None
    st.session_state.old_q_value = None
    st.session_state.step_count = 0
    st.session_state.path_history = [(0,0)]
    st.session_state.autoplay_active = False 
    st.session_state.autoplay_was_active_when_goal_reached = False

# --- Q-Learning Step Function ---
def step_q_learning():
    st.session_state.step_count += 1
    state = st.session_state.state
    Q = st.session_state.Q

    if state == (goal_x, goal_y):
        st.session_state.done = True
        return

    if np.random.rand() < st.session_state.epsilon:
        action = np.random.randint(4)
    else:
        max_val = np.max(Q[state[0], state[1], :])
        max_actions = np.where(Q[state[0], state[1], :] == max_val)[0]
        action = np.random.choice(max_actions)

    old_q_value = Q[state[0], state[1], action]

    next_state = (
        max(min(state[0] + (action == 1) - (action == 0), grid_size - 1), 0),
        max(min(state[1] + (action == 3) - (action == 2), grid_size - 1), 0),
    )

    reward = st.session_state.reward_table[next_state[0], next_state[1]]

    best_next = np.max(Q[next_state[0], next_state[1], :])
    Q[state[0], state[1], action] += st.session_state.alpha * (
        reward + st.session_state.gamma * best_next - Q[state[0], state[1], action]
    )

    st.session_state.last_action = action
    st.session_state.last_reward = reward
    st.session_state.updated_cell = (state[0], state[1], action)
    st.session_state.old_q_value = old_q_value
    st.session_state.state = next_state
    st.session_state.Q = Q
    st.session_state.path_history.append(next_state)
    if next_state == (goal_x, goal_y):
        st.session_state.done = True


# --- Sidebar ---
st.sidebar.header("Q-Learning")
st.sidebar.markdown("""
**Environment:** 3×3 Grid, start at (0,0), goal at (2,2)

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α [r + γ maxₐ' Q(s',a') - Q(s,a)]
```
""")
st.sidebar.markdown("#### 🧭 Navigation") # Added line
# The radio button directly updates st.session_state.current_page due to the 'key' argument.
st.sidebar.radio(
    "Go to",
    page_options,
    key="current_page", # This links the widget to st.session_state.current_page 
    horizontal=True,
    index=page_options.index(st.session_state.get("current_page", page_options[0])) # Ensures correct initial selection
)

st.sidebar.markdown("---") # Added line
st.sidebar.markdown("**Q-Learning Hyperparameters:**")
initial_alpha_for_check = st.session_state.alpha
initial_gamma_for_check = st.session_state.gamma
initial_epsilon_for_check = st.session_state.epsilon

st.sidebar.slider("Alpha (α) - Learning Rate", 0.0, 1.0, step=0.01, key="alpha", help="The learning rate (0 < α ≤ 1) determines how much new information overrides old information. A high value will make the agent learn faster but might make it unstable.")
st.sidebar.slider("Gamma (γ) - Discount Factor", 0.0, 1.0, step=0.01, key="gamma", help="The discount factor (0 ≤ γ ≤ 1) determines the importance of future rewards. A value of 0 makes the agent short-sighted, while a value closer to 1 makes it strive for long-term high rewards.")
st.sidebar.slider("Epsilon (ε) - Exploration Rate", 0.0, 1.0, step=0.01, key="epsilon", help="The exploration rate (0 ≤ ε ≤ 1) determines the probability of choosing a random action instead of the best-known action. It encourages exploration of the environment.")
st.sidebar.caption("Changing these values will reset the simulation.")

if (st.session_state.alpha != initial_alpha_for_check or
    st.session_state.gamma != initial_gamma_for_check or
    st.session_state.epsilon != initial_epsilon_for_check):
    reset_simulation_core_state(reset_q_table=True) # Ensure hard reset
    st.session_state.hyperparams_just_changed_warning = True
    st.rerun()

if st.session_state.pop("hyperparams_just_changed_warning", False):
    st.sidebar.warning("Hyperparameters changed. Simulation has been reset.")

# This warning will be shown if the reward table was changed (logic below, after page content)
if st.session_state.pop("rewards_just_changed_warning", False):
    st.sidebar.warning("Reward table modified. Simulation has been reset.")


# --- Page Content ---

if st.session_state.current_page == "Reward Table":
    st.subheader("Edit Reward Table") # Modified line
    st.caption(f"Modify rewards for states in the {grid_size}x{grid_size} grid. Goal is at ({goal_x}, {goal_y}).")
    st.markdown("Changes made here will reset the Q-Learning simulation (Q-table, agent path, etc.).")

    # Create a DataFrame from the current reward_table for the editor
    df_for_editor = pd.DataFrame(
        st.session_state.reward_table.copy(), # Use a copy to avoid direct mutation issues before processing
        index=[f"Row {i}" for i in range(grid_size)],
        columns=[f"Col {j}" for j in range(grid_size)]
    )

    edited_df = st.data_editor(
        df_for_editor,
        key="reward_editor_widget", # Unique key for the data_editor widget itself
        use_container_width=True
    )

    # Convert the output of the data_editor (edited_df) back to numpy
    edited_rewards_numpy = edited_df.to_numpy(dtype=float, na_value=-1.0) # Handle empty cells

    # If the content from the editor is different from the current st.session_state.reward_table,
    # then update st.session_state.reward_table.
    if not np.array_equal(st.session_state.reward_table, edited_rewards_numpy):
        st.session_state.reward_table = edited_rewards_numpy
        # A rerun is needed here to trigger the global check below immediately
        # with the updated st.session_state.reward_table.
        st.rerun()

# --- Global Check for Reward Table Changes (runs after potential update from editor) ---
if "reward_table" in st.session_state and "previous_reward_table" in st.session_state:
    if not np.array_equal(st.session_state.reward_table, st.session_state.previous_reward_table):
        reset_simulation_core_state(reset_q_table=True) # Ensure hard reset
        st.session_state.previous_reward_table = st.session_state.reward_table.copy() 
        st.session_state.rewards_just_changed_warning = True
        st.rerun() # Rerun to reflect reset and show warning in sidebar


if st.session_state.current_page == "Simulation":
    # Visualization legend in sidebar for this page
    st.sidebar.markdown("""
**Visualization:**
- 🌸 **Light pink column**: Agent's current state (s)
- 🟡 **Yellow cell**: Q(s,a) value that was just updated
- 🟢 **Green 'Max' row**: Max Q-value for each state
""")

    st.subheader("Simulation") # Added line
    with st.container(border=False): # Added container
        col_next, col_auto_play_pause, col_reset_grid, col_hard_reset = st.columns(4)

        with col_next:
            next_step_disabled = st.session_state.get("autoplay_active", False) or st.session_state.get("done", False)
            if st.button("🚶 Take Next Step", disabled=next_step_disabled, use_container_width=True): # Added use_container_width
                if not st.session_state.get("done", False):
                    step_q_learning()
                    st.rerun()
        with col_auto_play_pause:
            if not st.session_state.get("autoplay_active", False):
                autoplay_disabled = st.session_state.get("done", False)
                if st.button("▶️ Autoplay", disabled=autoplay_disabled, use_container_width=True): # Added use_container_width
                    st.session_state.autoplay_active = True
                    st.session_state.autoplay_was_active_when_goal_reached = False
                    st.rerun()
            else:
                if st.button("⏸️ Pause", use_container_width=True): # Added use_container_width
                    st.session_state.autoplay_active = False
                    st.rerun()
        with col_reset_grid:
            if st.button("🔄 Reset Grid Only", use_container_width=True): # Added use_container_width
                current_interval = st.session_state.get("autoplay_interval", 1.5)
                reset_simulation_core_state(reset_q_table=False) # Grid Reset
                st.session_state.autoplay_interval = current_interval
                st.rerun()
        with col_hard_reset:
            if st.button("💥 Hard Reset", use_container_width=True): # Modified Button text for brevity, Added use_container_width
                current_interval = st.session_state.get("autoplay_interval", 1.5)
                reset_simulation_core_state(reset_q_table=True) # Hard Reset
                st.session_state.autoplay_interval = current_interval
                st.rerun()

        # Moved Goal reached messages outside the button columns
        if st.session_state.get("done", False):
            if st.session_state.get("autoplay_was_active_when_goal_reached", False):
                 st.warning("🎉 Goal reached during autoplay! Autoplay has been paused. Click Reset to start over.")
            else:
                st.success("🎉 Goal reached! Click Reset to start over.")

    state = st.session_state.state
    grid_col, table_col = st.columns([1, 2])
    with grid_col:
        st.subheader("Grid Visualization")
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(grid_size + 1):
            ax.plot([0, grid_size], [i, i], color="gray", linewidth=2)
            ax.plot([i, i], [0, grid_size], color="gray", linewidth=2)
        ax.scatter(goal_y + 0.5, grid_size - goal_x - 0.5, marker="*", s=300, color="gold", edgecolor="darkgoldenrod", linewidth=2)
        ax.scatter(state[1] + 0.5, grid_size - state[0] - 0.5, marker="o", s=200, facecolor="mediumpurple", edgecolor="darkviolet", linewidth=2)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if len(st.session_state.path_history) > 1:
            path_x = [p[1] + 0.5 for p in st.session_state.path_history]
            path_y = [grid_size - p[0] - 0.5 for p in st.session_state.path_history]
            ax.plot(path_x, path_y, color="cyan", linewidth=1.5, alpha=0.6, linestyle='--')
        for r_coord in range(grid_size):
            for c_coord in range(grid_size):
                ax.text(c_coord + 0.8, grid_size - r_coord - 0.8, f"({r_coord},{c_coord})", ha="right", va="top", fontsize=8, color="gray", alpha=0.7)
        st.pyplot(fig)

    with table_col:
        st.subheader("Q-Table Values")
        df_data = {}
        for i in range(grid_size):
            for j in range(grid_size):
                df_data[f"({i},{j})"] = [f"{st.session_state.Q[i, j, k]:.3f}" for k in range(4)]
        df = pd.DataFrame(df_data, index=[actions_map[i] for i in range(len(actions_map))])
        max_values = st.session_state.Q.max(axis=2)
        max_row_data = {}
        for i in range(grid_size):
            for j in range(grid_size):
                current_max = -np.inf
                for action_idx in range(4):
                    current_max = max(current_max, st.session_state.Q[i,j,action_idx])
                max_row_data[f"({i},{j})"] = f"{current_max:.3f}"
        df.loc["Max"] = max_row_data

        def combined_styler(data_cell, row_name, col_name):
            style = ''
            current_r_state, current_c_state = st.session_state.state
            current_state_col_name = f"({current_r_state},{current_c_state})"
            if col_name == current_state_col_name:
                style = 'background-color: mistyrose;'
            if row_name == "Max":
                style = 'background-color: lightgreen;'
            if st.session_state.updated_cell is not None:
                updated_i, updated_j, updated_action_idx = st.session_state.updated_cell
                if col_name == f"({updated_i},{updated_j})":
                    action_name_for_updated_cell = actions_map[updated_action_idx]
                    if row_name == action_name_for_updated_cell:
                        style = 'background-color: yellow;'
            return style
        
        try:
            styled_df = df.style.apply(lambda s: [combined_styler(s.loc[idx], idx, s.name) for idx in s.index], axis=0)
            st.dataframe(styled_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error applying style: {e}")
            st.dataframe(df, use_container_width=True)

        st.subheader(f"Current Step Info (step = {st.session_state.step_count})")
        st.write(f"- **Current State (s):** ({str(state[0])}, {str(state[1])})")
        action = st.session_state.last_action
        if action is not None:
            st.write(f"- **Action Taken (a):** {actions_map[action]}")
            if (st.session_state.updated_cell is not None and 
                st.session_state.old_q_value is not None):
                i, j, a_idx = st.session_state.updated_cell
                old_val = st.session_state.old_q_value
                new_val = st.session_state.Q[i, j, a_idx]
                st.write(f"- **Q-Value Update:** Q(({i},{j}), {actions_map[a_idx]}) = {old_val:.3f} → {new_val:.3f}")
        st.write(f"- **Reward Received (r):** {st.session_state.last_reward}")


# --- Autoplay Execution Logic ---
if (st.session_state.current_page == "Simulation" and # Corrected page name
    st.session_state.get("autoplay_active", False)):
    if not st.session_state.get("done", False):
        step_q_learning()
        if st.session_state.done:
            st.session_state.autoplay_was_active_when_goal_reached = True
            st.session_state.autoplay_active = False
            st.rerun()
        else: # Corrected: Added colon
            time.sleep(st.session_state.autoplay_interval)
            st.rerun()
    elif st.session_state.get("done", False): 
        st.session_state.autoplay_active = False
        st.rerun()
