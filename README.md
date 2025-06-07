# Q-Learning Demo

This is a Streamlit application that provides a step-by-step visualization of the Q-Learning algorithm.
Users can interact with the simulation, adjust hyperparameters, and observe how the agent learns to navigate a grid environment to reach a goal.

## Features

-   Interactive 3x3 grid environment.
-   Visualization of the agent's path and the Q-table.
-   Adjustable hyperparameters: Alpha (learning rate), Gamma (discount factor), Epsilon (exploration rate).
-   Step-by-step execution or autoplay mode.
-   Ability to edit the reward table for each state.
-   Clear display of the Q-learning update rule and current step information.

## Setup and Installation

1.  **Clone the repository (if applicable) or download the files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Once the dependencies are installed, you can run the Streamlit application using the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Files

-   `app.py`: The main Streamlit application script for the Q-Learning demo.
-   `requirements.txt`: A list of Python packages required to run the application.
-   `README.md`: This file.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
