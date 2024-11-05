### Gymnasium Taxi v3

### Install the dependencies

```sh
  pip install -r requirements.txt
```

### Models

- In this project, we have 3 approaches for the taxi problem of Gymnasium. We use the `Taxi-v3` environment, provided by Gymnasium.

- There are 3 algorithms used in our project
  - Q-learning:

    ```sh
      py main.py
    ```

  - Double Q-learning:

    ```sh
      py double_q_learning.py
    ```

  - SARSA:
    ```sh
      py sarsa.py
    ```

- All these 3 approaches are able to solve the taxi problem. The training process takes about 10s for 15_000 episodes