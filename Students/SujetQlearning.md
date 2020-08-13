# AI Pool 2021 - Deep Learning - Value Function

TODO: Ajouter l'histoire + l'énoncé au sujet
PRECISER LES ACTION POSSIBLE AVEC BLOBENV()

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Write you code in Submit.py, you will see some usefull code allready implemented.<br>
When finished, submit your code to the work plateform.

Here is few things you need to know about the BlobEnv:

*   To do an action: ``BlobEnv().doAction(action: int) -> obs: [(int, int), (int, int)], new_obs: [(int, int), (int, int)], action_reward: int``
*   To show the current stat of the env: ``BlobEnv().show()``
*   To get the obs: ``BlobEnv().getObs() -> obs: [(int, int), (int, int)]``
*   To reset the env: ``BlobEnv().reset()``
*   What is ``obs``: It's the observation environement before your action
*   What is ``new_obs``: It's the observation environement after your action
*   What is action_reward: It's the reward you got from the action

Userfull links:<br>
*   What is ``pickle``: https://docs.python.org/3/library/pickle.html
*   what is ``matplot``: https://matplotlib.org/
*   What is ``Q Learning``: https://youtu.be/qhRNvCVVJaA
