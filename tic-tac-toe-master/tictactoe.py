"""
Tic Tac Toe Player
"""

import math
import numpy
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = 0

    for row in board:
        for cell in row:
            count += 1 if cell != EMPTY else 0

    return O if bool(count % 2) else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(len(board)):
        # check rows
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != EMPTY:
            return board[i][0]
        # check columns
        elif board[0][i] == board[1][i] == board[2][i] and board[0][i] != EMPTY:
            return board[0][i]

    # check diagonal 1
    if board[0][0] == board[1][1] == board[2][2] and board[1][1] != EMPTY:
        return board[1][1]
    # check diagonal 2
    elif board[0][2] == board[1][1] == board[2][0] and board[1][1] != EMPTY:
        return board[1][1]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or len(actions(board)) == 0:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    person = winner(board)
    if person == X:
        return 1
    elif person == O:
        return -1

    return 0


def max_value(board, min=math.inf):
    if terminal(board):
        return (utility(board), None)

    max = -math.inf
    best_action = None

    for action in actions(board):
        value = min_value(result(board, action), max)[0]
        # Aplha-Beta Pruning
        if value > min:
            return (value, action)

        if value > max:
            max = value
            best_action = action

    return (max, best_action)


def min_value(board, max=-math.inf):
    if terminal(board):
        return (utility(board), None)

    min = math.inf
    best_action = None

    for action in actions(board):
        value = max_value(result(board, action), min)[0]
        # Aplha-Beta Pruning
        if value < max:
            return (value, action)

        if value < min:
            min = value
            best_action = action

    return (min, best_action)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    return max_value(board)[1] if player(board) == X else min_value(board)[1]
