from Controller import Controller, Command, PysicalConstants
from typing import Union, Optional


def get_best_move(board):
    def is_winner(board, current_player):
        # Check if the current player has won
        for row in board:
            if all(cell == current_player for cell in row):
                return True

        for col in range(3):
            if all(board[row][col] == current_player for row in range(3)):
                return True

        if all(board[i][i] == current_player for i in range(3)) or all(
                board[i][2 - i] == current_player for i in range(3)):
            return True

        return False

    def is_draw(board):
        # Check if the game is a draw
        return all(cell is not None for row in board for cell in row)

    def evaluate(board, player):
        # Evaluate the current state of the board
        opponent = 1 if player == 0 else 0
        if is_winner(board, player):
            return 1
        elif is_winner(board, opponent):
            return -1
        elif is_draw(board):
            return 0
        return None

    def minimax(board, depth, maximizing_player, player):
        result = evaluate(board, player)

        if result is not None:
            return result

        if maximizing_player:
            max_eval = -float("inf")
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = player
                        eval = minimax(board, depth + 1, False, player)
                        board[i][j] = None
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = 1 if player == 0 else 0
                        eval = minimax(board, depth + 1, True, player)
                        board[i][j] = None
                        min_eval = min(min_eval, eval)
            return min_eval

    current_player = 0 if sum(row.count(0) for row in board) <= sum(row.count(1) for row in board) else 1
    best_move = None
    best_eval = -float("inf") if current_player == 0 else float("inf")

    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                board[i][j] = current_player
                eval = minimax(board, 0, False, current_player)
                board[i][j] = None

                if current_player == 0 and eval > best_eval:
                    best_eval = eval
                    best_move = (i, j)
                elif current_player == 1 and eval < best_eval:
                    best_eval = eval
                    best_move = (i, j)

    return best_move


class TickTackToe:
    def __init__(self, *, solo_play=False, start=None):
        self._controller = Controller()
        self._controller.goto_home_position()
        self._controller.process_command(Command(code=PysicalConstants.GRIPPER_MOVE, grasp=False))
        self._board: list[[bool | None]] = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        self._current_player: Union[PysicalConstants.WHITE, PysicalConstants.BLACK] = PysicalConstants.WHITE
        self._winner: Optional[Union[PysicalConstants.WHITE, PysicalConstants.BLACK]] = None
        self._game_over: bool = False
        self._turn: int = 0
        self._solo_play = solo_play
        if solo_play:
            if start is None:
                raise Exception("Who will start is not defined")
            if start:
                self._play(*get_best_move(self._board))

    def __repr__(self):
        # create a string representation of the board with a | and - as separators between cells
        board = ""
        for row in self._board:
            for cell in row:
                board += f"{PysicalConstants.WHITE_BLACK[cell] if cell is not None else ' '}|"
            board = board[:-1] + "\n"
            board += "-" * 5 + "\n"
        return board

    def reset(self):
        self._clear_board()
        self._winner = None
        self._game_over = False
        self._current_player = PysicalConstants.WHITE

    def _check_winner(self):
        for i in range(3):
            if self._board[i][0] == self._board[i][1] == self._board[i][2] is not None:
                self._winner = self._board[i][0]
                return True
            if self._board[0][i] == self._board[1][i] == self._board[2][i] is not None:
                self._winner = self._board[0][i]
                return True
        if self._board[0][0] == self._board[1][1] == self._board[2][2] is not None:
            self._winner = self._board[0][0]
            return True
        if self._board[0][2] == self._board[1][1] == self._board[2][0] is not None:
            self._winner = self._board[0][2]
            return True
        return False

    def _check_draw(self):
        for row in self._board:
            for cell in row:
                if cell is None:
                    return False
        return True

    def _check_game_over(self):
        return self._check_winner() or self._check_draw()

    def _make_move(self, x, y):
        self._controller.process_command(
            Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))
        self._controller.process_command(
            Command(code=PysicalConstants.SIMPLE_MOVE,
                    final_pos=PysicalConstants.WHITE_BLACK_PICK_UP[self._current_player]))
        self._controller.process_command(
            Command(code=PysicalConstants.PICK_UP,
                    z_offset=PysicalConstants.PICK_UP_Z + (PysicalConstants.BLOCK_HEIGHT * (self._turn // 2))))

        self._controller.process_command(
            Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))

        self._controller.process_command(
            Command(code=PysicalConstants.PARTS_MOVE, final_pos=PysicalConstants.BOARD[x][y][0]))
        self._controller.process_command(
            Command(code=PysicalConstants.PLACE_DOWN, z_offset=PysicalConstants.BOARD[x][y][1]))
        self._controller.goto_home_position()

    def _clear_board(self):
        white_pick_up, black_pick_up = [], []
        for x in range(3):
            for y in range(3):
                if self._board[x][y] is not None:
                    if self._board[x][y]:
                        black_pick_up.append((x, y))
                    else:
                        white_pick_up.append((x, y))
                    self._board[x][y] = None
        self._turn = 0

        for color, pos_arr in enumerate([white_pick_up, black_pick_up]):
            while len(pos_arr) != 0:
                x, y = pos_arr.pop()
                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.BOARD[x][y][0]))
                self._controller.process_command(
                    Command(code=PysicalConstants.PICK_UP, z_offset=PysicalConstants.BOARD[x][y][1]))

                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))

                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.WHITE_BLACK_PICK_UP[color]))
                self._controller.process_command(
                    Command(code=PysicalConstants.PLACE_DOWN,
                            z_offset=PysicalConstants.PICK_UP_Z + (
                                        PysicalConstants.BLOCK_HEIGHT * len(pos_arr)) + 0.004))

        self._controller.goto_home_position()

    def _play(self, x, y) -> bool:
        if self._game_over:
            raise Exception("Game is over")
        if self._board[x][y] is not None:
            return False
        self._board[x][y] = self._current_player
        print(self)
        self._make_move(x, y)
        if self._check_game_over():
            self._game_over = True
        self._current_player = PysicalConstants.WHITE if self._current_player == PysicalConstants.BLACK else PysicalConstants.BLACK
        self._turn += 1
        return True

    def main_loop(self):
        commands = [
            "a2",
            "a3",
            "b2",
            "b1",
            "c2",
        ]
        commands = [
            "a1",
            "a2",
            "a3",
            "b1",
            "b2",
            "b3",
            "c2",
            "c1",
            "c3"
        ]
        commands = [
            "a1",
            "a2",
            "a3",
            "b1",
            "b2",
            "b3",
        ]

        for cmd in commands:
            print(cmd, self.command(cmd))
            # time.sleep(1)
        self.turn_off()

    # -1 = invalid move
    # 0 = all good
    # 1 = game over
    # 2 = invalid command
    def command(self, cmd: str) -> int:
        # print(self)
        if cmd[0].lower() in ["a", "b", "c"]:
            x = ord(cmd[0].lower()) - ord("a")
            y = int(cmd[1]) - 1
            if not self._play(x, y):
                return -1
            if self._game_over:
                return 1
            self._play(*get_best_move(self._board))
            # print(self)
            return 0

        else:
            match cmd.lower():
                case "stop":
                    self._controller.paused = True
                    return 0
                case _:
                    return 2

    def turn_off(self):
        self.reset()
        self._controller.shutdown()

    def self_play(self):
        input()
        print(self.command("A2"))
        input()
        print(self.command("C1"))
        self._clear_board()
        self._controller.shutdown()


def main():
    ttt = TickTackToe(solo_play=True, start=False)
    ttt.self_play()


if __name__ == '__main__':
    main()
