import threading
import time

from Controller import Controller, Command, PysicalConstants
from typing import Union, Optional


def get_best_move(board):
    def is_winner(b, c_p):
        # Check if the current player has won
        for row in b:
            if all(cell == c_p for cell in row):
                return True

        for col in range(3):
            if all(b[row][col] == c_p for row in range(3)):
                return True

        if all(b[i][i] == c_p for i in range(3)) or all(
                b[i][2 - i] == c_p for i in range(3)):
            return True

        return False

    def is_draw(b):
        # Check if the game is a draw
        return all(cell is not None for row in b for cell in row)

    def evaluate(b, p):
        # Evaluate the current state of the board
        opponent = 1 if p == 0 else 0
        if is_winner(b, p):
            return 1
        elif is_winner(b, opponent):
            return -1
        elif is_draw(b):
            return 0
        return None

    def minimax(b, depth, maximizing_player, player):
        result = evaluate(b, player)

        if result is not None:
            return result

        if maximizing_player:
            max_eval = -float("inf")
            for x in range(3):
                for y in range(3):
                    if b[x][y] is None:
                        b[x][y] = player
                        evaluation = minimax(b, depth + 1, False, player)
                        b[x][y] = None
                        max_eval = max(max_eval, evaluation)
            return max_eval
        else:
            min_eval = float("inf")
            for x in range(3):
                for y in range(3):
                    if b[x][y] is None:
                        b[x][y] = 1 if player == 0 else 0
                        evaluation = minimax(b, depth + 1, True, player)
                        b[x][y] = None
                        min_eval = min(min_eval, evaluation)
            return min_eval

    current_player = 0 if sum(row.count(0) for row in board) <= sum(row.count(1) for row in board) else 1
    best_move = None
    best_eval = -float("inf") if current_player == 0 else float("inf")

    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                board[i][j] = current_player
                ev = minimax(board, 0, False, current_player)
                board[i][j] = None

                if current_player == 0 and ev > best_eval:
                    best_eval = ev
                    best_move = (i, j)
                elif current_player == 1 and ev < best_eval:
                    best_eval = ev
                    best_move = (i, j)

    return best_move


class TickTackToe:
    def __init__(self, *, solo_play=False, start=None):
        self._playing = False
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

    @property
    def game_over(self):
        return self._game_over

    @property
    def playing(self):
        return self._playing

    def _reset(self):
        self._clear_board()
        # self._winner = None
        # self._game_over = False
        # self._current_player = PysicalConstants.WHITE

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

        if (self._turn // 2) != 0:
            self._controller.process_command(
                Command(code=PysicalConstants.CARTESIAN_MOVE,
                        z_offset=-PysicalConstants.BLOCK_HEIGHT * (self._turn // 2)))

        self._controller.process_command(Command(code=PysicalConstants.PICK_UP, z_offset=PysicalConstants.PICK_UP_Z))

        self._controller.process_command(
            Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))

        self._controller.process_command(
            Command(code=PysicalConstants.PARTS_MOVE, final_pos=PysicalConstants.BOARD[x][y]["pos"]))
        self._controller.process_command(
            Command(code=PysicalConstants.PLACE_DOWN, z_offset=PysicalConstants.BOARD[x][y]["drop_z"]))
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
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.BOARD[x][y]["pos"]))
                self._controller.process_command(
                    Command(code=PysicalConstants.PICK_UP, z_offset=PysicalConstants.BOARD[x][y]["pick_up_z"]))

                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))

                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.WHITE_BLACK_PICK_UP[color]))

                if len(pos_arr) != 0:
                    self._controller.process_command(
                        Command(code=PysicalConstants.CARTESIAN_MOVE,
                                z_offset=-PysicalConstants.BLOCK_HEIGHT * len(pos_arr)))

                self._controller.process_command(
                    Command(code=PysicalConstants.PLACE_DOWN, z_offset=PysicalConstants.DROP_Z))

                self._controller.process_command(
                    Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))

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

    # -1 = invalid move
    # 0 = all good
    # 1 = game over
    # 2 = invalid command
    # 3 = playing
    def _command_thread(self, cmd: str) -> None:
        if self.game_over:
            # raise Exception("Game is over. Create a new instance to play again :)")
            print("Game is over. Create a new instance to play again :)")
            return

        if cmd[0].lower() in ["a", "b", "c"]:
            if self.playing:
                print("Already playing a move wait for it to finish :)")
                return
            self._playing = True
            x = ord(cmd[0].lower()) - ord("a")
            y = int(cmd[1]) - 1
            result = 0
            if not self._play(x, y):
                result = -1
            else:
                if self._game_over:
                    result = 1
                elif self._solo_play:
                    self._play(*get_best_move(self._board))
                    if self._game_over:
                        result = 1
            if result == 1:
                self._clean_up()
            self._playing = False
            print(f"Result for {cmd}: {result}")
        else:
            match cmd.lower():
                case "stop":
                    self._controller.paused = True
                    return
                case _:
                    return

    def command(self, cmd: str) -> None:
        threading.Thread(target=self._command_thread, args=(cmd,)).start()

    def _clean_up(self):
        self._reset()
        self._controller.shutdown()

    def turn_off(self):
        if self.game_over:
            print("Game is over. Create a new instance to play again :)")
            return
        while self._playing:
            time.sleep(0.1)
        self._clean_up()

    def demo_one_player(self):
        while not self.game_over:
            self.command(input("Enter move: "))
            while self.playing:
                time.sleep(0.1)
            # time.sleep(1)
        self.turn_off()

    def demo_two_players(self):
        move_list = ["A1", "A2", "A3", "B1", "B2", "B3", "C2", "C1", "C3"]
        for move in move_list:
            self.command(move)
            while self.playing:
                time.sleep(0.1)
            # time.sleep(1)
        self.turn_off()


def demo_two_players():
    ttt = TickTackToe()
    ttt.demo_two_players()


def demo_one_player():
    ttt = TickTackToe(solo_play=True, start=True)
    ttt.demo_one_player()


def main():
    demo_two_players()
    # demo_one_player()


if __name__ == '__main__':
    main()
