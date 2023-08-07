from Controller import Controller, Command, Constants
from typing import Union, Optional
import time


class TickTackToe:
    def __init__(self):
        self._controller = Controller()
        self._controller.goto_home_position()
        self._controller.process_command(Command(code=Constants.GRIPPER_MOVE, grasp=False))
        self._board: list[[bool | None]] = [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]
        self._current_player: Union[Constants.WHITE, Constants.BLACK] = Constants.WHITE
        self._winner: Optional[Union[Constants.WHITE, Constants.BLACK]] = None
        self._game_over: bool = False
        self._turn: int = 0

    def __repr__(self):
        # create a string representation of the board with a | and - as separators between cells
        board = ""
        for row in self._board:
            for cell in row:
                board += f"{Constants.WHITE_BLACK[cell] if cell is not None else ' '}|"
            board = board[:-1] + "\n"
            board += "-" * 5 + "\n"
        return board

    def reset(self):
        self._clear_board()
        self._winner = None
        self._game_over = False
        self._current_player = Constants.WHITE

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
            Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.PRE_PICK_UP))
        self._controller.process_command(
            Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.WHITE_BLACK_PICK_UP[self._current_player]))
        self._controller.process_command(
            Command(code=Constants.PICK_UP, z_offset=Constants.PICK_UP_Z + (Constants.BLOCK_HEIGHT * (self._turn % 2))))

        self._controller.process_command(
            Command(code=Constants.PARTS_MOVE, final_pos=Constants.BOARD[x][y][0]))
        self._controller.process_command(
            Command(code=Constants.PLACE_DOWN, z_offset=Constants.BOARD[x][y][1]))

        self._controller.goto_home_position()

    def _clear_board(self):
        for x in range(3):
            for y in range(3):
                if self._board[x][y] is not None:
                    self._controller.process_command(
                        Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.BOARD[x][y][0]))
                    self._controller.process_command(
                        Command(code=Constants.PICK_UP, z_offset=Constants.BOARD[x][y][1]))
                    self._controller.process_command(
                        Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.WHITE_BLACK_PICK_UP[self._board[x][y]]))
                    self._controller.process_command(
                        Command(code=Constants.PLACE_DOWN,
                                z_offset=Constants.PICK_UP_Z + (Constants.BLOCK_HEIGHT * (self._turn % 2))))
                    self._turn -= 1
                    self._board[x][y] = None
        self._controller.goto_home_position()

    def _play(self, x, y) -> bool:
        if self._game_over:
            raise Exception("Game is over")
        if self._board[x][y] is not None:
            return False
        self._board[x][y] = self._current_player
        self._make_move(x, y)
        if self._check_game_over():
            self._game_over = True
        self._current_player = Constants.WHITE if self._current_player == Constants.BLACK else Constants.BLACK
        self._turn += 1
        return True

    def main_loop(self):
        while True:
            while not self._game_over:
                print(self)
                x, y = input(f"Player {self._current_player}, enter x and y: ").split()
                x, y = int(x), int(y)
                if not self._play(x, y):
                    print("Invalid move!")
            print(self)
            print(f"Game over! Winner is {self._winner}")
            play_again = input("Play again? (y/n) ") == "y"
            self.reset()
            if not play_again:
                break
        self._controller.shutdown()

    def demo_main_loop(self):
        white_moves = [
            (Constants.A, 1),
            (Constants.B, 1),
            (Constants.C, 1),
        ]
        black_moves = [
            (Constants.A, 2),
            (Constants.B, 0),
        ]
        while len(white_moves) > 0 or len(black_moves) > 0:
            print(self)
            if self._current_player == Constants.WHITE:
                x, y = white_moves.pop(0)
            else:
                x, y = black_moves.pop(0)
            if not self._play(x, y):
                print("Invalid move!")
            time.sleep(1)
        print(self)
        print(f"Game over! Winner is {self._winner}")
        self.reset()
        self._controller.shutdown()


def main():
    TickTackToe().main_loop()


if __name__ == '__main__':
    main()
