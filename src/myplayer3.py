"""
Alpha-Beta Go Agent (5x5) — standalone agent code only.

This file intentionally depends on the course framework at runtime
(`read.py`, `write.py`, and `host.py` exposing a `GO` class), but that
framework is **not** included in this repository. See the repository
README for how to run this agent with your local copy of the course
harness without committing any third-party code.

Author: Dylan Senarath
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple

import random

# Runtime-only imports from the course harness (not shipped in this repo)
# from read import readInput
# from write import writeOutput
# from host import GO

Coord = Tuple[int, int]


class AlphaBetaAgent:
    """
    Alpha-Beta agent for 5x5 Go with a heuristic move ordering.
    Only the agent logic is provided here; the host environment is external.
    """

    def __init__(self, search_depth: int = 3, board_size: int = 5, debug: bool = False) -> None:
        self.search_depth = search_depth
        self.size = board_size
        self.debug = debug

        self.piece_type: Optional[int] = None
        self.is_first: bool = True
        self.score_diff: int = 0
        self.aggressive_opponent: bool = False
        self.total_captures: int = 0
        self._checked_first_move: bool = False


    def get_input(self, go, piece_type: int) -> Coord | str:
        """
        Decide the next move for `piece_type` on the given `go` state.
        Returns (i, j) or "PASS".
        """
        self.piece_type = piece_type

        # Determine if we moved first by scanning for any existing stones.
        if not self._checked_first_move:
            self.is_first = True
            for i in range(self.size):
                for j in range(self.size):
                    if go.board[i][j] != 0:
                        self.is_first = False
                        break
            self._checked_first_move = True

        # Score diff (host scoring may count komi; mirror original behavior)
        if self.is_first:
            self.score_diff = go.score(self.piece_type) - go.score(3 - self.piece_type)
        else:
            self.score_diff = (go.score(self.piece_type) + 1) - go.score(3 - self.piece_type)

        winning = self.score_diff >= 0
        self.aggressive_opponent = self._is_opponent_aggressive(go)

        moves = self._find_valid_moves(go)
        if not moves:
            return "PASS"

        _, best = self._alpha_beta(
            go=go,
            depth=0,
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=True,
            winning=winning,
        )

        if best is None:
            return "PASS"

        # lightweight stats (noisy prints gated by debug)
        lib = self._liberties(go, best, self.piece_type)
        cap = self._capture_potential(go, best, 3 - self.piece_type)
        self.total_captures += cap
        if self.debug:
            print(f"liberties={lib} cap={cap} my={1 + go.score(self.piece_type)} opp={go.score(3 - self.piece_type)}")
            print(f"score_diff={self.score_diff} first={self.is_first} opp_aggr={self.aggressive_opponent}")
            print(f"total_captures={self.total_captures}")

        return best

    # -----------------------------
    # Alpha-Beta with heuristic ordering
    # -----------------------------
    def _alpha_beta(
        self,
        go,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        winning: bool,
    ) -> Tuple[float, Optional[Coord]]:
        moves = self._find_valid_moves(go)
        moves.sort(key=lambda m: self._heuristic(go, m, winning), reverse=True)

        if depth == self.search_depth or go.game_end(self.piece_type) or not moves:
            return go.score(self.piece_type) - go.score(3 - self.piece_type), None

        if maximizing:
            best_val = -float("inf")
            best_move: Optional[Coord] = None
            for mv in moves:
                g = deepcopy(go)
                g.place_chess(mv[0], mv[1], self.piece_type)
                captured = len(g.remove_died_pieces(3 - self.piece_type))

                val, _ = self._alpha_beta(g, depth + 1, alpha, beta, False, winning)
                val += 2 * captured

                if val > best_val:
                    best_val, best_move = val, mv
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best_val, best_move

        else:
            worst_val = float("inf")
            worst_move: Optional[Coord] = None
            opp = 3 - self.piece_type
            for mv in moves:
                g = deepcopy(go)
                g.place_chess(mv[0], mv[1], opp)
                captured = len(g.remove_died_pieces(self.piece_type))

                val, _ = self._alpha_beta(g, depth + 1, alpha, beta, True, winning)
                val -= 2 * captured

                if val < worst_val:
                    worst_val, worst_move = val, mv
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return worst_val, worst_move

    # -----------------------------
    # Heuristics
    # -----------------------------
    def _heuristic(self, go, mv: Coord, winning: bool) -> float:
        """Combine several local features around `mv` into a single score."""
        conn = self._connection(go, mv, self.piece_type)                      # 0..13
        defense = self._liberties(go, mv, self.piece_type)                    # 0..15
        offense = self._capture_potential(go, mv, 3 - self.piece_type)        # 0..24
        opp_weak = self._opponent_weakness(go, mv)                            # ~0..?
        control = self._board_control(go, mv)                                 # 0..~9
        center = self._centering(go, mv)                                      # <=0

        if winning:
            if self.aggressive_opponent:
                score = (
                    10 * offense + 15 * defense + 15 * control +
                    5 * opp_weak + 5 * conn + 3 * center
                )
            else:
                score = (
                    8 * offense + 12 * defense + 15 * control +
                    5 * conn + 3 * center
                )
        else:
            score = (
                12 * offense + 15 * defense + 10 * control +
                5 * conn + 5 * opp_weak + 3 * center
            )
        return float(score)

    def _connection(self, go, mv: Coord, who: int) -> int:
        checked = set()
        q: List[Coord] = [mv]
        count = 0
        while q:
            i, j = q.pop(0)
            if (i, j) in checked:
                continue
            checked.add((i, j))
            count += 1
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if go.board[ni][nj] == who and (ni, nj) not in checked:
                        q.append((ni, nj))
        return count

    def _liberties(self, go, mv: Coord, who: int) -> int:
        checked = set()
        q: List[Coord] = [mv]
        libs = set()
        while q:
            i, j = q.pop(0)
            if (i, j) in checked:
                continue
            checked.add((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if go.board[ni][nj] == 0:
                        libs.add((ni, nj))
                    elif go.board[ni][nj] == who and (ni, nj) not in checked:
                        q.append((ni, nj))
        return len(libs)

    def _board_control(self, go, mv: Coord) -> float:
        max_dist = 3
        seen = set()
        q: List[Tuple[Coord, int]] = [(mv, 0)]
        score = 0.0
        while q:
            (i, j), d = q.pop(0)
            if d > max_dist:
                continue
            score += 1.0 / (d + 1)
            seen.add((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size and (ni, nj) not in seen:
                    if go.board[ni][nj] == 0:
                        q.append(((ni, nj), d + 1))
        return score

    def _centering(self, go, mv: Coord) -> float:
        i, j = mv
        c = (self.size - 1) / 2
        return -((i - c) ** 2 + (j - c) ** 2)

    def _capture_potential(self, go, mv: Coord, who: int) -> int:
        i, j = mv
        cap = 0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and go.board[ni][nj] == who:
                if self._liberties(go, (ni, nj), who) == 1:
                    cap += self._connection(go, (ni, nj), who)
        return cap

    def _opponent_weakness(self, go, mv: Coord) -> float:
        i, j = mv
        opp = 3 - self.piece_type
        s = 0.0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size and go.board[ni][nj] == opp:
                libs = self._liberties(go, (ni, nj), opp)
                s += 2.0 / (1 + libs)
        return s

    # -----------------------------
    # Move generation & opponent profiling
    # -----------------------------
    def _find_valid_moves(self, go) -> List[Coord]:
        moves: List[Coord] = []
        for i in range(self.size):
            for j in range(self.size):
                if go.valid_place_check(i, j, self.piece_type, test_check=True):
                    moves.append((i, j))
        return moves

    def _is_opponent_aggressive(self, go) -> bool:
        """Count enemy stones in positions where we have capture potential."""
        aggr = 0
        for i in range(self.size):
            for j in range(self.size):
                if go.board[i][j] == (3 - self.piece_type):
                    if self._capture_potential(go, (i, j), self.piece_type) > 0:
                        aggr += 1
        return aggr > 3


 
 if __name__ == "__main__":
     N = 5
     piece_type, previous_board, board = readInput(N)
     go = GO(N)
     go.set_board(piece_type, previous_board, board)
     agent = AlphaBetaAgent(search_depth=3, board_size=N, debug=True)
     action = agent.get_input(go, piece_type)
     writeOutput(action)
