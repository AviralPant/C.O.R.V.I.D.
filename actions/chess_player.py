"""Recommend a chess move from the active window using vision and Stockfish."""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil

from core.lazyimport import LazyModule, installed
from core import gemini

_PYAUTOGUI = installed("pyautogui")
pyautogui = LazyModule("pyautogui")


def _capture_active_window() -> bytes:
    if not _PYAUTOGUI:
        raise RuntimeError("PyAutoGUI is not installed. Run: pip install pyautogui")

    try:
        import pygetwindow
    except ImportError as e:
        raise RuntimeError(
            "pygetwindow is not installed. Run: pip install pygetwindow"
        ) from e

    window = pygetwindow.getActiveWindow()
    if window is None or window.isMinimized:
        raise RuntimeError("No visible active window was found.")

    region = (int(window.left), int(window.top), int(window.width), int(window.height))
    if region[2] <= 0 or region[3] <= 0:
        raise RuntimeError("The active window has invalid dimensions.")

    try:
        image = pyautogui.screenshot(region=region).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not capture the active window: {e}") from e

    image.thumbnail((1600, 1200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fen_from_screen(image_bytes: bytes, side_to_move: str) -> tuple[str | None, str]:
    from google.genai import types

    side_instruction = (
        f"The user confirms {side_to_move} is to move."
        if side_to_move != "auto"
        else "Infer whose turn it is only if the interface clearly indicates it."
    )
    prompt = (
        "Inspect this active-window screenshot for a chessboard. Return only a "
        "JSON object with keys `fen`, `confidence`, and `notes`. `fen` must be "
        "a complete six-field FEN string, or null if the board or side to move "
        "cannot be identified reliably. Use standard FEN orientation "
        "(rank 8 first, file a first). Confidence must be a number from 0 to 1 "
        "reflecting board-piece and turn recognition, not your chess skill. "
        "Do not infer pieces hidden by overlays. "
        f"{side_instruction}"
    )
    response = gemini.call(
        [types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
        tier=gemini.SMART,
        timeout_ms=45_000,
    )
    if response is None or not response.text:
        raise RuntimeError("The configured vision model did not return a board reading.")

    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError("The vision model returned an unreadable board position.") from e
    if not isinstance(result, dict):
        raise RuntimeError("The vision model returned an invalid board description.")

    fen = result.get("fen")
    confidence = result.get("confidence", 0)
    notes = str(result.get("notes") or "").strip()
    if not isinstance(fen, str) or not fen.strip():
        return None, notes
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        return None, "Board-reading confidence was invalid."
    if confidence < 0.8:
        return None, f"Board-reading confidence was only {confidence:.0%}. {notes}".strip()
    return fen.strip(), notes


def _find_stockfish() -> str | None:
    configured = os.environ.get("STOCKFISH_EXECUTABLE", "").strip()
    candidates = [configured] if configured else []
    candidates.extend(("stockfish", "stockfish.exe"))
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    return None


def _analyse_position(fen: str, side_to_move: str = "auto") -> str:
    try:
        import chess
        import chess.engine
    except ImportError as e:
        raise RuntimeError("python-chess is missing. Run: pip install python-chess") from e

    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise RuntimeError(f"The detected board position was not valid FEN: {e}") from e
    if side_to_move in {"white", "black"}:
        board.turn = chess.WHITE if side_to_move == "white" else chess.BLACK
    if not board.is_valid():
        raise RuntimeError("The detected chess position is illegal or incomplete.")
    if board.is_game_over():
        return "The position is already game over; there is no next move to recommend."

    engine_path = _find_stockfish()
    if not engine_path:
        return (
            "Stockfish was not found. Install Stockfish, add it to PATH, or set "
            "the STOCKFISH_EXECUTABLE environment variable to its executable path."
        )

    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path, timeout=10.0) as engine:
            analysis = engine.analyse(
                board, chess.engine.Limit(time=2.0), multipv=3
            )
    except Exception as e:
        raise RuntimeError(f"Stockfish analysis failed: {e}") from e

    if isinstance(analysis, list):
        if not analysis:
            raise RuntimeError("Stockfish returned no analysis.")
        best = analysis[0]
    else:
        best = analysis
    variation = best.get("pv") or []
    if not variation or variation[0] not in board.legal_moves:
        raise RuntimeError("Stockfish did not return a legal move.")

    recommended = variation[0]
    move_text = board.san(recommended)
    line_board = board.copy()
    line = []
    for move in variation[:6]:
        line.append(line_board.san(move))
        line_board.push(move)

    score = best.get("score")
    pov_score = score.pov(board.turn) if score else None
    if pov_score and pov_score.is_mate():
        mate_in = pov_score.mate()
        evaluation = f"mate in {mate_in}" if mate_in and mate_in > 0 else "mate is unavoidable"
    elif pov_score and pov_score.score() is not None:
        evaluation = f"{pov_score.score() / 100:+.2f} pawns"
    else:
        evaluation = "unavailable"

    side = "White" if board.turn else "Black"
    return (
        f"Recommended move for {side}: {move_text} ({recommended.uci()}). "
        f"Stockfish evaluation: {evaluation} from {side}'s perspective. "
        f"Principal variation: {' '.join(line)}. "
        "This is a recommendation only; no move was played."
    )


def chess_player(parameters: dict, player=None, speak=None) -> str:
    params = parameters or {}
    side_to_move = str(params.get("side_to_move", "auto")).strip().lower()
    if side_to_move not in {"auto", "white", "black"}:
        return "side_to_move must be auto, white, or black."

    if player:
        player.write_log("[Chess] Capturing the active window for board analysis.")
    if speak:
        speak(
            "I’ll inspect the active window with the configured vision model, "
            "then analyze the position locally with Stockfish. I won’t play a move."
        )

    try:
        image_bytes = _capture_active_window()
        fen, notes = _fen_from_screen(image_bytes, side_to_move)
        if not fen:
            detail = f" {notes}" if notes else ""
            return (
                "I couldn't confidently identify a complete chess position. "
                "Show the full board without overlays and try again, or specify "
                "side_to_move as white or black." + detail
            )
        result = _analyse_position(fen, side_to_move)
        if notes:
            result += f"\nBoard-reading note: {notes}"
    except Exception as e:
        result = f"Chess analysis failed, sir: {e}"

    if speak:
        speak(result)
    return result


TOOL = {
    "name": "chess_player",
    "description": (
        "Analyze the visible chessboard in the active window and recommend "
        "Stockfish's best next move. This captures the active window for "
        "configured vision analysis, then analyzes locally. It never clicks "
        "or plays a move."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "side_to_move": {
                "type": "STRING",
                "description": (
                    "Optional turn override: auto (default), white, or black. "
                    "Use white/black if the interface does not clearly show whose turn it is."
                ),
                "enum": ["auto", "white", "black"],
            }
        },
        "required": [],
    },
    "handler": chess_player,
}
