# app.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import re
import os
import shutil
import sys
import textwrap

# Platform-specific imports for single-character input
try:
    # Unix-like systems (Linux, macOS)
    import tty
    import termios
except ImportError:
    # Windows
    import msvcrt

# ------------------------
# ANSI escape codes for styling
# ------------------------
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J"
CURSOR_HOME = "\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# ------------------------
# --- THEME CUSTOMIZATION ---
# ------------------------
# BG: #00005f (dark blue) -> 256-color code 17
# FG: #87d7ff (light blue) -> 256-color code 117
BG_COLOR = "\033[48;5;17m"
FG_COLOR = "\033[38;5;117m"
# Highlight colors are the inverse of the main theme
HIGHLIGHT_BG_COLOR = "\033[48;5;117m"
HIGHLIGHT_FG_COLOR = "\033[38;5;17m"
# ------------------------

# ------------------------
# Helper Functions
# ------------------------
_ansi_escape_re = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def get_visible_length(s: str) -> int:
    """Calculates the visible length of a string by removing ANSI escape codes."""
    return len(_ansi_escape_re.sub('', s))

def get_single_char():
    """Gets a single character from standard input without requiring Enter."""
    if 'msvcrt' in sys.modules:
        return msvcrt.getch().decode('utf-8', errors='ignore')
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char

# ------------------------
# Markdown -> Heading tree
# ------------------------
@dataclass
class Node:
    level: int
    title: str
    bullets: List[str] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

class Doc:
    def __init__(self, raw: str) -> None:
        self.roots: List[Node] = []
        self.slides: List[dict] = []
        self._parse(raw)
        self._build_slides_depth_first()

    def _preprocess_markdown(self, raw: str) -> List[str]:
        lines = raw.splitlines()
        out: List[str] = []
        in_html, in_pct = False, False
        gfm_inline_html = re.compile(r'<!--.*?-->', flags=re.DOTALL)
        gfm_inline_pct = re.compile(r'%%.*?%%')
        gfm_comment_line1 = re.compile(r'^\s*\[//\]:\s*#\s*\((.*?)\)\s*$')
        gfm_comment_line2 = re.compile(r'^\s*\[[^\]]+\]:\s*<>\s*\((.*?)\)\s*$')

        for line in lines:
            s = line
            if in_html:
                if '-->' in s: s, in_html = s.split('-->', 1)[1], False
                else: continue
            if in_pct:
                if '%%' in s: s, in_pct = s.split('%%', 1)[1], False
                else: continue
            s = gfm_inline_html.sub('', s)
            s = gfm_inline_pct.sub('', s)
            if '<!--' in s and '-->' not in s: s, in_html = s.split('<!--', 1)[0], True
            if s.count('%%') % 2 == 1: s, in_pct = s.split('%%', 1)[0], True
            if gfm_comment_line1.match(s) or gfm_comment_line2.match(s): continue
            out.append(s if s.strip() else "")
        return out

    def _parse(self, raw: str) -> None:
        stack: List[Node] = []
        current_para: List[str] = []
        bullet_re = re.compile(r"^[-*]\s+")
        numbered_re = re.compile(r"^(\d+)[\.\)]\s+")

        def start_node(level: int, title: str):
            nonlocal stack
            _flush_para()
            while stack and stack[-1].level >= level: stack.pop()
            node = Node(level=level, title=title)
            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else: self.roots.append(node)
            stack.append(node)

        def _flush_para():
            nonlocal current_para
            if current_para and stack:
                joined = " ".join(line.strip() for line in current_para).strip()
                if joined: stack[-1].bullets.append(joined)
            current_para = []

        for raw_line in self._preprocess_markdown(raw):
            s = raw_line.rstrip()
            if not s.strip(): _flush_para(); continue
            stripped = s.lstrip()
            if stripped.startswith("#"):
                _flush_para()
                i = 0
                while i < len(stripped) and stripped[i] == "#": i += 1
                if i > 0 and i < len(stripped) and stripped[i] == " ":
                    start_node(i, stripped[i + 1:].strip() or " ")
                continue
            if bullet_re.match(stripped):
                _flush_para()
                if stack: stack[-1].bullets.append(bullet_re.sub("", stripped).strip())
                continue
            m = numbered_re.match(stripped)
            if m:
                _flush_para()
                if stack: stack[-1].bullets.append(f"{m.group(1)}. {numbered_re.sub('', stripped).strip()}")
                continue
            current_para.append(raw_line)
        _flush_para()

    def _path(self, node: Node) -> List[Node]:
        out = [node]
        p = node.parent
        while p: out.append(p); p = p.parent
        out.reverse()
        return out

    def _siblings_titles(self, node: Node) -> List[str]:
        if node.parent: return [c.title for c in node.parent.children]
        return [r.title for r in self.roots]

    def _build_slides_depth_first(self) -> None:
        def visit(n: Node):
            path = self._path(n)
            self.slides.append({
                "node": n, "path": path, "body": list(n.bullets),
                "siblings_map": [(self._siblings_titles(p), p.title) for p in path],
            })
            for c in n.children: visit(c)
        for r in self.roots: visit(r)

# ------------------------
# App
# ------------------------
class TerminalPresenter:
    def __init__(self) -> None:
        self.doc: Doc = Doc("")
        self.slide_index = 0
        self.phase = 0
        self.sent_idx = -1
        self.running = True

    def _md_path(self) -> Path:
        return Path(__file__).with_name("1h_Intro_to_LLMs.md")

    def _load_markdown(self) -> str:
        md_path = self._md_path()
        if md_path.exists():
            try: return md_path.read_text(encoding="utf-8")
            except Exception as e: return f"# Error\n\n- Could not read file: {e}"
        return "# Missing file\n\n- Put `1h_Intro_to_LLMs.md` next to this script."

    def _setup(self) -> None:
        self.doc = Doc(self._load_markdown())
        self.slide_index, self.phase, self.sent_idx = 0, 0, -1

    def _strip_markdown_links(self, text: str) -> str:
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)
        text = re.sub(r'<https?://[^>]+>', '', text)
        return re.sub(r'\s{2,}', ' ', text).strip()

    def _split_sentences(self, body_items: List[str]) -> List[str]:
        sents: List[str] = []
        numbered_re = re.compile(r"^\d+[\.\)]\s+")
        splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        for item in body_items:
            text = self._strip_markdown_links((item or "").strip())
            if not text: continue
            if numbered_re.match(text): sents.append(text); continue
            parts = splitter.split(text)
            for p in (p.strip() for p in parts if p.strip()): sents.append(p)
        return sents

    def _render_page(self) -> None:
        output = [CURSOR_HOME, CLEAR_SCREEN]
        width = shutil.get_terminal_size().columns
        slide = self.doc.slides[self.slide_index]
        path, siblings_map = slide["path"], slide["siblings_map"]
        for i, n in enumerate(path):
            titles, current_title = siblings_map[i]
            if i == 0:
                output.extend(self._render_header(n.title, width, highlight=(self.phase >= 1)))
            else:
                output.extend(self._render_siblings_bar(titles, current_title, width, highlight=(i < self.phase)))
        sentences = self._split_sentences(slide["body"])
        if sentences and self.phase >= (len(path) + 1):
            output.extend(self._render_body(sentences, self.sent_idx, width))
        sys.stdout.write("".join(output))
        sys.stdout.flush()

    def _render_header(self, title: str, width: int, highlight: bool) -> List[str]:
        title_txt = title.upper()
        if highlight:
            title_txt = f"{HIGHLIGHT_BG_COLOR}{HIGHLIGHT_FG_COLOR}{title_txt}{BG_COLOR}{FG_COLOR}"
        clock_txt = datetime.now().strftime("%H:%M")
        padding = max(1, width - get_visible_length(title_txt) - len(clock_txt) - 4)
        line = f"| {title_txt}{' ' * padding}{clock_txt} |"
        border = "+" + "-" * (width - 2) + "+"
        return [border, "\n", line, "\n", border, "\n"]

    def _render_siblings_bar(self, titles: List[str], current: str, width: int, highlight: bool) -> List[str]:
        if not titles: return []
        output, inner_width, sep, sep_len = [], width - 4, "  |  ", 5
        lines, current_line, current_len = [], [], 0
        for t in titles:
            label = t.upper()
            styled = f"{HIGHLIGHT_BG_COLOR}{HIGHLIGHT_FG_COLOR}{label}{BG_COLOR}{FG_COLOR}" if highlight and label == current.upper() else label
            needed = len(label) + (sep_len if current_line else 0)
            if current_line and current_len + needed > inner_width:
                lines.append(current_line); current_line, current_len = [], 0
            current_line.append(styled); current_len += needed
        if current_line: lines.append(current_line)
        for line in lines:
            line_str = sep.join(line)
            padding = inner_width - get_visible_length(line_str)
            output.append(f"| {line_str}{' ' * padding} |\n")
        output.append("+" + "-" * (width - 2) + "+\n")
        return output

    def _render_body(self, sentences: List[str], sent_idx: int, width: int) -> List[str]:
        output = ["\n"]
        inner_width = max(10, width - 6)
        for i, sentence in enumerate(sentences):
            wrapped_lines = textwrap.wrap(sentence.upper(), width=inner_width)
            for line in wrapped_lines:
                styled_line = f"{HIGHLIGHT_BG_COLOR}{HIGHLIGHT_FG_COLOR}{line}{BG_COLOR}{FG_COLOR}" if i == sent_idx else line
                output.append(f"  {styled_line}\n")
            if i < len(sentences) - 1:
                output.append("\n") # Reduced spacing
        return output

    @staticmethod
    def _cpl(a: List[Node], b: List[Node]) -> int:
        m = min(len(a), len(b)); i = 0
        while i < m and a[i] is b[i]: i += 1
        return i

    def _current_title_path(self) -> Tuple[str, ...]:
        slide = self.doc.slides[self.slide_index]
        return tuple(t.title.strip().upper() for t in slide["path"])

    def _find_slide_by_title_path(self, new_doc: Doc, path_titles: Tuple[str, ...]) -> int:
        for idx, s in enumerate(new_doc.slides):
            if tuple(t.title.strip().upper() for t in s["path"]) == path_titles: return idx
        return -1

    def action_next(self) -> None:
        slide = self.doc.slides[self.slide_index]
        path, has_body = slide["path"], any(b.strip() for b in slide["body"])
        max_phase = len(path) + (1 if has_body else 0)
        if self.phase < max_phase:
            self.phase += 1; self.sent_idx = -1; return
        if self.slide_index >= len(self.doc.slides) - 1: return
        prev_path = path
        self.slide_index += 1
        new_path = self.doc.slides[self.slide_index]["path"]
        k = self._cpl(prev_path, new_path)
        if len(new_path) == len(prev_path) and k == len(new_path) - 1: self.phase = min(len(new_path), self.phase)
        elif len(new_path) > len(prev_path) and k == len(prev_path): self.phase = k
        else: self.phase = min(k, len(new_path))
        self.sent_idx = -1

    def action_prev(self) -> None:
        if self.slide_index == 0: return
        curr_path = self.doc.slides[self.slide_index]["path"]
        self.slide_index -= 1
        prev_slide = self.doc.slides[self.slide_index]
        prev_path, prev_has_body = prev_slide["path"], any(b.strip() for b in prev_slide["body"])
        k = self._cpl(prev_path, curr_path)
        if len(prev_path) == len(curr_path) and k == len(prev_path) - 1: self.phase = min(len(prev_path), self.phase or len(prev_path))
        else: self.phase = len(prev_path) + (1 if prev_has_body else 0)
        self.sent_idx = -1

    def action_next_sentence(self) -> None:
        slide = self.doc.slides[self.slide_index]
        sentences = self._split_sentences(slide["body"])
        if sentences and self.phase >= (len(slide["path"]) + 1) and self.sent_idx < len(sentences) - 1:
            self.sent_idx += 1

    def action_prev_sentence(self) -> None:
        slide = self.doc.slides[self.slide_index]
        if self.phase >= (len(slide["path"]) + 1) and self.sent_idx >= 0:
            self.sent_idx -= 1

    def action_reload(self) -> None:
        try: new_doc = Doc(self._load_markdown())
        except Exception: return
        if not new_doc.slides: return
        old_path = self._current_title_path()
        new_index = self._find_slide_by_title_path(new_doc, old_path)
        self.doc = new_doc
        self.slide_index = new_index if new_index >= 0 else min(self.slide_index, len(self.doc.slides) - 1)
        slide = self.doc.slides[self.slide_index]
        has_body = any(b.strip() for b in slide["body"])
        max_phase = len(slide["path"]) + (1 if has_body else 0)
        self.phase = min(self.phase, max_phase)
        self.sent_idx = -1

    def run(self) -> None:
        try:
            # Set the theme and hide the cursor for the entire session
            sys.stdout.write(f"{BG_COLOR}{FG_COLOR}{HIDE_CURSOR}{CLEAR_SCREEN}")
            sys.stdout.flush()
            self._setup()
            while self.running:
                self._render_page()
                char = get_single_char()
                if char in ('j', '\r', '\n'): self.action_next()
                elif char == 'k': self.action_prev()
                elif char == 'l': self.action_next_sentence()
                elif char == 'h': self.action_prev_sentence()
                elif char == 'r': self.action_reload()
                elif char in ('q', '\x03'): self.running = False # q or Ctrl+C
        finally:
            # On exit, reset all styles, clear screen, and show cursor
            sys.stdout.write(f"{RESET}{CLEAR_SCREEN}{CURSOR_HOME}{SHOW_CURSOR}")
            sys.stdout.flush()

if __name__ == "__main__":
    TerminalPresenter().run()