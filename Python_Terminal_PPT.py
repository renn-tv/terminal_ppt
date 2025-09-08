# app.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import re

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Static

from pyfiglet import Figlet

# ------------------------
# Two-color theme (256-safe hex)
# ------------------------
BG   = "#00005f"   # ≈ xterm-256 color 17 (deep navy)
FONT = "#87d7ff"   # ≈ xterm-256 color 117 (light blue)

# ------------------------
# FIGlet helpers
# ------------------------
def _font_candidates(name: str) -> List[str]:
    c = {
        name,
        name.lower(),
        name.replace(" ", "_"),
        name.replace(" ", ""),
        name.replace("-", "_"),
    }
    return list(c)

class FigBlock:
    def __init__(self, font: str) -> None:
        self.fig = self._mk(font)

    def _mk(self, font: str) -> Figlet:
        for cand in _font_candidates(font):
            try:
                return Figlet(font=cand)
            except Exception:
                continue
        return Figlet(font="standard")

    def render(self, text: str, width: int, cap_width: int) -> Text:
        # Keep figlet wide enough, but never exceed cap_width
        self.fig.width = max(40, min(width, cap_width))
        art = self.fig.renderText(text)
        # IMPORTANT: never let Rich wrap; crop if too wide during resize
        return Text(art.rstrip("\n"), style=FONT, no_wrap=True, overflow="crop")

# All FIGlet in "future"
class HeadingFig(FigBlock):
    def __init__(self) -> None:
        super().__init__("future")

class BodyFig(FigBlock):
    def __init__(self) -> None:
        super().__init__("future")

# ------------------------
# L1 header (clock on the right inside same panel)
# ------------------------
class HeadingPanel(Static):
    def __init__(self, title: str, show_clock: bool = False, highlight: bool = False, fig_mode: bool = True) -> None:
        super().__init__()
        self.title = title
        self.show_clock = show_clock
        self.highlight = highlight
        self.fig_mode = fig_mode
        self._heading = HeadingFig()
        self._clock = BodyFig() if show_clock else None
        self._last_min: Optional[str] = None

    def on_mount(self) -> None:
        if self.show_clock:
            self.set_interval(1, self._refresh)
        self._refresh()

    def on_resize(self) -> None:
        self._refresh()

    def _render_title(self, width: int) -> Text:
        txt = self.title.upper()
        if self.fig_mode:
            out = self._heading.render(txt, width, cap_width=max(60, width))
        else:
            out = Text(txt, style=FONT, no_wrap=True, overflow="crop")
        if self.highlight:
            out = Text(out.plain, style=f"black on {FONT}", no_wrap=True, overflow="crop")
        return out

    def _render_clock(self, width: int) -> Text:
        if not self.show_clock:
            return Text("")
        now = datetime.now().strftime("%H:%M")
        self._last_min = now
        if self.fig_mode and self._clock:
            return self._clock.render(now, width, cap_width=min(48, width))
        else:
            return Text(now, style=FONT, no_wrap=True, overflow="crop")

    def _refresh(self) -> None:
        total_w = self.app.size.width if hasattr(self.app, "size") else 120
        inner = max(8, total_w - 4)  # account for panel borders + padding
        left_ratio, right_ratio = 8, 2
        left_w = int(inner * left_ratio / (left_ratio + right_ratio)) if self.show_clock else inner
        right_w = inner - left_w if self.show_clock else 0

        title_txt = self._render_title(left_w)
        clock_txt = self._render_clock(right_w)

        grid = Table.grid(expand=True, padding=(0, 0))
        grid.add_column(ratio=left_ratio)
        if self.show_clock:
            grid.add_column(ratio=right_ratio, justify="right", no_wrap=True)
            grid.add_row(Align.left(title_txt), Align.right(clock_txt))
        else:
            grid.add_row(Align.left(title_txt))

        self.update(Panel(grid, border_style=FONT, style=FONT, padding=(0, 1)))

# ------------------------
# Siblings bar (FIGlet chips or plain chips; wraps ONLY between headings)
# ------------------------
class SiblingsBar(Static):
    def __init__(self, titles: List[str], current: str, highlight: bool, fig_mode: bool = True, sep: str = "   |   ") -> None:
        super().__init__()
        self.titles = [t for t in titles if str(t).strip()]
        self.current = current
        self.highlight = highlight
        self.fig_mode = fig_mode
        self.sep = sep
        self._fig = Figlet(font="future")
        self._fig.width = 10000  # disable figlet internal wrap

    def on_mount(self) -> None:
        self._refresh()

    def on_resize(self) -> None:
        self._refresh()

    # ---- FIGlet chip helpers ----
    def _make_fig_chip(self, label: str, invert: bool) -> tuple[List[str], int, str]:
        art = self._fig.renderText(label.upper()).rstrip("\n")
        lines = art.splitlines()
        w = max((len(l) for l in lines), default=0)
        style = f"black on {FONT}" if invert else FONT
        return lines, w, style

    def _render_block_figlet(self, width: int) -> Text:
        if not self.titles:
            return Text("", style=FONT)

        chips = []
        cur_u = self.current.upper()
        for t in self.titles:
            invert = self.highlight and (t.upper() == cur_u)
            chips.append(self._make_fig_chip(t, invert))

        max_w = max(20, width - 4)
        sep_len = len(self.sep)
        rows: list[list[int]] = []
        cur_row: list[int] = []
        used = 0
        for i, (_, w, _) in enumerate(chips):
            need = (w if not cur_row else w + sep_len)
            if used and used + need > max_w:
                rows.append(cur_row)
                cur_row = [i]
                used = w
            else:
                cur_row.append(i)
                used += need
        if cur_row:
            rows.append(cur_row)

        out = Text()
        for r_idx, row in enumerate(rows):
            row_height = max(len(chips[i][0]) for i in row)
            norm = []
            for i in row:
                lines, w, style = chips[i]
                pad_lines = lines + [""] * (row_height - len(lines))
                norm.append((pad_lines, w, style))
            for line_idx in range(row_height):
                line_text = Text(no_wrap=True, overflow="crop")
                for c_idx, (lines, w, style) in enumerate(norm):
                    segment = lines[line_idx]
                    padded = segment + (" " * max(0, w - len(segment)))  # pad to width
                    line_text.append(Text(padded, style=style, no_wrap=True, overflow="crop"))
                    if c_idx < len(norm) - 1:
                        line_text.append(Text(self.sep, style=FONT, no_wrap=True, overflow="crop"))
                out.append(line_text)
                if not (r_idx == len(rows) - 1 and line_idx == row_height - 1):
                    out.append("\n")
        return out

    # ---- Plain chip helpers ----
    def _render_block_plain(self, width: int) -> Text:
        if not self.titles:
            return Text("", style=FONT)
        chips: List[Text] = []
        lens: List[int] = []
        cur = self.current.upper()
        for t in self.titles:
            label = t.upper()
            style = f"black on {FONT}" if (self.highlight and label == cur) else FONT
            chip = Text(label, style=style, no_wrap=True, overflow="crop")
            chips.append(chip)
            lens.append(len(chip.plain))

        sep_text = Text(self.sep, style=FONT, no_wrap=True, overflow="crop")
        sep_len = len(self.sep)
        max_w = max(20, width - 4)

        lines: List[Text] = []
        line = Text(no_wrap=True, overflow="crop")
        used = 0
        for i, chip in enumerate(chips):
            need = lens[i] + (sep_len if i < len(chips) - 1 else 0)
            if used > 0 and used + need > max_w:
                lines.append(line)
                line = Text(no_wrap=True, overflow="crop")
                used = 0
            line.append(chip)
            used += lens[i]
            if i < len(chips) - 1:
                line.append(sep_text)
                used += sep_len
        lines.append(line)

        out = Text(no_wrap=True, overflow="crop")
        for idx, l in enumerate(lines):
            out.append(l)
            if idx < len(lines) - 1:
                out.append("\n")
        return out

    def _refresh(self) -> None:
        total_w = self.app.size.width if hasattr(self.app, "size") else 120
        inner = max(8, total_w - 4)  # borders + padding
        block = self._render_block_figlet(inner) if self.fig_mode else self._render_block_plain(inner)
        self.update(Panel(Align.left(block), border_style=FONT, style=FONT, padding=(0, 1)))

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

    # ---------- presenter/comments preprocessor ----------
    def _preprocess_markdown(self, raw: str) -> List[str]:
        """Strip comments before parsing:
           - HTML: <!-- ... --> (inline or multi-line)
           - Obsidian: %% ... %% (inline or multi-line)
           - GFM tricks: [//]: # ( ... )  and  [note]: <> ( ... )
        """
        lines = raw.splitlines()
        out: List[str] = []
        in_html = False
        in_pct = False

        gfm_inline_html = re.compile(r'<!--.*?-->', flags=re.DOTALL)
        gfm_inline_pct  = re.compile(r'%%.*?%%')
        gfm_comment_line1 = re.compile(r'^\s*\[//\]:\s*#\s*\((.*?)\)\s*$')
        gfm_comment_line2 = re.compile(r'^\s*\[[^\]]+\]:\s*<>\s*\((.*?)\)\s*$')

        for line in lines:
            s = line

            # inside HTML block?
            if in_html:
                if '-->' in s:
                    s = s.split('-->', 1)[1]
                    in_html = False
                else:
                    continue

            # inside %% block?
            if in_pct:
                if '%%' in s:
                    s = s.split('%%', 1)[1]
                    in_pct = False
                else:
                    continue

            # remove inline pairs first
            s = gfm_inline_html.sub('', s)
            s = gfm_inline_pct.sub('', s)

            # start of HTML block (unclosed)
            if '<!--' in s and '-->' not in s:
                s = s.split('<!--', 1)[0]
                in_html = True

            # start of %% block (odd count after inline removal)
            if s.count('%%') % 2 == 1:
                s = s.split('%%', 1)[0]
                in_pct = True

            # whole-line GFM comments
            if gfm_comment_line1.match(s) or gfm_comment_line2.match(s):
                continue

            out.append(s if s.strip() else "")
        return out

    # ---------- markdown parser ----------
    def _parse(self, raw: str) -> None:
        stack: List[Node] = []
        current_para: List[str] = []

        bullet_re = re.compile(r"^[-*]\s+")
        numbered_re = re.compile(r"^(\d+)[\.\)]\s+")

        def start_node(level: int, title: str):
            nonlocal stack
            _flush_para()
            while stack and stack[-1].level >= level:
                stack.pop()
            node = Node(level=level, title=title)
            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else:
                self.roots.append(node)
            stack.append(node)

        def _flush_para():
            nonlocal current_para
            if current_para and stack:
                joined = " ".join(line.strip() for line in current_para).strip()
                if joined:
                    stack[-1].bullets.append(joined)
            current_para = []

        for raw_line in self._preprocess_markdown(raw):
            s = raw_line.rstrip()
            if not s.strip():
                _flush_para()
                continue

            stripped = s.lstrip()
            if stripped.startswith("#"):
                _flush_para()
                i = 0
                while i < len(stripped) and stripped[i] == "#":
                    i += 1
                if i > 0 and i < len(stripped) and stripped[i] == " ":
                    title = stripped[i + 1:].strip() or " "
                    start_node(i, title)
                continue

            if bullet_re.match(stripped):
                _flush_para()
                if stack:
                    stack[-1].bullets.append(bullet_re.sub("", stripped).strip())
                continue

            m = numbered_re.match(stripped)
            if m:
                _flush_para()
                if stack:
                    num = m.group(1)
                    stack[-1].bullets.append(f"{num}. {numbered_re.sub('', stripped).strip()}")
                continue

            current_para.append(raw_line)

        _flush_para()

    # ---------- tree utilities ----------
    def _path(self, node: Node) -> List[Node]:
        out: List[Node] = [node]
        p = node.parent
        while p:
            out.append(p)
            p = p.parent
        out.reverse()
        return out

    def _siblings_titles(self, node: Node) -> List[str]:
        if node.parent:
            return [c.title for c in node.parent.children]
        return [r.title for r in self.roots]

    # ---------- slides ----------
    def _build_slides_depth_first(self) -> None:
        def visit(n: Node):
            path = self._path(n)
            self.slides.append(
                {
                    "node": n,
                    "path": path,
                    "body": list(n.bullets),
                    "siblings_map": [(self._siblings_titles(p), p.title) for p in path],
                }
            )
            for c in n.children:
                visit(c)
        for r in self.roots:
            visit(r)

# ------------------------
# Page (headers + siblings + body sentences with highlight)
# ------------------------
BODY_SPACING_BLANK_LINES = 3

class Page(Static):
    def __init__(self, path: List[Node], siblings_map: List[Tuple[List[str], str]],
                 sentences: List[str], sent_idx: int, phase: int, fig_mode: bool) -> None:
        super().__init__()
        self.path = path
        self.siblings_map = siblings_map
        self.sentences = sentences
        self.sent_idx = sent_idx
        self.phase = phase
        self.fig_mode = fig_mode
        self._body_fig = BodyFig()

    def compose(self) -> ComposeResult:
        depth = len(self.path)
        for i, n in enumerate(self.path):
            titles, current_title = self.siblings_map[i]
            if i == 0:
                yield HeadingPanel(n.title, show_clock=True, highlight=(self.phase >= 1), fig_mode=self.fig_mode)
            else:
                yield SiblingsBar(titles, current_title, highlight=(i < self.phase), fig_mode=self.fig_mode)

        # body shows one click after deepest level highlighted
        if self.sentences and self.phase >= (depth + 1):
            yield Static(self._render_body(), id="body")

    def _render_body(self) -> Panel:
        total_w = self.app.size.width if hasattr(self.app, "size") else 120
        inner = max(8, total_w - 4)  # borders + padding
        blocks: List[Text] = []
        spacer = Text("\n" * BODY_SPACING_BLANK_LINES, style=FONT)

        for i, sentence in enumerate(self.sentences):
            if self.fig_mode:
                fig = self._body_fig.render(sentence.upper(), min(inner, 260), cap_width=260)
            else:
                fig = Text(sentence.upper(), style=FONT, no_wrap=True, overflow="crop")
            if i == self.sent_idx:
                fig = Text(fig.plain, style=f"black on {FONT}", no_wrap=True, overflow="crop")
            blocks.append(fig)
            if i < len(self.sentences) - 1:
                blocks.append(spacer)

        body = Text.assemble(*blocks) if blocks else Text("")
        return Panel(Align.left(body, vertical="top"), border_style=FONT, style=FONT, padding=(0, 1))

# ------------------------
# App
# ------------------------
class LessonApp(App):
    CSS = f"""
    Screen {{
        background: {BG};
    }}
    Static {{
        overflow: hidden;     /* clip during resize, don't wrap */
    }}
    #stage {{
        height: 1fr;
        width: 100%;
        padding: 0 2;
    }}
    #body {{
        margin: 0;
    }}
    """

    BINDINGS = [
        ("down", "next", "Next"),
        ("up", "prev", "Prev"),
        ("right", "next_sentence", "Next sentence"),
        ("left", "prev_sentence", "Prev sentence"),
        ("t", "toggle_font", "Toggle font"),
        ("r", "reload", "Reload markdown"),
        ("q", "quit", "Quit"),
    ]

    slide_index = reactive(0)
    phase = reactive(0)
    sent_idx = reactive(-1)
    fig_mode = reactive(True)  # True = FIGlet 'future', False = normal terminal text

    def _md_path(self) -> Path:
        return Path(__file__).with_name("1h_Intro_to_LLMs.md")

    def _load_markdown(self) -> str:
        md_path = self._md_path()
        if md_path.exists():
            try:
                return md_path.read_text(encoding="utf-8")
            except Exception:
                pass
        return "# Missing file\n\n- Put `1h_Intro_to_LLMs.md` next to app.py."

    def compose(self) -> ComposeResult:
        yield Container(id="stage")

    def on_mount(self) -> None:
        self.doc = Doc(self._load_markdown())
        self.slide_index = 0
        self.phase = 0
        self.sent_idx = -1
        self._render_page()

    # ---- markdown helpers ----
    def _strip_markdown_links(self, text: str) -> str:
        """Keep link text, drop URLs for Markdown links/images/autolinks."""
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)     # images -> alt
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)       # [label](url) -> label
        text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)      # [label][ref] -> label
        text = re.sub(r'<https?://[^>]+>', '', text)               # <http://...> -> remove
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _split_sentences(self, body_items: List[str]) -> List[str]:
        """Split body into sentences. Numbered bullets remain whole; paragraphs split on . ! ? followed by space."""
        sents: List[str] = []
        numbered_re = re.compile(r"^\d+[\.\)]\s+")
        splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        for item in body_items:
            text = (item or "").strip()
            if not text:
                continue
            text = self._strip_markdown_links(text)
            if not text:
                continue
            if numbered_re.match(text):
                sents.append(text)
                continue
            parts = splitter.split(text)
            for p in parts:
                p = p.strip()
                if p:
                    sents.append(p)
        return sents

    # ---- render ----
    def _render_page(self) -> None:
        stage = self.query_one("#stage", Container)
        stage.remove_children()
        slide = self.doc.slides[self.slide_index]
        sentences = self._split_sentences(slide["body"])
        page = Page(slide["path"], slide["siblings_map"], sentences, self.sent_idx, phase=self.phase, fig_mode=self.fig_mode)
        stage.mount(page)
        path_titles = " / ".join(n.title for n in slide["path"])
        mode = "FIGLET" if self.fig_mode else "TEXT"
        self.sub_title = f"{self.slide_index+1}/{len(self.doc.slides)} — {path_titles} — phase {self.phase} — {mode}"

    @staticmethod
    def _cpl(a: List[Node], b: List[Node]) -> int:
        m = min(len(a), len(b)); i = 0
        while i < m and a[i] is b[i]:
            i += 1
        return i

    def _current_title_path(self) -> Tuple[str, ...]:
        """Uppercased, trimmed titles from L1 -> current slide."""
        slide = self.doc.slides[self.slide_index]
        return tuple(t.title.strip().upper() for t in slide["path"])

    def _find_slide_by_title_path(self, new_doc: Doc, path_titles: Tuple[str, ...]) -> int:
        """Return index of slide whose path titles match exactly; else -1."""
        for idx, s in enumerate(new_doc.slides):
            titles = tuple(t.title.strip().upper() for t in s["path"])
            if titles == path_titles:
                return idx
        return -1

    # ---- navigation ----
    def action_next(self) -> None:
        slide = self.doc.slides[self.slide_index]
        path = slide["path"]
        depth = len(path)
        has_body = bool([b for b in slide["body"] if str(b).strip()])
        max_phase = depth + (1 if has_body else 0)

        if self.phase < max_phase:
            self.phase += 1
            self.sent_idx = -1
            self._render_page()
            return

        if self.slide_index >= len(self.doc.slides) - 1:
            return

        prev_path = path
        self.slide_index += 1
        new_slide = self.doc.slides[self.slide_index]
        new_path = new_slide["path"]

        k = self._cpl(prev_path, new_path)

        if len(new_path) == len(prev_path) and k == len(new_path) - 1:
            # next sibling -> keep that level highlighted (no interim un-highlight)
            self.phase = min(len(new_path), self.phase)
        elif len(new_path) == len(prev_path) + 1 and k == len(prev_path):
            # going deeper -> new level starts un-highlighted
            self.phase = k
        else:
            # general case -> highlight only shared prefix
            self.phase = min(k, len(new_path))

        self.sent_idx = -1
        self._render_page()

    def action_prev(self) -> None:
        """Arrow Up navigates headings only: jump to previous node."""
        if self.slide_index == 0:
            return

        curr_path = self.doc.slides[self.slide_index]["path"]
        self.slide_index -= 1
        prev_slide = self.doc.slides[self.slide_index]
        prev_path = prev_slide["path"]

        k = self._cpl(prev_path, curr_path)

        if len(prev_path) == len(curr_path) and k == len(prev_path) - 1:
            # previous sibling -> keep that level highlighted
            self.phase = min(len(prev_path), self.phase or len(prev_path))
        else:
            # default: land fully revealed (incl. body if any)
            prev_has_body = bool([b for b in prev_slide["body"] if str(b).strip()])
            self.phase = len(prev_path) + (1 if prev_has_body else 0)

        self.sent_idx = -1
        self._render_page()

    def action_first(self) -> None:
        self.slide_index = 0
        self.phase = 0
        self.sent_idx = -1
        self._render_page()

    def action_last(self) -> None:
        self.slide_index = len(self.doc.slides) - 1
        last = self.doc.slides[self.slide_index]
        last_depth = len(last["path"])
        last_has_body = bool([b for b in last["body"] if str(b).strip()])
        self.phase = last_depth + (1 if last_has_body else 0)
        self.sent_idx = -1
        self._render_page()

    # ---- sentence highlight ----
    def action_next_sentence(self) -> None:
        slide = self.doc.slides[self.slide_index]
        depth = len(slide["path"])
        sentences = self._split_sentences(slide["body"])
        if not sentences or self.phase < (depth + 1):
            return
        if self.sent_idx < len(sentences) - 1:
            self.sent_idx += 1
            self._render_page()

    def action_prev_sentence(self) -> None:
        slide = self.doc.slides[self.slide_index]
        depth = len(slide["path"])
        sentences = self._split_sentences(slide["body"])
        if not sentences or self.phase < (depth + 1):
            return
        if self.sent_idx >= 0:
            self.sent_idx -= 1
            self._render_page()

    # ---- font toggle ----
    def action_toggle_font(self) -> None:
        self.fig_mode = not self.fig_mode
        self._render_page()

    # ---- manual reload (super simple) ----
    def action_reload(self) -> None:
        """Re-read Markdown, rebuild slides, and stay on the same slide (by title path).
        Preserve phase (clamped). Reset sentence highlight."""
        try:
            raw = self._md_path().read_text(encoding="utf-8")
            new_doc = Doc(raw)
        except Exception:
            return  # fail silently; keep current deck

        # Match current slide by path titles (stable if headings unchanged)
        old_path_titles = self._current_title_path()
        new_index = self._find_slide_by_title_path(new_doc, old_path_titles)
        if new_index < 0:
            # Fallback: clamp to last slide if indices changed
            new_index = min(self.slide_index, len(new_doc.slides) - 1) if new_doc.slides else 0

        # Swap doc and clamp phase to new slide's max
        if not new_doc.slides:
            return  # nothing to show
        self.doc = new_doc
        self.slide_index = new_index

        slide = self.doc.slides[self.slide_index]
        depth = len(slide["path"])
        has_body = bool([b for b in slide["body"] if str(b).strip()])
        max_phase = depth + (1 if has_body else 0)
        self.phase = min(self.phase, max_phase)  # keep body visible if it was

        self.sent_idx = -1  # text can be un-highlighted after reload
        self._render_page()

if __name__ == "__main__":
    LessonApp().run()
