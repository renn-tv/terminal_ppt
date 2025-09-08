# Terminal PPT

Terminal-based slide presenter built with Textual/Rich. Write your talk in Markdown; present with large FIGlet headings and high-contrast terminal visuals. Keyboard-only, fast, and robust.

## Features

* **Markdown → slides**

  * Headings (`#`, `##`, …) form the slide path.
  * Bullets, numbered items, and paragraphs become the body.
  * Links render as **label only** (URLs hidden).
  * Presenter comments are ignored: HTML (`<!-- -->`), Obsidian (`%% %%`), GFM (`[//]: # (...)`, `[note]: <> (...)`).
* **Always-visible context**

  * Top line shows level-1 heading with a clock.
  * Each deeper level shows all sibling headings; current sibling is inverted.
* **Navigation model**

  * Depth-first order.
  * Body appears one click after the deepest heading is revealed.
  * Moving between siblings stays highlighted (no intermediate un-highlight).
* **Manual reload**

  * Press `r` to re-read the Markdown.
  * Remains on the **same slide** and **keeps highlight phase**; body stays visible if it was.
* **Rendering stability**

  * No wrapping, cropped overflow, clipped during resize to avoid jitter.
* **Theme & fonts**

  * Two-color theme (default 256-safe): background `#00005f`, text `#87d7ff`.
  * FIGlet font `future`; toggle FIGlet/plain at runtime.

## Requirements

* Python 3.9+
* Packages: `textual`, `rich`, `pyfiglet`

## Quick Start

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install textual rich pyfiglet
python app.py
```

Place your deck as `1h_Intro_to_LLMs.md` next to `app.py` (or adjust the path in the code).

## Key Bindings

| Key | Action                                      |
| --- | ------------------------------------------- |
| ↓   | Next (reveal deeper → next slide)           |
| ↑   | Previous heading slide                      |
| →   | Highlight next sentence in body             |
| ←   | Highlight previous sentence in body         |
| t   | Toggle FIGlet ↔ plain text                  |
| r   | Reload Markdown (stay on slide, keep phase) |
| q   | Quit                                        |

## Authoring Notes

* Headings create the navigation path; content under a heading appears on that slide.
* Numbered bullets remain whole lines; paragraphs split on sentence boundaries for highlighting.
* Comments that won’t render:

  * `<!-- hidden -->`, `<!-- block ... -->`
  * `%% hidden %%`, `%% block ... %%`
  * `[//]: # (hidden)`, `[note]: <> (hidden)`

## Configuration

* Colors: edit `BG` and `FONT` constants at the top of `app.py`.

  * Defaults are 256-palette friendly; truecolor hex also works in capable terminals.
* FIGlet font: change in `HeadingFig` / `BodyFig`.
* Markdown file name: adjust `LessonApp._md_path()`.

## Notes on Colors

* Truecolor terminals render exact hex; others downsample to the nearest palette color.
* Defaults (`#00005f` / `#87d7ff`) look consistent on 256-color terminals.

## Project Layout

```
app.py                   # Textual application
1h_Intro_to_LLMs.md      # Source Markdown deck
```
