# Terminal PPT

A lightweight, zero-dependency presentation tool that runs directly in your terminal. It renders presentations from a simple Markdown file, allowing you to focus on content while providing a clean, retro-style presentation experience.

![Demo Screenshot](https://dentro.de/ai/images/terminal_ppt.png)

## Features

-   **Zero Dependencies**: Runs using only Python's standard library. No `pip install` required.
-   **Markdown-Driven**: Slides, headings, and bullet points are generated directly from a `.md` file.
-   **Cross-Platform**: Works on Windows, macOS, and Linux terminals.
-   **Incremental Animation**: Reveal content step-by-step: first headers, then the body, and finally sentence-by-sentence highlighting.
-   **Themable**: Easily customize the foreground and background colors with ANSI color codes.
-   **Responsive Layout**: The display automatically adjusts to your terminal's width.
-   **Instant Input**: Navigate your presentation with single key presses—no need to press `Enter`.
-   **Live Updates**: Change and save the source markdown file and see the updates directly in the presentation.
-   **Clock included**: On top of screen, needs reload for update.
-   **Location / Breadcrump**: Always visible with highlighting of nested header structure, also shows progress to audience.

## Prerequisites

-   Python 3.6 or higher.

## Installation

No installation is needed. Simply clone the repository or download the Python script.

## Usage

The application is designed to be simple to use. Just create your content in a Markdown file and run the script. Scale the terminal window and increase the terminal font size for presentation mode. You can keep source markdown and terminal open at same time for live updates.

### 1. Create Your Markdown File

Create a file named `1h_Intro_to_LLMs.md` in the same directory as the script. The script is hardcoded to look for this file, but you can change the filename.

### 2. Add Your Content

The presentation structure is based on Markdown headings and lists.

-   `# Heading 1` creates a main slide topic.
-   `## Heading 2` creates a sub-topic under the last `Heading 1`. The tool supports multiple levels of headings (`###`, `####`, etc.).
-   `- A bullet point.` or `* A bullet point.` creates a text item on the current slide.
-   Numbered lists (`1. First item`) are also supported.
-   Presenter notes or comments can be added using standard HTML (`<!-- comment -->`) or Obsidian (`%% comment %%`) syntax and will be ignored.

**Example `1h_Intro_to_LLMs.md`:**

    # LLM Intro and Howto

    <!-- This is a presenter note and will not be displayed. -->

    ## How They Are Created
    - This is the first point about creation.
    - This is the second. It can be a very long sentence, and the application will automatically wrap the text to fit the screen.

    ## Technical Details
    1. Detail one: Pre-training.
    2. Detail two: Fine-tuning.

    ## How To Use
    - By using APIs.
    - Or by running local models.

    ## Questions?
    - This is the final slide for questions.

### 3. Run the Application

Execute the Python script from your terminal:

    python terminal_ppt.py

### 4. Navigate Your Presentation

Use the following keys to navigate:

| Key(s)               | Action                                     |
| -------------------- | ------------------------------------------ |
| `j` or `Enter`       | Advance to the next step (reveal or slide) |
| `k`                  | Go to the previous slide (fully revealed)  |
| `l`                  | Highlight the next sentence in the body    |
| `h`                  | Highlight the previous sentence in the body|
| `r`                  | Reload the content from the Markdown file  |
| `q` or `Ctrl+C`      | Quit the application                       |

## Customization

You can easily change the color theme by editing the constants at the top of the script. The colors are defined using 256-color ANSI escape codes.

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

You can find a reference for 256-color codes [here](https://www.ditig.com/256-colors-cheat-sheet) to pick your own theme.
