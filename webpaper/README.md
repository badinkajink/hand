# MorphoHand project site (Typst → HTML)

A research-paper-grade, media-rich web write-up of the whole project: morphology /
grasp optimization, RL manipulation, and hardware validation. Authored in
[Typst](https://typst.app) and exported to **static HTML** — so it carries
paper-quality math *and* embedded video, which neither a `.tex` PDF nor plain
Markdown does well.

## Build

```bash
webpaper/build.sh           # compiles src/*.typ -> build/*.html, copies assets
python3 -m http.server -d webpaper/build 8080   # then open http://localhost:8080
```

## Structure

```
webpaper/
  src/
    template.typ     shared: theme CSS, math->SVG show rule, media()/callout() helpers, nav
    index.typ        landing / overview (3 cards)
    morphology.typ   §1 morphology + grasp optimization   (outline, port from paper/)
    rl.typ           §2 RL manipulation + policy switching (fully written)
    hardware.typ     §3 hardware validation                (outline, port from hand_paper/)
    assets/          curated media referenced by pages (videos/images)
  build.sh           build script (all the Typst flags baked in)
  build/             generated HTML (gitignored)
```

Add a page: create `src/<name>.typ`, add `<name>` to `PAGES` in `build.sh`, and add
a nav entry to `nav-links` in `template.typ`.

## Typst-HTML gotchas (why the flags are what they are)

These cost real time to discover — do not relearn them:

1. **HTML export needs `--features html`** and is marked experimental in 0.14.x. It
   works for our needs; the build script silences the boilerplate warnings.
2. **Bare `$math$` is silently DROPPED in HTML export.** `template.typ` installs
   `#show math.equation: it => html.frame(it)`, which renders every equation to
   inline **SVG**. Author math normally; the rule handles it globally.
   - Trade-off: SVG math is not selectable/searchable text and not screen-reader
     accessible, and each equation adds a few KB. Acceptable for a deep-reference doc.
3. **Video/image `src` are emitted as plain strings** — the media file does *not*
   need to exist at compile time, only at serve time (in `build/assets/`). Reference
   media as `assets/<file>` and put the file in `src/assets/` (build.sh copies it).
4. **Typst is snap-confined** (`/snap/bin/typst`): it cannot read `/tmp` (private
   tmp under confinement). All sources must live under `$HOME` (i.e. the repo). The
   `--root <repo>` flag sets path resolution.
5. **No native multi-page site concept** — nav is generated from `nav-links` in
   `template.typ` and one `.typ` compiles to one `.html`.

## Authoring helpers (from `template.typ`)

- `#show: conf.with(title: "...", current: "rl")` — page wrapper (theme + math rule + nav).
- `#media("assets/x.mp4", label: [...], caption: [...])` — video figure.
- `#fig("assets/x.png", label: [...], caption: [...])` — image figure.
- `#callout([...], tag: "Hard-won", kind: "warn")` — highlighted box (`kind: "note"|"warn"`).
- `#refbox([...])` — references footer.

## Content provenance

- `rl.typ` — synthesized from `docs/rl/` (RESEARCH_STATE.md, reorientation.md,
  policy_switching_literature.md).
- `morphology.typ` — to port from `paper/main.tex` (+ `paper/references.bib`).
- `hardware.typ` — to port from `hand_paper/main.tex`.
