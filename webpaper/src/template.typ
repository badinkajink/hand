// ============================================================================
// MorphoHand project site — shared template for Typst HTML export.
//
// Usage in each page (.typ):
//     #import "template.typ": conf, media, fig, callout, refbox
//     #show: conf.with(title: "RL Manipulation", current: "rl")
//     = My heading
//     ...content...
//
// Key facts about this pipeline (learned the hard way — see webpaper/README.md):
//   * Compile with:  typst compile --features html --format html --root <repo> ...
//   * Bare $math$ is DROPPED in HTML export; `conf` installs a show rule that
//     renders every equation to inline SVG via html.frame(). Author math normally.
//   * <video>/<img> srcs are emitted as plain strings — the media file does NOT
//     need to exist at compile time, only at serve time (in build/assets/).
//   * Typst is snap-confined: all source files must live under $HOME (the repo).
// ============================================================================

#let nav-links = (
  ("index.html", "Overview", "index"),
  ("morphology.html", "Morphology & Grasp Opt", "morphology"),
  ("rl.html", "RL Manipulation", "rl"),
  ("hardware.html", "Hardware Validation", "hardware"),
)

#let site-css = "
:root{--maxw:860px;--fg:#1b1b1f;--muted:#5c5f66;--accent:#2a5db0;--accent2:#7a3ea3;--bg:#fff;--soft:#f5f6f8;--line:#e3e5ea;--warn:#b4690e;}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);line-height:1.65;margin:0;font-size:17px;}
nav.top{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;flex-wrap:wrap;align-items:center;}
nav.top .brand{font-weight:700;color:var(--fg);text-decoration:none;margin-right:8px;}
nav.top a{color:var(--muted);text-decoration:none;font-size:15px;}
nav.top a:hover{color:var(--accent);}
nav.top a.active{color:var(--accent);font-weight:600;}
.wrap{max-width:var(--maxw);margin:0 auto;padding:8px 20px 100px;}
h1{font-size:2.1em;line-height:1.2;margin:.6em 0 .3em;letter-spacing:-.01em;}
h2{font-size:1.5em;margin:1.8em 0 .4em;padding-top:.3em;border-top:1px solid var(--line);}
h3{font-size:1.2em;margin:1.4em 0 .3em;color:#2b2b30;}
a{color:var(--accent);}
p{margin:.7em 0;}
svg{vertical-align:middle;max-width:100%;}
figure{margin:1.8em 0;}
figure video,figure img{width:100%;border-radius:10px;box-shadow:0 1px 10px rgba(0,0,0,.10);display:block;}
figcaption{color:var(--muted);font-size:.92em;margin-top:.55em;line-height:1.5;}
figcaption .lbl{color:var(--fg);font-weight:600;}
table{border-collapse:collapse;width:100%;margin:1.3em 0;font-size:.93em;}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;}
th{background:var(--soft);font-weight:600;}
tr:nth-child(even) td{background:#fafbfc;}
code{background:var(--soft);padding:.12em .38em;border-radius:5px;font-size:.88em;font-family:'SF Mono',Menlo,Consolas,monospace;}
pre{background:#1b1b1f;color:#e8e8ea;padding:14px 16px;border-radius:10px;overflow-x:auto;font-size:.84em;line-height:1.5;}
pre code{background:none;padding:0;color:inherit;}
blockquote{border-left:3px solid var(--accent);margin:1em 0;padding:.2em 1.1em;color:var(--muted);background:var(--soft);border-radius:0 8px 8px 0;}
.callout{border:1px solid var(--line);border-left:4px solid var(--accent);background:var(--soft);border-radius:0 10px 10px 0;padding:.7em 1.1em;margin:1.4em 0;}
.callout.warn{border-left-color:var(--warn);}
.callout .tag{font-weight:700;font-size:.78em;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);display:block;margin-bottom:.2em;}
.callout.warn .tag{color:var(--warn);}
.refbox{font-size:.9em;color:var(--muted);border-top:1px solid var(--line);margin-top:3em;padding-top:1em;}
details.deep{border:1px solid var(--line);border-radius:10px;margin:1.3em 0;background:#fcfcfd;overflow:hidden;}
details.deep>summary{cursor:pointer;padding:.7em 1.05em;font-weight:600;list-style:none;display:flex;align-items:center;gap:.55em;user-select:none;background:var(--soft);}
details.deep>summary::-webkit-details-marker{display:none;}
details.deep>summary::before{content:'\\25B8';color:var(--accent);font-size:.9em;transition:transform .15s;display:inline-block;}
details.deep[open]>summary::before{transform:rotate(90deg);}
details.deep>summary .kind{font-size:.72em;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700;margin-left:auto;}
details.deep>.body{padding:.3em 1.15em 1em;}
.fullpaper-btn{position:fixed;right:18px;bottom:18px;z-index:20;background:var(--accent);color:#fff;border:none;border-radius:24px;padding:10px 16px;font-size:14px;font-weight:600;cursor:pointer;box-shadow:0 3px 14px rgba(0,0,0,.22);}
.fullpaper-btn:hover{background:#21498c;}
.hero{padding:1.2em 0 .4em;}
.hero .sub{color:var(--muted);font-size:1.15em;margin-top:-.2em;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:1.6em 0;}
.card{border:1px solid var(--line);border-radius:12px;padding:16px 18px;text-decoration:none;color:var(--fg);transition:box-shadow .15s,transform .15s;display:block;}
.card:hover{box-shadow:0 4px 18px rgba(0,0,0,.10);transform:translateY(-2px);}
.card h3{margin:.1em 0 .3em;color:var(--accent);}
.card p{margin:0;color:var(--muted);font-size:.92em;}
"

// ---- navbar -----------------------------------------------------------------
#let navbar(current) = html.elem("nav", attrs: (class: "top"), {
  html.elem("a", attrs: (href: "index.html", class: "brand"), "MorphoHand")
  for (href, label, key) in nav-links {
    let cls = if key == current { "active" } else { "" }
    html.elem("a", attrs: (href: href, class: cls), label)
  }
})

// ---- page wrapper / config --------------------------------------------------
// Installs the math->SVG show rule, the stylesheet, the navbar, and the
// max-width content column. Apply with `#show: conf.with(...)`.
// JS: a floating button that opens/closes every <details.deep> on the page —
// the "full paper experience" toggle. Collapsed is the default; one click
// expands everything for continuous reading.
#let fullpaper-js = "
(function(){
  function ds(){return Array.from(document.querySelectorAll('details.deep'));}
  var b=document.createElement('button');b.className='fullpaper-btn';
  function sync(){var anyClosed=ds().some(function(d){return !d.open});
    b.textContent=anyClosed?'\u{1F4D6} Full paper view':'\u{2715} Collapse all';}
  b.addEventListener('click',function(){var anyClosed=ds().some(function(d){return !d.open});
    ds().forEach(function(d){d.open=anyClosed});sync();});
  document.addEventListener('DOMContentLoaded',function(){if(ds().length){document.body.appendChild(b);sync();}});
})();
"

#let conf(title: "MorphoHand", current: "index", doc) = {
  set document(title: title)
  show math.equation: it => html.frame(it)   // <-- the load-bearing line
  html.elem("style", site-css)
  navbar(current)
  html.elem("div", attrs: (class: "wrap"), doc)
  html.elem("script", fullpaper-js)
}

// A collapsible deep-dive block (collapsed by default). `kind` is a small
// right-aligned tag, e.g. "derivation", "related work", "extended results".
#let det(summary, body, kind: none, open: false) = {
  let sattrs = (class: "deep")
  if open { sattrs.insert("open", "") }
  html.elem("details", attrs: sattrs, {
    html.elem("summary", {
      summary
      if kind != none { html.elem("span", attrs: (class: "kind"), kind) }
    })
    html.elem("div", attrs: (class: "body"), body)
  })
}

// ---- media helpers ----------------------------------------------------------
// A video figure. `src` is relative to the page (e.g. "assets/foo.mp4").
#let media(src, label: none, caption: none, poster: none, loop: false) = {
  let attrs = (src: src, controls: "", playsinline: "")
  if poster != none { attrs.insert("poster", poster) }
  if loop { attrs.insert("loop", ""); attrs.insert("muted", "") }
  html.elem("figure", {
    html.elem("video", attrs: attrs)[]
    if caption != none or label != none {
      html.elem("figcaption", {
        if label != none { html.elem("span", attrs: (class: "lbl"), label); [ ] }
        if caption != none { caption }
      })
    }
  })
}

// A static image figure.
#let fig(src, label: none, caption: none) = html.elem("figure", {
  html.elem("img", attrs: (src: src, loading: "lazy"))[]
  if caption != none or label != none {
    html.elem("figcaption", {
      if label != none { html.elem("span", attrs: (class: "lbl"), label); [ ] }
      if caption != none { caption }
    })
  }
})

// A callout box. kind: "note" (accent) or "warn".
#let callout(body, tag: "Note", kind: "note") = html.elem(
  "div", attrs: (class: if kind == "warn" { "callout warn" } else { "callout" }),
  { html.elem("span", attrs: (class: "tag"), tag); body },
)

// A references footer.
#let refbox(body) = html.elem("div", attrs: (class: "refbox"), body)
