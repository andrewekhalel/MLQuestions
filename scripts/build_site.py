#!/usr/bin/env python3
"""Generate the static site for GitHub Pages from README.md and NLP/README.md.

The READMEs stay the single source of truth: every topic page, the hub page,
sitemap.xml and robots.txt are derived from them at build time.

    pip install markdown
    python3 scripts/build_site.py
    python3 -m http.server -d _site 8000

Override the canonical origin when using a custom domain:

    SITE_URL=https://mlquestions.dev python3 scripts/build_site.py
"""
import html
import json
import os
import pathlib
import re
import shutil
import xml.sax.saxutils as sx

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"
SITE_URL = (os.environ.get("SITE_URL")
            or "https://andrewekhalel.github.io/MLQuestions").rstrip("/")
REPO_URL = "https://github.com/andrewekhalel/MLQuestions"
YEAR = "2026"
AFFILIATE = ("tap_s=", "amzn.to", "tryexponent.com")

# section heading in README.md -> (url slug, <h1>/<title> topic, meta description)
TOPICS = {
    "Machine Learning Fundamentals": (
        "machine-learning-fundamentals-interview-questions",
        "Machine Learning Fundamentals",
        "Core machine learning interview questions on the bias-variance trade-off, "
        "overfitting, regularization, and supervised vs unsupervised learning."),
    "Algorithms and Ensembles": (
        "machine-learning-algorithms-interview-questions",
        "Ensemble Methods",
        "Interview questions on ensemble methods, bagging and boosting, and why "
        "ensembles outperform individual models."),
    "Data Preprocessing and Feature Engineering": (
        "data-preprocessing-interview-questions",
        "Data Preprocessing",
        "Interview questions on data normalization, imbalanced datasets, data "
        "augmentation, and label vs one-hot encoding."),
    "Optimization and Training": (
        "optimization-interview-questions",
        "Optimization",
        "Interview questions on gradient descent, learning rate, momentum, "
        "batch vs stochastic gradient descent, and epochs vs iterations."),
    "Deep Learning and Neural Networks": (
        "deep-learning-interview-questions",
        "Deep Learning",
        "Deep learning interview questions on ReLU, batch normalization, dropout, "
        "vanishing gradients, ResNets, LSTMs, transformers, autoencoders and GANs."),
    "Computer Vision": (
        "computer-vision-interview-questions",
        "Computer Vision",
        "Computer vision interview questions on convolutions, receptive fields, "
        "max-pooling, segmentation, non-maximum suppression, RANSAC and RCNNs."),
    "Dimensionality Reduction": (
        "dimensionality-reduction-interview-questions",
        "Dimensionality Reduction",
        "Interview questions on PCA, LDA, t-SNE, UMAP and the curse of "
        "dimensionality, including how these techniques compare."),
    "Model Evaluation and Validation": (
        "model-evaluation-interview-questions",
        "Model Evaluation",
        "Interview questions on train/validation/test splits, stratified "
        "cross-validation, precision, recall, F1-score and ROC curves."),
    "Probability and Statistics": (
        "statistics-interview-questions",
        "Statistics for ML",
        "Statistics interview questions for machine learning on Type I vs Type II "
        "error, random number generation, and Bayesian vs frequentist inference."),
    "Coding and Implementation": (
        "machine-learning-coding-interview-questions",
        "ML Coding",
        "Coding interview questions for ML engineers: connected components, sparse "
        "matrices, integral images, square roots, bitstrings and linked lists."),
}
NLP_TOPIC = ("nlp-interview-questions", "NLP",
             "NLP interview questions and answers on stemming vs lemmatization, "
             "dependency parsing, n-grams, word embeddings, perplexity and BLEU.")

MD = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists"])


def render(md_text):
    MD.reset()
    out = MD.convert(md_text)
    # affiliate and referral links must be disclosed to crawlers
    def rel(m):
        href = m.group(1)
        if not href.startswith("http"):
            return m.group(0)
        r = ("sponsored nofollow noopener" if any(a in href for a in AFFILIATE)
             else "noopener")
        return f'<a href="{href}" rel="{r}"'
    return re.sub(r'<a href="([^"]+)"', rel, out)


def text_of(html_frag):
    t = re.sub(r"<(script|style).*?</\1>", " ", html_frag, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def parse(md_path, flat_section=None):
    """-> [(section_name, [(question, answer_md), ...]), ...]"""
    lines = md_path.read_text().splitlines()
    sections, cur_sec, cur_q, buf = [], flat_section, None, []

    def flush():
        if cur_q is not None:
            sections[-1][1].append((cur_q, "\n".join(buf).strip()))

    if flat_section:
        sections.append((flat_section, []))
    for line in lines:
        h2 = re.match(r"^##\s+(?!#)(.*)$", line)
        h3 = re.match(r"^###\s+(?!#)(.*)$", line)
        if h2 and not flat_section:
            flush()
            cur_q, buf = None, []
            cur_sec = h2.group(1).strip()
            sections.append((cur_sec, []))
        elif h3 and cur_sec is not None:
            flush()
            cur_q = re.sub(r"^\d+[.)]\s*", "", h3.group(1).strip())
            buf = []
        elif cur_q is not None:
            buf.append(line)
    flush()
    return [(n, qs) for n, qs in sections if qs]


CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{--bg:#fff;--fg:#1a1a1a;--mut:#5c6370;--acc:#0b5fff;--line:#e6e8eb;--code:#f5f6f8}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#9aa3af;
--acc:#79a8ff;--line:#262b33;--code:#171a20}}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:52rem;margin:0 auto;padding:1.5rem 1.15rem 4rem}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:1.95rem;line-height:1.25;margin:.6rem 0 .5rem;letter-spacing:-.02em}
h2{font-size:1.3rem;margin:2.4rem 0 .6rem;padding-bottom:.3rem;
border-bottom:1px solid var(--line)}
h3{font-size:1.09rem;margin:2rem 0 .5rem}
p,li{overflow-wrap:break-word}
nav.bc{font-size:.85rem;color:var(--mut)}nav.bc a{color:var(--mut)}
.lede{font-size:1.06rem;color:var(--mut);margin:0 0 1.2rem}
.qa{border-top:1px solid var(--line);padding-top:.2rem}
.qa h3{scroll-margin-top:1rem}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
code{background:var(--code);padding:.15em .38em;border-radius:4px}
pre{background:var(--code);padding:.9rem 1rem;border-radius:8px;overflow-x:auto}
pre code{background:none;padding:0}
ul.toc{list-style:none;padding:0}ul.toc li{margin:.42rem 0}
ul.toc .n{color:var(--mut);font-size:.85rem}
.cards{display:grid;gap:.7rem;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));
padding:0;list-style:none}
.cards a{display:block;border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem;
color:inherit;height:100%}
.cards a:hover{border-color:var(--acc);text-decoration:none}
.cards strong{display:block;color:var(--acc)}.cards span{color:var(--mut);font-size:.85rem}
.pager{display:flex;justify-content:space-between;gap:1rem;margin-top:3rem;
border-top:1px solid var(--line);padding-top:1rem;font-size:.92rem}
footer{margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem;
color:var(--mut);font-size:.88rem}
"""


def page(*, slug, title, desc, h1, body, jsonld, crumb=None):
    url = f"{SITE_URL}/" if not slug else f"{SITE_URL}/{slug}/"
    bc = ""
    if crumb:
        bc = (f'<nav class="bc"><a href="{SITE_URL}/">'
              f'ML Interview Questions</a> / {html.escape(crumb)}</nav>')
    scripts = "\n".join(
        f'<script type="application/ld+json">{json.dumps(b, ensure_ascii=False)}</script>'
        for b in jsonld)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:url" content="{url}">
<meta name="twitter:card" content="summary">
<style>{CSS}</style>
{scripts}
</head>
<body><div class="wrap">
{bc}
<h1>{h1}</h1>
{body}
<footer>
<p>Open source on <a href="{REPO_URL}">GitHub</a> — contributions welcome.
Star the repo to support it.</p>
</footer>
</div></body>
</html>
"""
    dest = OUT / slug / "index.html" if slug else OUT / "index.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(doc)
    return url


def faq_ld(pairs):
    return {"@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": text_of(render(q)),
                            "acceptedAnswer": {"@type": "Answer",
                                               "text": text_of(render(a))[:1200]}}
                           for q, a in pairs if a.strip()]}


def crumb_ld(topic, url):
    return {"@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": 1,
                 "name": "Machine Learning Interview Questions",
                 "item": f"{SITE_URL}/"},
                {"@type": "ListItem", "position": 2, "name": topic, "item": url}]}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    pages = []  # (slug, topic, desc, pairs)
    for name, pairs in parse(ROOT / "README.md"):
        if name in TOPICS:
            slug, topic, desc = TOPICS[name]
            pages.append((slug, topic, desc, pairs))
    nlp = parse(ROOT / "NLP" / "README.md", flat_section="NLP")
    if nlp:
        pages.append((*NLP_TOPIC, nlp[0][1]))
    assert pages, "no topic sections parsed — did the README headings change?"

    total = sum(len(p[3]) for p in pages)
    urls = []

    for i, (slug, topic, desc, pairs) in enumerate(pages):
        prev_p = pages[i - 1] if i else None
        next_p = pages[i + 1] if i + 1 < len(pages) else None
        title = f"{topic} Interview Questions and Answers ({YEAR})"
        qa = "".join(
            f'<div class="qa"><h3 id="{sl}">{render(q).strip()[3:-4]}</h3>\n{render(a)}</div>\n'
            for q, a, sl in ((q, a, slugify(q)) for q, a in pairs))
        toc = "".join(f'<li><a href="#{slugify(q)}">{render(q).strip()[3:-4]}</a></li>'
                      for q, _ in pairs)
        pager = '<div class="pager">' + (
            f'<a href="{SITE_URL}/{prev_p[0]}/">&larr; {html.escape(prev_p[1])}</a>'
            if prev_p else "<span></span>") + (
            f'<a href="{SITE_URL}/{next_p[0]}/">{html.escape(next_p[1])} &rarr;</a>'
            if next_p else "<span></span>") + "</div>"
        body = (f'<p class="lede">{html.escape(desc)}</p>'
                f"<h2>{len(pairs)} questions on this page</h2>"
                f'<ul class="toc">{toc}</ul>{qa}{pager}')
        url = page(slug=slug, title=title, desc=desc, h1=html.escape(title),
                   body=body, crumb=topic,
                   jsonld=[faq_ld(pairs), crumb_ld(topic, f"{SITE_URL}/{slug}/")])
        urls.append(url)

    # --- hub page: unique content (every question title), answers live on topics
    cards = "".join(
        f'<li><a href="{SITE_URL}/{s}/"><strong>{html.escape(t)}</strong>'
        f"<span>{len(p)} questions</span></a></li>"
        for s, t, _, p in pages)
    listing = "".join(
        f'<h2><a href="{SITE_URL}/{s}/">{html.escape(t)}</a></h2><ul class="toc">'
        + "".join(f'<li><a href="{SITE_URL}/{s}/#{slugify(q)}">'
                  f"{render(q).strip()[3:-4]}</a></li>" for q, _ in p)
        + "</ul>" for s, t, _, p in pages)
    hub_title = f"Machine Learning Interview Questions and Answers ({YEAR})"
    hub_desc = (f"{total} machine learning interview questions with answers: ML basics, "
                "deep learning, computer vision, NLP, statistics and coding. "
                "Free and open source.")
    hub_ld = {"@context": "https://schema.org", "@type": "CollectionPage",
              "name": hub_title, "description": hub_desc, "url": f"{SITE_URL}/",
              "inLanguage": "en",
              "isPartOf": {"@type": "WebSite", "name": hub_title,
                           "url": f"{SITE_URL}/"}}
    urls.insert(0, page(
        slug="", title=hub_title, desc=hub_desc, h1=html.escape(hub_title),
        jsonld=[hub_ld],
        body=(f'<p class="lede"><strong>{total} machine learning interview questions '
              "with answers</strong>, grouped by topic and maintained as an open source "
              f"project since 2018. Free, no signup.</p>"
              f'<ul class="cards">{cards}</ul>'
              "<h2>Every question, by topic</h2>" + listing)))

    (OUT / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"<url><loc>{sx.escape(u)}</loc></url>\n" for u in urls)
        + "</urlset>\n")
    (OUT / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n")
    (OUT / ".nojekyll").write_text("")

    print(f"{len(urls)} pages, {total} questions -> {OUT}")


def slugify(text):
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    s = re.sub(r"[`*_]", "", s).lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    return re.sub(r"\s+", "-", s.strip())[:80].strip("-")


if __name__ == "__main__":
    main()
