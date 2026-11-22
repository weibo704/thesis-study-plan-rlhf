"""Render the stage-1 flow figure to PNG.

The .drawio file is the editable source and the .svg is the vector version
for print; this produces the raster that goes into the Word report. Drawn
directly with PIL because no SVG rasteriser is available on this machine
(no cairo, and macOS Quick Look silently clips the right-hand side).

    python docs/make_figure.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2                      # supersampling factor
W, H = 1400, 820
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
BLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

AGENT = ("#FDF3E2", "#B06A00", 2)
LEARN = ("#E9ECFA", "#3A4A9F", 2)
FROZEN = ("#E3F1EF", "#0D6158", 1.4)
PLAIN = ("#F5F7F9", "#7D8B99", 1.4)
ACT = ("#E9F5EE", "#0D6158", 1.4)
CLAR = ("#FBEAE5", "#A33A22", 1.4)
REC = ("#EEF1F4", "#7D8B99", 1.4)

im = Image.new("RGB", (W * S, H * S), "white")
d = ImageDraw.Draw(im)
_cache = {}


def font(size: float, bold: bool = False):
    key = (round(size * S), bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(BLD if bold else REG, key[0])
    return _cache[key]


def text(x, y, s, size=11, bold=False, fill="#45505C", anchor="mm"):
    d.text((x * S, y * S), s, font=font(size, bold), fill=fill, anchor=anchor)


def box(x, y, w, h, head, body, pal, hb=True):
    f, st, sw = pal
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=7 * S,
                        fill=f, outline=st, width=max(1, round(sw * S)))
    lines = body.split("|") if body else []
    total = (19 if head else 0) + len(lines) * 15
    cy = y + h / 2 - total / 2 + 7
    if head:
        text(x + w / 2, cy, head, 12.5, hb, "#1A1F26")
        cy += 19
    for ln in lines:
        text(x + w / 2, cy, ln, 11)
        cy += 15


def lane(x, y, w, h, label, col):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=9 * S,
                        outline=col, width=round(1.2 * S))
    text(x + 14, y + 13, label, 11, True, col, anchor="lm")


def dashed(p0, p1, col, wd, dash=6, gap=4):
    (x0, y0), (x1, y1) = p0, p1
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n == 0:
        return
    steps = int(n / (dash + gap)) or 1
    for i in range(steps + 1):
        a = i * (dash + gap) / n
        b = min(a + dash / n, 1.0)
        if a >= 1:
            break
        d.line([(x0 + (x1 - x0) * a) * S, (y0 + (y1 - y0) * a) * S,
                (x0 + (x1 - x0) * b) * S, (y0 + (y1 - y0) * b) * S],
               fill=col, width=wd)


def arrow(pts, col="#5A6572", sw=1.5, dash=False, label=None, lx=0, ly=0,
          anchor="mm"):
    wd = max(1, round(sw * S))
    for a, b in zip(pts, pts[1:]):
        if dash:
            dashed(a, b, col, wd)
        else:
            d.line([a[0] * S, a[1] * S, b[0] * S, b[1] * S], fill=col, width=wd)
    (px, py), (qx, qy) = pts[-2], pts[-1]
    dx, dy = qx - px, qy - py
    n = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / n, dy / n
    k = 5.5
    d.polygon([(qx * S, qy * S),
               ((qx - k * ux + k * 0.55 * uy) * S, (qy - k * uy - k * 0.55 * ux) * S),
               ((qx - k * ux - k * 0.55 * uy) * S, (qy - k * uy + k * 0.55 * ux) * S)],
              fill=col)
    if label:
        text(lx, ly, label, 10.5, True, col, anchor=anchor)


text(40, 30, "How a user query becomes a study plan, and how feedback changes "
     "the next one", 19, True, "#1A1F26", anchor="lm")
text(40, 58, "Two agents cooperate: one writes the candidate plans, the other "
     "decides which feedback is worth learning from.", 12.5, False, "#5A6572",
     anchor="lm")

lane(40, 90, 1320, 170, "PER QUERY  —  runs every time the learner asks for a plan", "#7D8B99")
box(70, 135, 140, 80, "User query", "goal, level,|hours per week", PLAIN, False)
box(260, 135, 150, 80, "Retrieval", "embed query, take|top 25 of 1,899", FROZEN, False)
box(460, 128, 180, 94, "Plan Generator Agent",
    "writes 6 distinct plans|5 courses each, ordered|temperature 0.4", AGENT)
box(690, 128, 170, 94, "Reward Model",
    "scores all 6|learned from feedback|6 plan features", LEARN)
box(910, 135, 160, 80, "Best-of-N",
    "pick top score, or sample|to keep propensities usable", PLAIN, False)
box(1130, 135, 150, 80, "Study plan shown", "ordered course list", FROZEN)

lane(40, 300, 1320, 300, "FEEDBACK LOOP  —  what the learner says changes the "
     "SCORER, not the generator", "#B06A00")
box(90, 470, 170, 95, "Learner model",
    "max level, max hours,|topics to avoid,|beliefs in prose", AGENT)
box(340, 340, 190, 110, "Preference learning",
    "Bradley–Terry fit|never sees a reward value,|only which plan won", LEARN)
box(600, 340, 180, 110, "Accepted feedback",
    "becomes pairwise|comparisons: plan A|preferred over plan B", PLAIN, False)
box(860, 340, 200, 110, "Feedback Agent  (LangGraph)",
    "Is this credible? actionable?|Does it contradict what|"
    "they said before?  temp 0.7", AGENT)
box(1130, 350, 150, 80, "Learner reacts", "thumbs up / down|+ a reason in words",
    PLAIN, False)
box(880, 480, 160, 60, "ACT", "interpret, update|the learner model", ACT)
box(1070, 480, 150, 60, "CLARIFY", "ask one question,|change nothing yet", CLAR)
box(1240, 480, 110, 60, "RECORD", "log it, no|policy change", REC)

lane(40, 630, 1320, 130, "FROZEN  —  read by retrieval and by the reward; built "
     "once, hashed, replayable with no API call", "#0D6158")
box(90, 665, 180, 70, "Course catalog", "1,899 courses|188 skills, 15 families", FROZEN)
box(330, 665, 180, 70, "Prerequisite graph", "254 edges|→ 122 study plans", FROZEN)
box(570, 665, 180, 70, "Simulated learners", "500 personas|goal, level, hours", FROZEN)
box(810, 665, 180, 70, "Reward spec", "frozen + hashed|before any run", FROZEN)
box(1050, 660, 260, 80, "Firewall",
    "the recommender never reads the prerequisite|"
    "graph, the reward spec, or hidden state", CLAR)

for a, b in [(210, 260), (410, 460), (640, 690), (860, 910), (1070, 1130)]:
    arrow([(a, 175), (b, 175)])
arrow([(1205, 215), (1205, 350)])
arrow([(1130, 390), (1060, 390)])
arrow([(960, 450), (960, 480)], dash=True)
arrow([(1060, 400), (1145, 400), (1145, 480)], dash=True)
arrow([(1060, 430), (1295, 430), (1295, 480)], dash=True)
arrow([(880, 510), (690, 510), (690, 450)], "#0D6158", 2.2,
      label="only the ACT branch reaches the model", lx=752, ly=524)
arrow([(960, 540), (960, 588), (175, 588), (175, 565)], "#B06A00", 1.5, dash=True)
arrow([(600, 395), (530, 395)], "#3A4A9F")
arrow([(435, 340), (435, 280), (660, 280), (660, 175), (690, 175)], "#3A4A9F", 2.6,
      label="retrains the scorer", lx=548, ly=270)
arrow([(90, 517), (58, 517), (58, 175), (70, 175)], "#B06A00", 1.8, dash=True,
      label="constrains the next query", lx=66, ly=281, anchor="lm")

text(40, 788, "Legend", 11, True, "#5A6572", anchor="lm")
for lx, pal, label in [(112, AGENT, "LLM agent — makes a judgement"),
                       (380, LEARN, "learned from feedback"),
                       (600, FROZEN, "frozen artifact"),
                       (770, PLAIN, "plain step")]:
    d.rounded_rectangle([lx * S, 781 * S, (lx + 14) * S, 795 * S], radius=3 * S,
                        fill=pal[0], outline=pal[1], width=round(1.6 * S))
    text(lx + 21, 788, label, 11, False, "#5A6572", anchor="lm")
text(900, 788, "dashed = a branch the agent chooses, or a soft constraint", 11,
     False, "#5A6572", anchor="lm")

out = Path(__file__).parent / "stage1-flow.png"
im.resize((W * 2, H * 2), Image.LANCZOS).save(out, dpi=(200, 200))
print(f"wrote {out} ({W*2}x{H*2})")
