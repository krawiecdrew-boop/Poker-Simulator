"""
Poker Equity Lab (Streamlit)

Install:  pip install streamlit pandas jinja2
Run:      python3 -m streamlit run poker_sim.py

Section 1: compare hands against hands / ranges (Monte Carlo).
Section 2: EV matrix - how every starting hand (all 169) does against one
           hand or a range you build (grid + top-% slider), judged against your pot odds.
"""
import random
import re

import pandas as pd
import streamlit as st

# =====================================================================
# ENGINE
# Cards are ints 0..51: rank = c >> 2 (0 = deuce ... 12 = ace), suit = c & 3
# =====================================================================
RANKS = "23456789TJQKA"
GRID = RANKS[::-1]          # grid order: A K Q ... 2
SUITS = "shdc"
SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def card_pretty(c):
    return RANKS[c >> 2] + SYM[SUITS[c & 3]]


def parse_cards(text):
    """'As Kd 7c' or 'AsKd7c' -> [48, 41, ...]. Raises ValueError on bad input."""
    found = re.findall(r"[2-9TJQKAtjqka][shdcSHDC]", text)
    if len("".join(text.split()).replace(",", "")) != 2 * len(found):
        raise ValueError(f"Couldn't read cards in '{text}'. Use e.g. 'As Kd'.")
    cards = [RANKS.index(r.upper()) * 4 + SUITS.index(s.lower()) for r, s in found]
    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate card entered.")
    return cards


def evaluate(cards):
    """Score the best 5-card hand from 5-7 cards. Higher score wins."""
    cnt = [0] * 13
    sc = [0] * 4
    sm = [0] * 4
    mask = 0
    for c in cards:
        r, s = c >> 2, c & 3
        cnt[r] += 1
        sc[s] += 1
        sm[s] |= 1 << r
        mask |= 1 << r

    def straight_high(m):
        for h in range(12, 3, -1):
            if (m >> (h - 4)) & 31 == 31:
                return h
        return 3 if (m & 0x100F) == 0x100F else -1   # wheel A-2-3-4-5

    def pack(cat, arr):
        v = cat
        for i in range(5):
            v = v * 13 + (arr[i] if i < len(arr) else 0)
        return v

    def kick(excl, n):
        return [r for r in range(12, -1, -1) if cnt[r] and r not in excl][:n]

    flush_suit = next((s for s in range(4) if sc[s] >= 5), -1)
    if flush_suit >= 0:
        h = straight_high(sm[flush_suit])
        if h >= 0:
            return pack(8, [h])                                  # straight flush

    quads = [r for r in range(12, -1, -1) if cnt[r] == 4]
    trips = [r for r in range(12, -1, -1) if cnt[r] == 3]
    pairs = [r for r in range(12, -1, -1) if cnt[r] == 2]

    if quads:
        return pack(7, [quads[0]] + kick([quads[0]], 1))
    if trips and (len(trips) > 1 or pairs):
        return pack(6, [trips[0], trips[1] if len(trips) > 1 else pairs[0]])
    if flush_suit >= 0:
        top = [r for r in range(12, -1, -1) if (sm[flush_suit] >> r) & 1][:5]
        return pack(5, top)
    h = straight_high(mask)
    if h >= 0:
        return pack(4, [h])
    if trips:
        return pack(3, [trips[0]] + kick([trips[0]], 2))
    if len(pairs) > 1:
        return pack(2, [pairs[0], pairs[1]] + kick([pairs[0], pairs[1]], 1))
    if pairs:
        return pack(1, [pairs[0]] + kick([pairs[0]], 3))
    return pack(0, kick([], 5))


def grid_label(i, j):
    """Label for grid cell (row i, col j): pairs on the diagonal, suited above, offsuit below."""
    if i == j:
        return GRID[i] * 2
    if i < j:
        return GRID[i] + GRID[j] + "s"
    return GRID[j] + GRID[i] + "o"


def _add_label(out, a, b, suffix):
    if a == b:
        out.add(a + a)
    elif suffix:
        out.add(a + b + suffix)
    else:
        out.add(a + b + "s")
        out.add(a + b + "o")


def parse_range(text):
    """Parse 'TT+, AQs+, AKo, 99-66, A5s-A2s' into a set of labels like {'AA', 'AKs'}."""
    out = set()
    k = "([AKQJT2-9])"
    gi = GRID.index
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        tok = tok.upper().replace("S", "s").replace("O", "o")
        m = re.fullmatch(f"{k}{k}([so]?)(\\+?)", tok)
        if m:
            a, b, suf, plus = m.groups()
            if gi(a) > gi(b):
                a, b = b, a
            if a == b:
                stop = 0 if plus else gi(a)
                for i in range(gi(a), stop - 1, -1):
                    _add_label(out, GRID[i], GRID[i], "")
            else:
                stop = gi(a) + 1 if plus else gi(b)
                for j in range(gi(b), stop - 1, -1):
                    _add_label(out, a, GRID[j], suf)
            continue
        m = re.fullmatch(f"{k}{k}([so]?)-{k}{k}([so]?)", tok)
        if m:
            a, b, suf, c, d, _ = m.groups()
            if a == b and c == d:
                for i in range(min(gi(a), gi(c)), max(gi(a), gi(c)) + 1):
                    _add_label(out, GRID[i], GRID[i], "")
            else:
                for j in range(min(gi(b), gi(d)), max(gi(b), gi(d)) + 1):
                    _add_label(out, a, GRID[j], suf)
    return out


def range_combos(labels):
    """Expand labels into every specific two-card combo."""
    combos = []
    for lab in labels:
        ra, rb = RANKS.index(lab[0]), RANKS.index(lab[1])
        for s1 in range(4):
            for s2 in range(4):
                if len(lab) == 2:
                    ok = s1 < s2
                elif lab[2] == "s":
                    ok = s1 == s2
                else:
                    ok = s1 != s2
                if ok:
                    combos.append((ra * 4 + s1, rb * 4 + s2))
    return combos


def simulate(players, board, n_sims, progress=None):
    """
    players: list of {'hand': [c1, c2]} or {'range': set_of_labels}
    board:   list of 0-5 known cards
    Returns per-player win / tie / equity fractions.
    """
    dead = set(board)
    for p in players:
        if "hand" in p:
            dead.update(p["hand"])
    combos = []
    for p in players:
        if "range" in p:
            cb = [c for c in range_combos(p["range"]) if c[0] not in dead and c[1] not in dead]
            if not cb:
                raise ValueError("A range has no combos left after removing cards already in use.")
            combos.append(cb)
        else:
            combos.append(None)

    free = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    n = len(players)
    wins, ties, eq = [0] * n, [0] * n, [0.0] * n
    valid = 0

    for it in range(n_sims):
        used = set(dead)
        hands = []
        ok = True
        for p, cb in zip(players, combos):
            if cb is None:
                hands.append(p["hand"])
                continue
            for _ in range(30):
                h = random.choice(cb)
                if h[0] not in used and h[1] not in used:
                    used.update(h)
                    hands.append(h)
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        avail = [c for c in free if c not in used]
        full_board = list(board) + random.sample(avail, need)

        best, winners = -1, []
        for i, h in enumerate(hands):
            s = evaluate(list(h) + full_board)
            if s > best:
                best, winners = s, [i]
            elif s == best:
                winners.append(i)
        valid += 1
        share = 1 / len(winners)
        for i in winners:
            eq[i] += share
            if len(winners) == 1:
                wins[i] += 1
            else:
                ties[i] += 1

        if progress and it % 2000 == 0:
            progress(it / n_sims)

    valid = max(valid, 1)
    return {
        "win": [w / valid for w in wins],
        "tie": [t / valid for t in ties],
        "equity": [e / valid for e in eq],
        "hands": valid,
    }


def equity_grid(opp, board, sims_per_hand, progress=None):
    """
    Equity of every one of the 169 starting-hand types against one opponent.

    opp: {'random': True}          -> a completely random hand
         {'hand': [c1, c2]}        -> one specific hand
         {'range': set_of_labels}  -> a random hand drawn from the range
    board: 0-5 known cards (shared by everyone)

    Returns {label: equity or None}. None means no combo of that hand is possible
    (e.g. every combo is blocked by the board / opponent's cards).
    """
    dead = set(board)
    if "hand" in opp:
        dead.update(opp["hand"])
    opp_cb = None
    if "range" in opp:
        opp_cb = [c for c in range_combos(opp["range"]) if c[0] not in dead and c[1] not in dead]
        if not opp_cb:
            raise ValueError("The opponent range has no combos left after removing cards already in use.")

    free = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    board = list(board)

    labels = [grid_label(i, j) for i in range(13) for j in range(13)]
    out = {}
    for n, lab in enumerate(labels):
        hero_cb = [c for c in range_combos({lab}) if c[0] not in dead and c[1] not in dead]
        if not hero_cb:
            out[lab] = None
            continue
        eq, valid = 0.0, 0
        for _ in range(sims_per_hand):
            h = random.choice(hero_cb)
            avail = [c for c in free if c != h[0] and c != h[1]]

            if "hand" in opp:
                o = opp["hand"]
                draw = random.sample(avail, need)
            elif opp_cb is not None:
                for _try in range(30):
                    o = random.choice(opp_cb)
                    if o[0] not in h and o[1] not in h:
                        break
                else:
                    continue
                avail = [c for c in avail if c != o[0] and c != o[1]]
                draw = random.sample(avail, need)
            else:
                d = random.sample(avail, 2 + need)
                o, draw = d[:2], d[2:]

            fb = board + draw
            hs = evaluate([h[0], h[1]] + fb)
            os_ = evaluate([o[0], o[1]] + fb)
            valid += 1
            if hs > os_:
                eq += 1
            elif hs == os_:
                eq += 0.5
        out[lab] = eq / valid if valid else None
        if progress:
            progress((n + 1) / len(labels))
    return out


# =====================================================================
# UI
# =====================================================================
PRESETS = {
    "Premium": "TT+,AQs+,AKo",
    "Pairs": "22+",
    "Broadway": "ATs+,KTs+,QTs+,JTs,ATo+,KTo+,QTo+,JTo",
    "Suited aces": "A2s+",
    "Clear": "",
}
COLORS = ["#2fb36d", "#4a93ee", "#e8649c", "#e5a526"]


# All 169 starting hands, strongest first (ranked by equity vs a random hand).
RANKED = (
    "AA KK QQ JJ TT 99 88 AKs 77 AQs AKo AJs ATs AQo AJo KQs 66 ATo A9s KJs KTs KQo A8s A7s A9o 55 KJo QJs K9s A8o A6s KTo A5s A4s QTs A7o K8s A3s QJo JTs A5o A2s A6o K9o 44 Q9s QTo K7s K6s A4o Q8s K8o J9s A3o K5s K7o K4s Q9o JTo A2o K3s T9s Q7s K6o J8s Q6s J9o Q8o 33 K5o K2s Q5s J7s K4o T8s Q4s Q7o J8o T9o Q6o Q3s 98s K3o J6s T7s 22 K2o Q5o J4s J7o Q2s J5s T8o Q4o 97s T6s J6o 98o 87s J3s Q3o T7o J2s Q2o 96s T5s J5o T4s J4o 86s T6o 97o T3s 95s 87o J3o 76s T2s T5o J2o 85s 96o 94s T4o 75s 86o 93s 84s 65s T3o 92s 95o 76o 64s 74s T2o 54s 85o 83s 94o 82s 75o 93o 73s 53s 84o 65o 63s 92o 74o 43s 54o 64o 72s 52s 83o 62s 42s 82o 73o 53o 63o 32s 43o 72o 62o 52o 42o 32o"
).split()


def top_pct_range(pct):
    """Strongest hands making up the top `pct` percent (0-100) of all 1326 combos."""
    target = pct / 100 * 1326
    out, cum = set(), 0
    for lab in RANKED:
        n = len(range_combos({lab}))
        if cum + n / 2 <= target:
            out.add(lab)
        cum += n
    return out


def range_to_df(labels):
    data = [[grid_label(i, j) in labels for j in range(13)] for i in range(13)]
    return pd.DataFrame(data, index=list(GRID), columns=list(GRID))


def df_to_range(df):
    return {grid_label(i, j) for i in range(13) for j in range(13) if bool(df.iloc[i, j])}


def set_range(i, labels):
    st.session_state[f"range_{i}"] = labels
    st.session_state[f"ver_{i}"] += 1      # new key resets the grid editor


def apply_text(i):
    set_range(i, parse_range(st.session_state[f"txt_{i}"]))


def set_ev_range(labels):
    st.session_state["ev_range"] = labels
    st.session_state["ev_ver"] += 1        # new key resets the grid editor


def apply_ev_text():
    set_ev_range(parse_range(st.session_state["ev_txt"]))


def apply_ev_slider():
    set_ev_range(top_pct_range(st.session_state["ev_pct"]))


def _heat_colors(diff_df):
    """Green when equity beats break-even, red when it doesn't; stronger = further away."""
    def css(d):
        if pd.isna(d):
            return "color: #888"
        a = min(abs(d) / 0.25, 1.0) * 0.75
        rgb = "47,179,109" if d >= 0 else "232,100,108"
        return f"background-color: rgba({rgb},{a:.2f})"
    return diff_df.map(css) if hasattr(diff_df, "map") else diff_df.applymap(css)


def ev_section():
    st.divider()
    st.header("EV Calculator")
    st.caption("Shows how all starting hands do against one hand or a range for a given scenario, "
               "and whether calling is profitable given certain pot odds")

    if "ev_range" not in st.session_state:
        st.session_state["ev_pct"] = 15
        st.session_state["ev_range"] = top_pct_range(15)
        st.session_state["ev_ver"] = 0

    a, b = st.columns([2, 1])
    mode = a.radio("Opponent holds", ["Hand", "Range"], horizontal=True, key="ev_mode")
    ev_board_text = b.text_input("Board (optional)", key="ev_board", placeholder="e.g. 3h 4d 5c",
                                 help="0, 3, 4 or 5 cards")

    errors, opp, opp_key = [], {}, ()
    if mode == "Hand":
        t = st.text_input("Opponent's hole cards", value="As Ad", key="ev_opp_hand",
                          placeholder="e.g. As Ad")
        try:
            h = parse_cards(t)
            if len(h) != 2:
                raise ValueError("Enter exactly two cards for the opponent.")
            opp, opp_key = {"hand": h}, ("hand", tuple(h))
            st.caption(" ".join(card_pretty(c) for c in h))
        except ValueError as e:
            errors.append(str(e))
    else:
        st.slider("Top % of hands", 0, 100, key="ev_pct", on_change=apply_ev_slider,
                  help="Fills the grid with the strongest X% of hands (ranked by equity vs a "
                       "random hand). 0% = nothing, 100% = every hand. Then fine-tune cell by cell.")
        pcols = st.columns(len(PRESETS))
        for (name, text), pc in zip(PRESETS.items(), pcols):
            pc.button(name, key=f"ev_pre_{name}", on_click=set_ev_range,
                      args=(parse_range(text),), use_container_width=True)
        tc, bc = st.columns([5, 1])
        tc.text_input("Range text", key="ev_txt", placeholder="e.g. TT+,AQs+,AKo",
                      label_visibility="collapsed")
        bc.button("Apply text", key="ev_apply", on_click=apply_ev_text, use_container_width=True)
        edited = st.data_editor(range_to_df(st.session_state["ev_range"]),
                                key=f"ev_grid_{st.session_state['ev_ver']}",
                                use_container_width=True, height=490)
        labels = df_to_range(edited)
        st.session_state["ev_range"] = labels
        n = len(range_combos(labels))
        st.caption(f"{n} combos ({n / 13.26:.1f}% of hands). "
                   "Diagonal = pairs, above = suited, below = offsuit.")
        if labels:
            opp, opp_key = {"range": labels}, ("range", tuple(sorted(labels)))
        else:
            errors.append("Opponent range is empty.")

    board = []
    try:
        board = parse_cards(ev_board_text) if ev_board_text.strip() else []
        if len(board) not in (0, 3, 4, 5):
            errors.append("Board must have 0, 3, 4 or 5 cards.")
    except ValueError as e:
        errors.append(str(e))
    if not errors and "hand" in opp and set(opp["hand"]) & set(board):
        errors.append("The same card is used more than once.")

    p1, p2, p3 = st.columns(3)
    call_amt = p1.number_input("Amount to call", min_value=0.0, value=10.0, step=1.0, key="ev_call")
    win_amt = p2.number_input("Amount you'd win (pot before your call)", min_value=0.01,
                              value=10.0, step=1.0, key="ev_win")
    sims = p3.select_slider("Simulations per hand", [200, 500, 1000, 2000, 5000, 10_000], value=500, key="ev_sims",
                            help="169 hands are simulated, so total work = 169 x this number.")

    for e in errors:
        st.warning(e)

    sig = (opp_key, tuple(board), sims)
    if st.button("Calculate matrix", type="primary", disabled=bool(errors), key="ev_run"):
        bar = st.progress(0.0, text="Simulating all 169 hands...")
        try:
            res = equity_grid(opp, board, sims, lambda f: bar.progress(f, text="Simulating all 169 hands..."))
            st.session_state["ev_result"] = {"sig": sig, "eq": res}
        except ValueError as e:
            st.error(str(e))
        bar.empty()

    data = st.session_state.get("ev_result")
    if errors or not data or data["sig"] != sig:
        if not errors:
            st.info("Pick an opponent, set your pot odds, and click **Calculate matrix**.")
        return

    eq = data["eq"]
    breakeven = call_amt / (call_amt + win_amt)

    def ev_of(e):
        return e * (win_amt + call_amt) - call_amt

    eq_df = pd.DataFrame([[eq[grid_label(i, j)] for j in range(13)] for i in range(13)],
                         index=list(GRID), columns=list(GRID), dtype=float)
    diff = eq_df - breakeven

    ok_labels = [l for l, e in eq.items() if e is not None and e > breakeven]
    ok_combos = sum(len(range_combos({l})) for l in ok_labels)
    m1, m2, m3 = st.columns(3)
    m1.metric("Break-even equity", f"{breakeven * 100:.1f}%",
              help=f"Calling {call_amt:g} to win {win_amt:g}")
    m2.metric("Profitable hand types", f"{len(ok_labels)} / 169")
    m3.metric("Profitable combos", f"{ok_combos / 13.26:.1f}% of all hands")

    view = st.radio("Show", ["Equity %", "EV (chips)", "Sorted list"], horizontal=True, key="ev_view")

    if view == "Equity %":
        st.dataframe((eq_df * 100).style.apply(lambda _: _heat_colors(diff), axis=None)
                     .format("{:.1f}", na_rep="–"), use_container_width=True, height=490)
    elif view == "EV (chips)":
        ev_df = eq_df.map(lambda e: ev_of(e) if pd.notna(e) else float("nan")) \
            if hasattr(eq_df, "map") else eq_df.applymap(lambda e: ev_of(e) if pd.notna(e) else float("nan"))
        st.dataframe(ev_df.style.apply(lambda _: _heat_colors(diff), axis=None)
                     .format("{:+.2f}", na_rep="–"), use_container_width=True, height=490)
        ex = breakeven + 0.01
        st.caption(f"Average chips won or lost per call. Example: a hand with {ex * 100:.0f}% equity "
                   f"calling {call_amt:g} to win {win_amt:g} = {ex:.2f} x {win_amt:g} - "
                   f"{1 - ex:.2f} x {call_amt:g} = {ev_of(ex):+.2f} chips.")
    else:
        rows = [{"Hand": l, "Equity %": round(e * 100, 1), "EV (in chips)": round(ev_of(e), 2)}
                for l, e in eq.items() if e is not None]
        st.dataframe(pd.DataFrame(rows).sort_values("Equity %", ascending=False)
                     .reset_index(drop=True), use_container_width=True, height=490)

    st.caption("Grid layout: diagonal = pairs, above = suited, below = offsuit. Green = equity beats "
               "the break-even needed for your pot odds, red = it doesn't. "
               )


def main():
    st.set_page_config(page_title="Poker Equity Lab", page_icon="♠️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.title("♠️ Poker Equity Lab")
    st.caption("Compare hands, ranges, or a hand against a range.")

    for i in range(4):
        st.session_state.setdefault(f"range_{i}", parse_range("TT+,AQs+,AKo") if i == 1 else set())
        st.session_state.setdefault(f"ver_{i}", 0)

    st.header("Equity Comparison")
    sc1, sc2, sc3 = st.columns(3)
    n_players = sc1.slider("Players", 2, 4, 2)
    n_sims = sc2.select_slider("Simulations", [5_000, 20_000, 50_000, 100_000], value=20_000)
    board_text = sc3.text_input("Board (optional)", placeholder="e.g. Ah 7d 2c", help="0, 3, 4 or 5 cards")
    st.caption("Cards: rank (2-9 T J Q K A) + suit (s h d c). Example: `As Kd`")
    st.write("")

    players, errors = [], []
    cols = st.columns(n_players)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"<h3 style='color:{COLORS[i]};margin-bottom:0'>Player {i + 1}</h3>", unsafe_allow_html=True)
            mode = st.radio("Type", ["Hand", "Range"], key=f"mode_{i}", horizontal=True,
                            index=1 if i == 1 else 0, label_visibility="collapsed")
            if mode == "Hand":
                txt = st.text_input("Hole cards", value="As Ks" if i == 0 else "", key=f"hand_{i}",
                                    placeholder="e.g. As Ks")
                try:
                    cards = parse_cards(txt)
                    if len(cards) != 2:
                        raise ValueError(f"Player {i + 1}: enter exactly two cards.")
                    players.append({"hand": cards})
                    st.caption(" ".join(card_pretty(c) for c in cards))
                except ValueError as e:
                    errors.append(str(e))
            else:
                pcols = st.columns(len(PRESETS))
                for (name, text), pc in zip(PRESETS.items(), pcols):
                    pc.button(name, key=f"pre_{i}_{name}", on_click=set_range,
                              args=(i, parse_range(text)), use_container_width=True)
                st.text_input("Range text", key=f"txt_{i}", placeholder="e.g. TT+,AQs+,AKo")
                st.button("Apply text", key=f"apply_{i}", on_click=apply_text, args=(i,))
                edited = st.data_editor(range_to_df(st.session_state[f"range_{i}"]),
                                        key=f"grid_{i}_{st.session_state[f'ver_{i}']}",
                                        use_container_width=True, height=490)
                labels = df_to_range(edited)
                st.session_state[f"range_{i}"] = labels
                n_combos = len(range_combos(labels))
                st.caption(f"{n_combos} combos ({n_combos / 13.26:.1f}% of hands). "
                           "Diagonal = pairs, above = suited, below = offsuit.")
                if labels:
                    players.append({"range": labels})
                else:
                    errors.append(f"Player {i + 1}: range is empty.")

    board = []
    try:
        board = parse_cards(board_text) if board_text.strip() else []
        if len(board) not in (0, 3, 4, 5):
            errors.append("Board must have 0, 3, 4 or 5 cards.")
    except ValueError as e:
        errors.append(str(e))

    if not errors:
        known = list(board) + [c for p in players if "hand" in p for c in p["hand"]]
        if len(set(known)) != len(known):
            errors.append("The same card is used more than once.")

    st.divider()
    if errors:
        for e in errors:
            st.warning(e)
    if st.button("Run simulation", type="primary", disabled=bool(errors)):
        bar = st.progress(0.0, text="Simulating...")
        try:
            result = simulate(players, board, n_sims, lambda f: bar.progress(f, text="Simulating..."))
            st.session_state["result"] = result
        except ValueError as e:
            st.error(str(e))
        bar.empty()

    result = st.session_state.get("result")
    if result and len(result["equity"]) == n_players:
        st.subheader("Results")
        rcols = st.columns(n_players)
        for i, rc in enumerate(rcols):
            with rc:
                st.metric(f"Player {i + 1} equity", f"{result['equity'][i] * 100:.1f}%")
                st.progress(min(result["equity"][i], 1.0))
                st.caption(f"Win {result['win'][i] * 100:.1f}%  ·  Tie {result['tie'][i] * 100:.1f}%")
        st.bar_chart(pd.DataFrame({
            "Win": [w * 100 for w in result["win"]],
            "Tie": [t * 100 for t in result["tie"]],
        }, index=[f"Player {i + 1}" for i in range(n_players)]))
        st.caption(f"{result['hands']:,} hands dealt. Ties split the pot evenly.")

    ev_section()


if __name__ == "__main__":
    main()
