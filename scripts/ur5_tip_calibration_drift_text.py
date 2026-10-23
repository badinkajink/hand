"""The prose of the board-drift page (scripts/ur5_tip_calibration_drift_page.py), from the study's numbers R."""
from __future__ import annotations

import numpy as np

GROUPS = ["presses at 0.25 and 0.5 N", "presses at 1 N", "presses at 2 N", "presses at 3 to 5 N", "rolls and twists"]
SHORT = {"presses at 0.25 and 0.5 N": "0.25 to 0.5 N", "presses at 1 N": "1 N", "presses at 2 N": "2 N",
         "presses at 3 to 5 N": "3 to 5 N", "rolls and twists": "Rolls and twists"}
SHOWN = ["first", "time+field", "learned+field", "before", "before+field", "decay+field", "both"]


def sections(R, F, P, ZEROS):
    esc, fig, table = P.esc, P.fig, P.table
    name = dict(ZEROS)
    T, B, LB = R["table"], R["bias"], R["loc_bias"]
    S = []
    S.append(("terms", "Terms and metrics", """<dl class="terms">
<dt>Zero</dt><dd>The board's reading taken as no contact. The estimator reads each hold's change from it (µT, the five
chips' three components, after each chip's temperature correction).</dd>
<dt>Rest</dt><dd>0.5 s out of contact at an action's rest pose, 1.5 mm from the sphere, before and after the action. Its
mean reading is a measured zero.</dd>
<dt>Field term</dt><dd>Earth's field in each chip's axes at the tool's orientation, $B_k = b_{0,k} + A_k\\,R_{BE}^{\\top} w$.
Here $R_{BE}$ is the flange's rotation, $w$ the field and $A_k$ chip $k$'s rotation, fitted to first2's 12 gravity
stations: 44.3 µT, 59 deg below the horizontal, 1.8 µT rms left.</dd>
<dt>Step</dt><dd>An action's rest after less its rest before, both less the field term. Its size is the norm over the 15
components.</dd>
<dt>Share kept</dt><dd>How much of a step the next action's rest before still holds: 1 if all of it stays, 0 if all of it
went back.</dd>
<dt>Location and force errors</dt><dd>The estimated contact point's distance from the true one along the surface (mm),
and the normal force's error (N, estimate less the HEX-E's), against the workstation's ground truth.</dd>
</dl>"""))

    S.append(("zero", "The zero over the session", (
        f"<p>The session sphere2 has {R['n_rests']} rests over {R['minutes']:.0f} min, from 21:08 to 23:27, in three "
        f"runs of the program. Less the first rest, the readings moved by up to {R['raw_span_uT']:.0f} µT on a "
        f"component. The field term takes out {100 * R['field_share']:.0f} % of that change (Figure 1). The tool's "
        "orientation changed from one target to the next, and Earth's field turned with it. Less the field term, the "
        f"zero ended {R['accum_end_uT']:.0f} µT from the first rest, over all 15 components. A line in time fitted to the "
        f"session's first half moves {R['drift_time_only_uT_min']:.2f} µT a minute.</p>"
        + fig(1, F["zero"], "The board's rests against time, as their change from the first rest over all 15 components. "
              "Grey: the readings. Blue and red: less the field term, before and after each action. Orange: the learned "
              "zero of Section 4. Purple, dashed: the line in time. The learned zeros are fitted left of the dotted line "
              "and tested right of it.")
    )))

    al = R["align"]
    tops = sorted({round(x["top_N"], 2) for x in al})
    by_top = {t: [x for x in al if round(x["top_N"], 2) == t] for t in tops}
    sizes = [np.median([x["step_uT"] for x in v]) for t, v in by_top.items() if len(v) >= 4]
    per_N = float(np.median([x["change_uT"] / x["top_N"] for x in al if x["top_N"] <= 2.0]))
    S.append(("steps", "What a press leaves", (
        f"<p>{R['n_steps']} actions have rests on both sides. Their steps are {R['step_med_uT']:.1f} µT in the median "
        f"and {R['step_p90_uT']:.1f} µT at the 90th percentile. A press of 1 or 2 N changes the board by about "
        f"{per_N:.0f} µT per newton, so a step is about the change of {R['step_med_uT'] / per_N:.1f} N. Its size hardly "
        f"depends on the press: {min(sizes):.0f} to "
        f"{max(sizes):.0f} µT in the median from 1 N to 5 N (Figure 2, left). Its direction does. The cosine between a "
        f"press's step and its top hold's change is {R['align_cos_med']:.2f} in the median, and positive in "
        f"{100 * R['align_pos']:.0f} % of {len(al)} presses. For random directions in 15 components the median is "
        "0.19. A press "
        f"leaves a little of itself behind, {100 * R['align_share_med']:.0f} % of its change along it.</p>"
        f"<p>The next action's rest came {R['gap_med_s']:.0f} s later in the median and still held "
        f"{100 * R['keep_med']:.0f} % of the step (Figure 2, right). On the session's first half, the share kept against "
        "the time to that rest fits"
        "$$k(\\Delta t) = a + (1 - a)\\,e^{-\\Delta t/\\tau}$$"
        f"with $a$ = {R['decay_a']:.2f} and $\\tau$ = {R['decay_tau_s']:.0f} s. Most of a step fades, and the steps "
        f"add up to little. Had each one stayed, the zero would have moved {R['steps_sum_uT']:.0f} µT by the end. It moved "
        f"{R['accum_end_uT']:.0f} µT.</p>"
        + fig(2, F["steps"], "Left: each action's step against its top force, by region of the pad. Right: the share of "
              "the step that the next action's rest still holds, against the time to that rest. Dashed: all of it stays "
              "(red), all of it goes back (blue).")
    )))

    S.append(("learned", "Learned zeros", (
        "<p>Two learned zeros never re-zero after the session's first rest. Both were fitted on the first half of the "
        f"session, to {R['t_split_min']:.0f} min, and tested on the second. The first is the zero against time since "
        "the first rest, a line per component, as eFlesh subtracts its drift. The second adds each press's step, "
        "predicted from its top force, its time in contact, its target and its region, and fading as above."
        "$$z(t) = z_0 + v\\,(t - t_0) + \\sum_i \\hat s_i\\, k(t - t_i)$$</p>"
        f"<p>On the second half's rests, the line in time misses by {R['miss_med_uT']['time']:.1f} µT in the median, "
        f"against {R['miss_med_uT']['first']:.1f} µT for the first rest alone. The learned steps miss by "
        f"{R['miss_med_uT']['learned']:.0f} µT. Their prediction misses each step by {R['step_pred_med_uT']:.1f} µT, "
        f"where the steps themselves are {R['step_size_med_uT']:.1f} µT. A press's force, place and length do not tell "
        "its step.</p>"
    )))

    loc_rows = [[esc(name[k])] + [f"{T[g][k]['loc_med']:.1f} ({T[g][k]['loc_p90']:.1f})" for g in GROUPS] for k in SHOWN]
    f_rows = [[esc(name[k])] + [f"{B[g][k]['err']:+.2f} ({B[g][k]['ratio']:.2f})" for g in GROUPS] for k in SHOWN]
    hi = "presses at 3 to 5 N"
    lo = "presses at 0.25 and 0.5 N"
    one = "presses at 1 N"
    upper = ["presses at 2 N", hi, "rolls and twists"]
    d_loc = max(abs(T[g]["decay+field"]["loc_med"] - T[g]["before+field"]["loc_med"]) for g in upper)
    d_f = max(abs(B[g]["decay+field"]["err"] - B[g]["before+field"]["err"]) for g in upper)
    tw = "rolls and twists"
    f_loc = abs(T[tw]["before+field"]["loc_med"] - T[tw]["before"]["loc_med"])
    f_f = abs(B[tw]["before+field"]["err"] - B[tw]["before"]["err"])
    stale = [B[g]["first"]["err"] - B[g]["before"]["err"] for g in GROUPS]
    offl = [B[g]["both"]["err"] - B[g]["before"]["err"] for g in (one, "presses at 2 N", hi)]
    S.append(("estimator", "The estimator against each zero", (
        f"<p>Tables 1 and 2 and Figure 3 give the viewer's estimator on each of the second half's holds, against "
        "each zero. Re-zeroing at rest is the zero to keep. With the rest just before the press as its zero, the "
        f"estimator's median error is {T['presses at 2 N']['before']['loc_med']:.1f} mm at 2 N and "
        f"{T[hi]['before']['loc_med']:.1f} mm at 3 to 5 N. The decay-aware zero adds to that rest what the earlier "
        "steps go on recovering during the press."
        "$$z(t) = r_j + \\sum_{i<j} s_i \\left[ k(t - t_i) - k(t_j - t_i) \\right]$$</p>"
        f"<p>That moves the zero by {R['decay_corr_med_uT']:.1f} µT in the median and {R['decay_corr_p90_uT']:.1f} µT at "
        f"the 90th percentile. From 2 N up it changes the scores by at most {d_loc:.2f} mm and {d_f:.2f} N. At 1 N, "
        "where the estimator places half the contacts about 10 mm off, it moves the median location from "
        f"{T[one]['before+field']['loc_med']:.1f} to {T[one]['decay+field']['loc_med']:.1f} mm.</p>"
        "<p>The field term changes by up to 10 µT between rest and hold in the rolls and 13 µT in the twists. "
        f"Subtracting it moves their scores by {f_loc:.3f} mm and {f_f:.3f} N. The estimator fits a uniform field of its "
        "own (a 30 µT prior) and takes Earth's term into it.</p>"
        "<p>The learned zeros are worse than no re-zero at all. At 2 N they give "
        f"{T['presses at 2 N']['time+field']['loc_med']:.1f} mm (the line in time) and "
        f"{T['presses at 2 N']['learned+field']['loc_med']:.1f} mm (the learned steps).</p>"
        "<p>No re-zero, the first rest, looks as good as re-zeroing from 1 N up, and better from 2 N up. In both cases "
        "its offset cancels part of an error of the estimator's own. Against the rest before, the estimator reads "
        f"{B[lo]['before']['err']:+.2f} N at 0.25 to 0.5 N and {B[hi]['before']['ratio']:.2f} of the force at 3 to "
        f"5 N. The stale zero adds {min(stale):.1f} to {max(stale):.1f} N, by level. That hides part of the shortfall at "
        f"high force and raises the error at low force to {B[lo]['first']['err']:+.2f} N. The location behaves the same "
        f"way. From 2 N up, the contacts lie {LB['before']['mean_norm']:.1f} mm off on average with the rest before, "
        f"and {LB['first']['mean_norm']:.1f} mm with the stale zero. Their scatter about that mean is "
        + (f"the same, {LB['before']['scatter']:.1f} mm for both. " if f"{LB['before']['scatter']:.1f}" == f"{LB['first']['scatter']:.1f}"
           else f"{LB['before']['scatter']:.1f} and {LB['first']['scatter']:.1f} mm. ")
        + "A calibrated estimator would remove "
        "the bias, and the stale zero's advantage with it.</p>"
        "<p>With the offline zero, the two rests interpolated to the hold, the estimator reads "
        f"{-max(offl):.2f} to {-min(offl):.2f} N lower than with the rest before, from 1 N up. A step points along the press's "
        "own change, so moving the zero toward the rest after takes part of the press out of its own reading.</p>"
        + table(["Zero", *[SHORT[g] for g in GROUPS]], loc_rows, num_cols=(1, 2, 3, 4, 5),
                caption="The contact point's error (mm, median and 90th percentile) on the session's second half, by "
                "force level.", n=1)
        + table(["Zero", *[SHORT[g] for g in GROUPS]], f_rows, num_cols=(1, 2, 3, 4, 5),
                caption="The normal force's error (N, estimate less the HEX-E's, median) and the estimate's ratio to the "
                "HEX-E's (median), by force level.", n=2)
        + fig(3, F["scores"], "The estimator on the second half's holds against each zero, by force level. Left: the "
              "contact point's error (median). Right: the normal force's error (median, estimate less the HEX-E's).")
    )))

    S.append(("use", "What to use", """<ul>
<li>Re-zero at rest just before contact, after the chips' temperature correction. The viewer's auto-zero does this when
the pad has rested for 3 s. Neither the field term nor the decay needs subtracting for this estimator.</li>
<li>The board's change for the calibration's ground truth could be taken against the rest before alone, rather than the
two rests interpolated (Table 2). The analysis now offers that for the board (<code>--board-zero before</code>), as it
does for the HEX-E (<code>--zero before</code>); the workstation is rerunning sphere2 with it.</li>
<li>Learning should go into the estimator's response, not the zero. It reads high at low force, short at high force,
and places contacts with a bias of about 2 mm. The workstation's predictors, trained on the same session's presses,
reach 0.84 mm and 0.09 N.</li>
</ul>"""))

    S.append(("future", "Future work", f"""<ul>
<li>Longer rests in a session on mount v4. Only {R['long_gaps']} steps here had more than a minute to the next rest.
So the part of a step that lasts, $a$ in the fit, is this page's least certain number.</li>
<li>A step model from the press's own change, which the step's direction follows, in place of its force and place.</li>
<li>The estimator's gain and offset by region, fitted on the reference presses, and tested on a later session.</li>
</ul>"""))

    lede = (f"Over the {R['minutes'] / 60:.1f} h calibration session sphere2, the fingertip board's reading at rest "
            f"changed by up to {R['raw_span_uT']:.0f} µT. Earth's field turning with the tool made "
            f"{100 * R['field_share']:.0f} % of that change. The estimator removes a uniform field of its own, so "
            "subtracting the field changes none of its answers. A press leaves the drift that matters: a step of "
            f"{R['step_med_uT']:.0f} µT in the median, about the change of 0.3 N, along the press's own change. A share "
            f"of {R['decay_a']:.2f} of it lasts past a minute, and the rest fades with a time constant of "
            f"{R['decay_tau_s']:.0f} s. The steps could not be predicted from a press's force, place or length, so no "
            "learned zero beat measuring the zero at rest. Re-zeroing at rest just before each press is the zero to keep. A stale "
            "zero looks better only where its offset cancels part of the estimator's own errors, which limit it more "
            f"than the drift does. The estimator reads {B[lo]['before']['err']:.1f} N high at 0.25 to 0.5 N and gives "
            f"{B[hi]['before']['ratio']:.2f} of the force at 3 to 5 N. From 2 N up its contacts lie "
            f"{LB['before']['mean_norm']:.1f} mm off on average.")
    summary = (f"sphere2's board zero: {100 * R['field_share']:.0f} % of its {R['raw_span_uT']:.0f} µT change is Earth's "
               f"field, which the estimator absorbs; each press leaves a {R['step_med_uT']:.0f} µT step along its own "
               f"change, {R['decay_a']:.2f} of it lasting and the rest fading in {R['decay_tau_s']:.0f} s. Learned zeros "
               "lose to re-zeroing at rest, and the estimator's response (high at low force, "
               f"{B[hi]['before']['ratio']:.2f} of the force at 3 to 5 N) dominates the error.")
    return S, lede, summary
