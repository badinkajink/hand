# Hand–object scale matching: suggestion and literature summary (owner-pasted, 2026-10-07)

Source: the owner pasted this on 2026-10-07. Sunyu Wang suggested reproducing the hand–object scaling
effect on SR2; the text below is a ChatGPT response about that suggestion, kept verbatim. Its
citations and the claims attributed to them have not been checked. The task note
`docs/handoff/20261007-hand_object_scale_matching.md` verifies them before anything is built on them.

Owner, 2026-10-07: "it would be helpful to tinker on this and plan (and execute toy experiments) a
reasonable experimental protocol (and paper). from what he thinks, and i think i agree?, this is the
most feasible path toward a publishable paper that: 1) answers an interesting question 2) demonstrates
the utility of the hand and 3) is not unreasonable. the improved contact modeling sorta gets
overshadowed here, but it can just be part of the paper. if it has to be a journal paper to fit
everything, that's completely fine."

---

Yes. After checking the literature, I think Sunyu's suggestion is much better than it may have sounded in conversation. The important caveat is that I would call the phenomenon hand–object scale matching, not simply “big hands are better for big objects.” The latter is directionally right, but conflates several morphology variables whose effects are not identical.
1. The underlying phenomenon is real. In human prehension, object size relative to hand size is a classic organizing variable. Studies across adults and children find that grip configurations transition at similar object-size/hand-size ratios, rather than at the same absolute object size. PubMed Ergonomic studies likewise find that grip force, normal force, and contact area depend strongly on handle diameter relative to hand length, and that the preferred/force-maximizing grip span shifts upward for larger hands. PubMed
   For dexterous manipulation specifically, however, it is not as simple as monotonically “smaller object → smaller hand is better.” Bullock, Feix, and Dollar measured human precision-manipulation workspace and found a preferred intermediate object scale relative to hand length; the smallest objects actually produced the smallest workspace. PubMed So biomechanics says much more defensibly: performance depends on the scale of the object relative to the hand, and there are favorable hand–object scale relationships.
   On the robotic side, the evidence is almost comically on-the-nose. Hota and Kumar's 2019 paper is literally titled Effect of hand design and object size on the workspace of three-fingered hands. For a fully actuated three-finger tripod hand, they found that farther-apart finger bases give larger manipulation workspace for larger objects, while closer finger bases give larger workspace for smaller objects. They also found that longer finger links increase workspace for both large and small objects—which is exactly why I would not reduce this to “every dimension of a big hand should scale with object size.” ScienceDirect
   Even earlier, Borràs and Dollar's dimensional synthesis found an explicit geometric relationship: for their three-finger precision-manipulation model, maximal workspace tended to occur when palm radius was roughly target-object radius plus half a finger length. Sage Journals
2. Yes, this is science, but the science cannot be “we discovered that wide hands handle big things.” If you wrote Nancy saying that as the research question, it would sound naive precisely because her community has already studied this theoretically.
   The stronger framing is:
   Use a well-understood hand–object scaling effect as a canonical physical experiment, then ask whether its predicted structure appears across a continuous physical morphology landscape, why it appears, how it differs between tasks, and where simulation gets it wrong.

   That is quite different.
   In fact, choosing something whose first-order answer we think we know is arguably the right experiment for \(SR^2\). You have built an instrument. A new scientific instrument is initially most convincing when it can recover a known phenomenon, characterize it much more densely than was previously practical, and then reveal second-order behavior that the simple theory misses.
   The cleanest mathematical statement is almost embarrassingly simple. Let \(s\) be some measure of physical hand scale—initially finger-base span—and \(d\) object diameter. For task \(T\), measure
   \[
   J_T(s,d),
   \]
   then find
   \[
   s_T^*(d)=\arg\max_s J_T(s,d).
   \]
   The boring-but-important prediction is
   \[
   \frac{d s_T^*}{d d}>0:
   \]
   larger objects prefer larger base spacing.
   The science starts immediately afterward. Is \(s_T^*/d\) approximately constant? Is the relationship linear? Where does it saturate? Does it collapse onto a dimensionless scale law? Is \(s_{\mathrm{grasp}}^*(d)\) the same as \(s_{\mathrm{reorient}}^*(d)\)? What happens to robustness around the optimum? Does simulation predict the location and width of that optimum? Which physical quantity explains it?
   Those are real questions.
   And I would particularly expect the task difference to be interesting. A morphology that gives an excellent initial grasp is not necessarily good for manipulation. Stable grasping cares about contact geometry, force transmission, and disturbance resistance. Reorientation additionally cares about remaining motion after the grasp is formed: joint reserve, fingertip tangential workspace, singularities, collisions, torque margin, and the ability to change contact without ejecting the object. Your current cylinder experiments have already given you a taste of exactly that distinction: object motion and retention are not the same metric.
3. Pieces of this have absolutely been done before; I do not see that as bad news. The exact kinematic hypothesis is already in Hota and Kumar 2019. ScienceDirect Borràs and Dollar 2015 systematically synthesize three-finger hand geometry for precision manipulation workspace. Sage Journals Elangovan et al. optimize link dimensions and interfinger distance for an object and show that changing finger-base distance produces substantially different manipulation workspaces; they then build a gripper with reconfigurable finger bases. Enlighten Publications RUTH goes further and actively reconfigures a two-DoF palm to grasp and manipulate objects of different sizes and shapes on real hardware. Sage Journals A 2026 variable-stroke soft hand also directly reports improved physical grasping when its initial finger spacing is matched to object size. PubMed And a very recent 2026 reconfigurable dual-opposition hand formulates an object-size-conditioned workspace and demonstrates rotation across a broad range of object sizes. arXiv
   So “reconfigurable spacing lets a hand accommodate different object sizes” is emphatically not the novelty.
   What I have not found in these papers is quite the experiment \(SR^2\) naturally enables: a dense, controlled physical map in which morphology is treated as the independent variable, swept smoothly through many configurations on the same hardware, across a systematic object-size gradient and multiple tasks, with the corresponding simulated landscape evaluated at exactly the same points.
That last distinction is important. RUTH, for example, uses palm reconfiguration as part of the manipulation mechanism. It is trying to make one hand versatile. Your hand can instead hold morphology fixed during a trial and say:
this exact physical hand is morphology \(m=0.32\);
this one is \(m=0.34\);
this one is \(m=0.36\).

Same motors. Same links. Same fingertips. Same controller family. Same object. Same sensing. Then change object size and do it again.
That is much closer to an experimental morphology rig than a reconfigurable end effector.
I think there is a particularly clean experiment sitting here
For the first pass, I would resist the temptation to use all six morphology coordinates. Define a one-dimensional family of increasingly large tripod morphologies—basically scale the three finger-base positions outward/inward about a common center. Then sweep perhaps 6–10 object diameters.
You get a matrix like
\[
\begin{array}{c|cccccc}
 & d_1 & d_2 & d_3 & d_4 & d_5 & d_6\\
\hline
s_1 & J & J & J & J & J & J\\
s_2 & J & J & J & J & J & J\\
\vdots &&&&&&\\
s_n & J & J & J & J & J & J
\end{array}
\]
for each task.
The result you hope to see is not merely “upper-right good, lower-left good.” You want a ridge through the hand-size/object-size plane. Then ask what determines its shape.
For each cell, you already know how to calculate candidate explanatory variables:
- fingertip reach / joint-limit margin;
- Jacobian conditioning or manipulability;
- contact normals / force closure;
- available joint travel after establishing the grasp;
- actuator torque margin;
- contact force and retention;
- simulated perturbation robustness.
Then the paper becomes something like:
Object scale moves the optimal physical morphology smoothly through design space; the optimum can be explained by X for grasping but Y for reorientation, and simulation captures A while systematically mispredicting B.

That is considerably more interesting than “large hand → large ball.”
And once you have the 1-D result, you can let \(SR^2\) flex its actual muscles. Your morphology isn't really one-dimensional size. You can independently vary the thumb-to-pair span and index–middle opening—the two coordinates you are already using in your population figure. Then instead of one scalar \(s\), you get
\[
J_T(s_{\mathrm{thumb}},s_{\mathrm{pair}},d).
\]
Now you can ask whether a large object wants the entire tripod uniformly expanded, or whether thumb opposition and pair opening scale differently. That sounds much more like hand science.
The fixed phalanges are not fatal—in some ways they're useful
You're right that the current \(SR^2\) cannot really make a geometrically scaled “big hand” and “small hand.” Its phalanges remain fixed.
So I would be very precise and call the manipulated variable finger-base spacing / palm morphology, not hand size.
That actually gives you a cleaner causal experiment:
Keeping finger geometry, actuation, sensing, material, and contact surfaces fixed, how should finger placement change with object scale?

Then the proposed variable-phalanx-length version of \(SR^2\) becomes a very natural second experiment. Once you can scale both finger placement and link lengths, you can ask whether true approximately homothetic hand scaling gives a cleaner size-invariant relationship.
And there is already reason to expect that the answer will not simply be “everything scales together”: Hota and Kumar found that longer links tended to increase workspace for both small and large objects. ScienceDirect Hu et al. likewise find that hand proportions have nontrivial optima—e.g. both excessively long and excessively short thumbs hurt modeled manipulation dexterity. ScienceDirect
So the simple intuition becomes the baseline that the experiment can eventually complicate.
Why \(SR^2\), specifically?
This is where I think Sunyu's suggestion really makes sense.
The selling point isn't merely that your hand can get wide and narrow. It is that it can make the morphology variable dense, repeatable, automated, and controlled.
If you fabricated five hands, you'd always have to wonder whether one had slightly different assembly, tendon tension, print properties, servo mounting, etc. With \(SR^2\), the exact same three fingers can move through the entire base-layout experiment.
More importantly, you can do things that are tedious enough to basically not happen with manual reconfiguration:
randomize morphology order; revisit the same morphology after 50 trials; perturb a base by \(2\) mm and rerun; adaptively sample around a surprising transition; trace an optimum continuously as object diameter changes; warm-start the controller at the neighboring morphology; compare the identical morphology days later.
That is where programmatic morphology becomes scientifically meaningful rather than merely convenient.
There is even a nice collaboration fit with Sunyu's current work. His new hand-object mechanism framework is explicitly trying to represent manipulation through contact-relative quantities that are insensitive to the specific hand kinematics/geometry, and he demonstrates it across five different hands in simulation. arXiv \(SR^2\) gives you the inverse experimental capability: continuously change the embodiment while keeping the task and representation controlled. If their control representation works across your morphology sweep, you have dramatically reduced the “maybe this hand just had a worse controller” confound.
That feels like a pretty natural Pollard-lab collaboration to me.
What I'd say to Nancy
I would not email:
We want to test whether bigger hands are better for bigger objects.

I would email something much closer to:
Sunyu suggested using a simple, well-understood hand/object scaling effect as a first experiment with \(SR^2\). Prior kinematic work predicts that larger objects favor greater finger-base spacing while smaller objects favor more compact layouts, which seems like a useful canonical case rather than a new hypothesis by itself. What I think our platform could add is a dense physical map of that relationship: sweeping the same fingers continuously through morphology space across object sizes and tasks, comparing the resulting physical and simulated landscapes, and trying to identify why the optimum shifts and where the simple scaling intuition breaks down. I'm especially interested in whether grasping and in-hand manipulation select the same morphology and whether a morphology-independent control representation can keep the experiment from becoming controller-limited.

I would be completely comfortable sending that to Nancy.
It signals that you did the literature check, aren't claiming an obvious phenomenon as novel, and understand why your apparatus could expose something deeper.
The Hota paper is actually a gift here. It converts Sunyu's hallway intuition into a concrete prior prediction:
fully actuated three-finger hand + tripod grasp + variable finger-base placement + variable object size → wide bases for big objects, compact bases for small objects. ScienceDirect

Your experiment can essentially say: great—does that plot exist in reality? What happens when the objective changes from kinematic workspace to actual grasp robustness or reorientation? And can we explain deviations?
That is exactly the sort of “simple phenomenon, deeper measurement” experiment I would use to justify a scientific instrument.
