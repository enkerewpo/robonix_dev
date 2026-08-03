## Scene quality + resource analysis, and where I think we should go next

I went through the evidence artifact behind the results table (the five worlds on
the review page, `DATA[]` in the generated report), the ConceptGraphs ingest path
(`system/scene/scene_service/ingest/perception_concept_graphs.py` on this branch),
and the Jetson build/run targets. Summary of what the numbers actually say, where
we diverge from the ConceptGraphs paper, and what I'd fix — accuracy first,
then the resource story, which I think is currently the bigger adoption blocker.

---

### 1. The error budget, measured

Micro totals over the five worlds, decomposed by *failure kind* rather than by
world (recomputed from the per-object statuses in the artifact, not re-stated
from the summary table):

| Bucket | Count | Note |
|---|---|---|
| Correct object, correct label | 138 | |
| Correct object, **wrong label** | 82 | still counted as TP; this is what drops label acc to 0.627 |
| **Duplicate** (unmatched prediction *on top of* a real object) | 64 | 49% of all FP |
| **Ghost** (no admissible truth nearby) | 67 | 51% of all FP |
| **Missed** truth | 86 | |

So the 0.670 micro-F1 is not one problem. It is three roughly equal problems:
identity fragmentation (64), spurious geometry (67), and naming (82).

Stratifying by ground-truth object size makes the pattern much sharper:

| GT max dimension | n | recall | label acc among matched |
|---|---|---|---|
| < 0.30 m | 69 | **0.54** | **0.30** |
| ≥ 0.30 m | 237 | 0.77 | 0.69 |

Small objects are where the pipeline collapses, on *both* axes at once. Missed
truths are dominated by exactly those: wineglass ×7, book ×5, plate ×4, carafe
×3, orange ×3, bowl/knife/spoon/apple ×2 each. This is a systematic effect, not
scene-specific noise.

Second, the label errors are overwhelmingly *near-synonym* errors inside our own
vocabulary. Pairing each mislabelled prediction to its nearest mislabelled truth:
**46 / 82 (56%) are confusions within one coarse category** — shelf↔cabinet↔counter
(9), cup↔jar/can (8), picture_frame↔window (4), couch↔chair (3), television↔monitor
(2), pot↔pan (2), and so on. The remaining 36 are genuine cross-category errors
and are concentrated on the small objects above.

Third, precision and recall are confounded with exploration. `coverage` in the
artifact is 0.47 (office), 0.60 (kitchen), 0.77 (complete_apartment). Office
"recall 0.917" and complete_apartment "recall 0.679" are not comparable to each
other while half of one room was never visited. We should be reporting
`recall@covered` alongside raw recall.

---

### 2. Where we diverge from ConceptGraphs — and what each divergence costs

The module docstring says we keep ConceptGraphs as the perception backbone. In
practice five things changed, and I believe four of them are the direct cause of
the numbers above.

**(a) Association and merging are class-gated. In the paper they are class-agnostic.**

This is the big one. `_association_compatible()` requires both labels to fall in
the same merge group, and `allow_cross_class_merge` defaults to `False`
(reported as `"allow_cross_class_merge": false` in every run). ConceptGraphs
deliberately never lets the detector's class name participate in association —
an object's identity is its point cloud plus its CLIP descriptor, and the name is
a *property* derived afterwards. We made the name a *key*, which reintroduces
exactly the over-segmentation the paper's design exists to avoid.

The artifact shows this is not theoretical. In every one of the five worlds:

```
canonical_eligible_pairs: 0
cross_class_geometry_rejected_pairs: 61 / 132 / 183 / … 
```

The canonical `merge_overlap_objects` pass **never fired once, in any world.**
Every merge that happened came from one of our bespoke bypasses. A representative
rejected pair from the office run:

```
left: "bathtub"  right: "couch"
center_distance_m: 0.335   voxel_overlap: 0.714   clip_cosine: 0.711
class_compatible: false   →  rejected
```

That is one physical sofa split into two tracks with two labels, with 71% volume
overlap and 0.71 CLIP cosine, and we refuse to merge it *because the labels
differ* — when the differing labels are themselves the bug. This is where most of
the 64 duplicates come from, and a good share of the 67 ghosts (a fragment that
drifts off the real object stops being "duplicate" and becomes "ghost").

**(b) The spatial similarity is not the paper's, and the substitute is
view-dependent.**

The paper's `overlap` is a nearest-neighbour ratio: the fraction of points in the
detection whose nearest neighbour in the object cloud is within a radius. We
substituted exact voxel-set intersection at a 2.5 cm grid
(`_voxel_pcd_overlap_matrix`) because the upstream path went through
`pytorch3d.box3d_overlap` and crashed on coplanar vertices. That was the right
call for stability, but exact voxel intersection is a *strictly harsher* measure
in precisely the case that matters: the same surface observed from a different
viewpoint is sampled at different points, and with RGB-D noise ≥ 2.5 cm the two
observations often land in adjacent voxels and score ~0 overlap. So spatial
similarity collapses toward zero exactly when we need it to carry the cross-view
merge — which is why the merge threshold had to be hand-walked between 0.55, 0.85
and 1.10 in the config comments, and why we then needed the distance gate, the
adaptive distance gate, the one-to-one gate, the co-observed 2D-IoU gate and the
disjoint-history gate on top. A radius-tolerant nn-ratio (cKDTree, radius ≈ 1–2
voxels) restores the paper's behaviour and would let most of that gate stack go.

**(c) Naming is a closed 105-word argmax plus a hand-tuned rerank. The paper
names objects with an LLM over multi-view crops.**

Our label path is: YOLO-World argmax over a fixed 105-class prompt list → a
temporal histogram → `_clip_rerank_label`, which only reranks *within a manually
configured candidate set* per label (`window ↔ picture frame`, `chair ↔ monitor`, …)
with per-label margins in config. The office run carries a margin table with
entries like `window: 0.002`, `monitor: 0.0`, `shelf: 0.06`. Those are fitted to
these Webots worlds; they will not transfer.

Two structural consequences:

1. The vocabulary contains ~8 near-synonymous storage words (shelf / bookshelf /
   cabinet / drawer / dresser / wardrobe / counter / nightstand) and ~8 near-synonymous
   vessel words (cup / mug / glass / wineglass / paper cup / jar / can / thermos).
   ViT-B/32 text prototypes for those are nearly collinear, so the choice among
   them is close to a coin flip. That is the 56% within-group error rate above,
   and no amount of per-pair margin tuning fixes it — the pairs are combinatorial.
2. Anything outside the 105 words is unnameable and, because YOLO-World is also
   our *proposal* mechanism, largely undetectable. In the paper, proposals come
   from class-agnostic SAM, so recall is bounded by segmentation, not by a word
   list.

Worth noting: **we already run a VLM on numbered-box renders of the tracked
objects** — `scene_graph/builder.py` + `image_relations.py`, every ~30 s, for
relation extraction. It is receiving exactly the input the paper's captioning
stage needs, and we throw the naming opportunity away.

**(d) CLIP descriptors are computed on 224² resizes of small crops.**

At 640×480, a wineglass at 2 m is ~20 px wide. Upsampled to 224² it carries
essentially no signal, so both the descriptor (association) and the rerank
(naming) degrade together. That is the mechanism behind the 0.54 recall / 0.30
label accuracy on sub-30 cm objects. Compounding it: `downsample_voxel_size` 2.5 cm
with `min_points_threshold` 50 and `obj_min_points` 20 means a 8 cm-wide object
has to survive a filter sized for furniture.

**(e) Pose provenance is unlike the paper's evaluation setting.**

ConceptGraphs is evaluated on Replica/ScanNet with ground-truth poses. We run on
live SLAM, and the artifact shows what that means:

```
transform_source_counts: { stamped_full_tf: 26,
                           stamped_odom_plus_current_map_correction: 298 }
map_correction_total_translation_m: 2.409
```

92% of frames did **not** get a real TF lookup at the image stamp — they got
odometry composed with the *current* map→odom correction, i.e. a correction from a
different time than the image. With 2.4 m of accumulated correction over a 6.7 m
path, that is a direct contributor to the 0.28 m median centre error and the 0.57
cloud-inlier fraction in office. The mapping metrics agree that the substrate is
part of the problem: `wall_angle_p95_error_deg` is 9.7° in kitchen and **25.5°** in
complete_apartment — the SLAM map itself is skewed in the large world, and object
geometry inherits it. Some of what we are currently charging to "the perception
algorithm" is really pose plumbing.

---

### 3. Why Scene doesn't fit on a Jetson

This is the complaint I hear most, and I think it's the one with the best
effort-to-value ratio. Nothing here is a research problem.

**Two full CLIP ViT-B/32 models are resident at once.** `_try_load_models()`
constructs `YOLO(yolov8l-world.pt)` — ultralytics instantiates a `clip-anytorch`
ViT-B/32 text tower to service `set_classes()` — and then separately builds
`open_clip` ViT-B/32 from `/opt/models/open_clip_pytorch_model.bin` (≈605 MB fp32).
We pay for the same architecture twice, and after `set_classes()` the first one
is dead weight: the prompt embeddings are baked and the text tower is never used
again at runtime.

**Everything is fp32.** No `.half()`, no autocast anywhere in the ingest path
(the only `torch.no_grad()` uses are in the rerank/embed helpers). Orin's fp16
tensor cores are idle. This is 2× weight memory and roughly 2× latency, for free.

**No TensorRT / ONNX path.** Once `set_classes()` has run, YOLO-World is a
fixed-vocabulary YOLOv8-l and `model.export(format="engine", half=True)` applies
directly. On AGX Orin the difference between PyTorch fp32 and TRT fp16 for a
YOLOv8-l at 640 is roughly an order of magnitude.

**`yolov8l-world` is the default on every target,** including `jetson-native`.
The `-l` variant is ~4× the FLOPs of `-s`. There is no profile switch.

**The import surface is enormous, and most of it is dead.** Because we import
concept-graphs' `slam.*` modules, we transitively require `hydra-core`,
`omegaconf`, `openai`, `wandb`, `rich`, `matplotlib`, `seaborn`, `faiss-cpu` and
`supervision` — the requirements file has comments explaining that several of
these exist purely to satisfy unconditional upstream imports we never call. We
actually use about eight functions from that package, and we already replaced its
`pytorch3d` path with our own. `open3d` (only used for voxel downsample, DBSCAN
and AABB — all of which are short numpy/`cKDTree`) is several hundred MB of RSS
and the most fragile aarch64 dependency in the stack.

**There is no perception-free profile.** Issue #156 already asked for this: a
deployment that only wants robot pose, room annotations and map binding still
drags in torch, CUDA context, YOLO-World and MobileSAM. On an Orin Nano sharing 8
GB with rtabmap, Nav2 and the DDS stack, that is the difference between "boots"
and "doesn't".

**Per-tick CPU grows quadratically with map size, in pure Python, under the
inference lock.** `_voxel_pcd_overlap_matrix` builds a `frozenset` of 3-tuples per
object cloud, and `_voxel_pcd_overlap_torch(det_list, self._map_objects)` calls it
against **all** map objects **every tick** — no frustum filter, no spatial
pre-gate, no caching between ticks. In complete_apartment that is 165 objects ×
up to `obj_pcd_max_points`=5000 points, so on the order of 8×10⁵ Python tuple
allocations per tick at 1.6 Hz, plus O(N²) set intersections in the cleanup pass.
On x86 this is merely wasteful; on Orin's cores it is the tick budget.

---

### 4. Proposed direction

Ordered by what I think is expected-gain-per-effort. Happy to be argued out of
the ordering.

#### Accuracy

**A1 — Make identity class-agnostic again.** Default `allow_cross_class_merge` to
true; delete the class gate from association and from `merge_overlap_objects`
eligibility. Keep the *geometric* identity test we already built (voxel coverage +
extent ratio + centroid distance) as the safety net — it is the right kind of
gate, because it constrains physics rather than vocabulary. Treat the label as a
per-track histogram property. This should recover most of the 64 duplicates;
micro precision 0.627 → ~0.75 on the duplicates alone, F1 → ~0.74, before any
naming work.

**A2 — Replace exact voxel intersection with a radius-tolerant nn-ratio.**
`cKDTree(object_points).query_ball_point(det_points, r=2×voxel)` → fraction
matched. This is the paper's measure, it is view-invariant, and it lets us delete
several of the compensating gates. Do A1 and A2 together and re-tune
`merge_threshold` once, from evidence, rather than per-world.

**A3 — Move naming to the VLM we already run.** `image_relations.py` already
renders numbered boxes for the tracked objects every ~30 s. Extend that one call
to return `{box_id: name}` alongside the relations, and let YOLO-World be a pure
*proposal* mechanism whose class output feeds the histogram as a weak prior only.
This deletes the closed vocabulary from the naming path, deletes the per-label
margin table, and directly attacks the 46 near-synonym errors — a VLM looking at
a picture of a kitchen does not confuse a counter with a cabinet. Keep the current
path as the offline/no-credentials fallback.

**A4 — Scale-aware geometry admission.** Make `downsample_voxel_size`,
`min_points_threshold` and `obj_min_points` functions of the detection's apparent
extent instead of constants sized for furniture (e.g. voxel =
`clamp(0.3 × min_extent, 0.01, 0.05)`). Combined with a higher-resolution second
detection pass (larger `imgsz`, or SAHI-style tiling on the central crop) gated to
when the base is stationary, this targets the 0.54 small-object recall.

**A5 — Fix pose provenance.** Lengthen the tf2 buffer so
`lookup_transform(map, camera_optical, image_stamp)` actually succeeds instead of
falling back to `odom + current correction` on 92% of frames, and drop frames
whose map→odom correction moved more than a few cm between the image stamp and the
lookup. Worth a meaningful fraction of the 0.28 m centre error, and it is plumbing,
not research.

**A6 — Report `recall@covered` and per-size-bucket metrics** in the benchmark, and
make the gates relative to those. Today a coverage regression and a perception
regression are indistinguishable in the summary table, and small-object
regressions are invisible because they are averaged against furniture.

#### Resources

**R1 — Bake YOLO-World prompt embeddings at build time** (`set_classes()` then
`model.save()`), so the runtime never instantiates ultralytics' CLIP text tower.
Drops `clip-anytorch` and one entire ViT-B/32 (~600 MB fp32 host + its device copy).
Changing the vocabulary becomes a build step, which is also the right semantics
for a deployed robot.

**R2 — fp16 across the board** (YOLO, MobileSAM, CLIP) with an env escape hatch.
~2× on both memory and latency, on Orin especially.

**R3 — TensorRT engines for YOLO-World and the MobileSAM encoder**, built on first
boot into `/opt/models`, cached by device + TRT version. This is where the Orin
tick budget actually comes from.

**R4 — A `perception.profile` knob** (`full` / `jetson` / `lite`) selecting model
sizes (`yolov8l-world` → `yolov8s-world`), `imgsz`, tick period and detection
caps. Today every target gets the x86-5090 configuration.

**R5 — Vendor the ~8 concept-graphs functions we actually call** into
`scene_service/ingest/cg_kernels.py` and drop the `/opt/concept-graphs` dependency
along with `hydra-core`, `omegaconf`, `openai`, `wandb`, `rich`, `matplotlib`,
`seaborn`, `faiss-cpu`, `supervision`. We have already forked the hard part
(the overlap kernel); the remaining coupling is thin, and the image and RSS wins
are large.

**R6 — Drop Open3D.** Voxel downsample is `np.unique` on packed int64 voxel keys;
DBSCAN is `cKDTree` + union-find; the AABB is `min`/`max`. Biggest single RSS win
and it removes the most fragile aarch64 dependency.

**R7 — Vectorise and cache the overlap kernel.** Pack voxel indices into a single
int64 key, `np.unique` per cloud, cache the key array on the object with a version
counter, and intersect with sorted-array set ops (or one global `np.unique` +
sparse boolean product for the whole matrix). Two to three orders of magnitude,
and it stops allocating ~10⁶ Python tuples per tick.

**R8 — Frustum- and distance-gate the per-tick similarity.** Compare detections
only against map objects near the current view — we already compute visibility for
the dropout watchdog, so the information is in hand. In complete_apartment this is
165 → typically <20 candidates.

**R9 — Move periodic cleanup off the inference lock.** Compute the merge plan on a
snapshot in a background thread, apply it transactionally. Today a cleanup tick
stalls ingestion.

**R10 — fp16 safetensors in `/opt/models`** instead of fp32 `.bin`, and store
per-object clouds as contiguous float32 XYZ + uint8 RGB rather than Open3D
`PointCloud` objects (~3× smaller, and it is the representation the kernels in R7
want anyway).

**R11 — Land the `lite` profile from #156** — Scene with pose, rooms, annotations
and map binding and *zero* torch import. This is the one that turns "Scene won't
run on my Orin Nano" into "Scene runs, without object perception", which is a far
better answer than "buy a bigger Jetson".

---

### 5. Suggested sequencing

R7 + R8 first — pure wins, no behaviour change, and they make every subsequent
benchmark sweep cheaper. Then A1 + A2 together with one re-tune of
`merge_threshold`, which is the largest single accuracy step and also lets us
delete gate code rather than add it. Then R1–R4 as the Jetson enablement batch,
and R11 in parallel since it is independent. A3 after that, since it changes what
the label metric is measuring and I would rather not confound it with A1/A2. A5
and A6 can go any time and would make all of the above easier to evaluate.

One process note: the CI GPU capacity failure means these five worlds are WorkPC
captures rather than CI-reproduced. Before we tune thresholds against them again,
I would like the sweep to be reproducible in CI — the per-label margin table in
the current config is a sign that we have been fitting to unreproducible runs.
