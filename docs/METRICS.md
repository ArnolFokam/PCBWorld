# Metrics — how a routed board is scored

Every method is scored the same way on the same boards, by KiCad's own design-rule checker.
This page says what the reported numbers mean. The per-column reference and the CLI are in
[eval/README.md](../eval/README.md). The formal definitions are in the paper
([PCBWorld.pdf](../PCBWorld.pdf), §4.3 and Appendix I).

## The protocol

A method routes each board five times. The rollout with the largest potential gain is kept,
and every number below is read off that one rollout, then averaged over boards. The paper
writes this as @5; @1 means a single rollout with no selection.

Two metrics ignore the selection. Time averages over all five rollouts, so a method that
gives up early on a hard board is still charged for what it spent. Parse-fail also reads all
five. Deterministic baselines are run once, where @5 and @1 coincide.

## The numbers

| Metric | Want | Is |
|---|---|---|
| Clean pass (CP) | ↑ | fraction of boards that came back fully connected **and** with zero DRC errors |
| Potential gain (Pot.) | ↑ | Φ(routed) − Φ(bare board), the reward potential of [docs/ENVIRONMENT.md](ENVIRONMENT.md) |
| Routability (Rout.) | ↑ | fraction of the required connections that got routed: 0 on the bare board, 1 when every net is connected |
| DRV | ↓ | error-level design-rule violations left on the board |
| Wirelength (WL) | ↓ | total routed copper, in mm |
| Via | ↓ | vias placed |
| Time | ↓ | wallclock seconds per rollout, from the first engine call to the end of the episode |
| Parse-fail | ↓ | LLM agents only: fraction of boards where no rollout loads as a valid `.kicad_pcb` |

**Clean pass is the headline.** Routability alone says the nets are joined, not that the board
could be manufactured. CP demands both, so it is the closest thing here to ready-to-fabricate.

**Potential gain is the quality measure, and it also does the selecting.** It is a single
number over violations, wirelength and via count, which is why one rollout per board can be
picked by it and the rest dropped.

Wirelength and via count are secondary. A method that shortens copper by leaving nets
unrouted looks good on them and bad on everything else, so read them next to Rout. and CP.

## One difference worth knowing

DRV and clean pass count **error-level** violations only. The training potential counts
errors plus three promoted warnings: dangling track, dangling via, and net conflict. The
difference is deliberate, not a bug, and it means a board can score a clean pass while still
carrying promoted warnings. Details in
[eval/README.md](../eval/README.md) under "Reading the DRV breakdown".

## Where the numbers come from

One pipeline scores every method, in three stages
([eval/pipeline.py](../eval/pipeline.py)):

1. **rollout** — the method routes the boards and saves each result as a `.kicad_pcb`.
2. **DRC** — each saved board is re-opened and checked against its own project design rules,
   not a loosened template.
3. **aggregate** — the per-rollout rows are selected and averaged into per-board numbers.

Because stage 2 reads saved boards, a method never scores itself. Anything that produces a
routed `.kicad_pcb` can be dropped into stage 1 and compared on the same terms. Running it is
in [docs/QUICKSTART.md](QUICKSTART.md).
