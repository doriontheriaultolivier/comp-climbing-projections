# Persistent performance-surprise projection

## Decision question

Can repeated gaps between an athlete's pre-competition Elo and the
Performance-Elo shown in that competition identify a real change in ability
early enough to improve the next forecast?

**Current answer: probably yes, but as a separate next-event projection layer
until prospective validation is complete.** The stable cumulative Elo remains
the long-term reference.

## Coach-readable model

1. Freeze the athlete's rating before the competition.
2. Calculate the Performance-Elo shown in each round.
3. Collapse all rounds in the same competition into one quality-weighted,
   robust observation. A qualification, semifinal and final from one event are
   evidence about one performance, not three independent confirmations.
4. Compare that event performance with the frozen rating. This gap is the
   **performance surprise**.
5. Increase the next-event form adjustment only when independent competitions
   repeatedly point in the same direction. The default persistence rule to
   challenge is two of the latest three eligible competitions, with stronger
   shrinkage when evidence is weak.
6. Decay the adjustment with time and show it separately from stable Elo.

This supports a coach's interpretation—new environment, changed training,
health, confidence or terrain fit—without claiming that the results identify
which cause produced the change.

## Mathematical contract

For athlete `i` at competition `t`:

`surprise(i,t) = robust_event_Performance-Elo - frozen_pre-event_Elo`

The observation is clipped, weighted for evidence quality and divided by its
expected noise. A latent recent-form state is updated only with competitions
strictly earlier than the forecast date. A changepoint probability or an
equivalent persistence score governs how much process uncertainty increases.

`next-event projection = stable Elo + shrunk persistent-form adjustment`

The stable Elo ledger remains zero-sum. For event simulation, form adjustments
are recentered across the entered field; subtracting the field mean preserves
relative differences and makes the added projection layer zero-sum within that
forecast. If adaptive sensitivity is ever applied directly to the cumulative
ledger, every pair must use a shared symmetric gain or the full event delta
must be recentered. Athlete-specific one-sided K factors are prohibited.

## Existing empirical evidence

Two leakage-safe research challengers already test most of this idea:

- The 180-day rating-lag challenger improved 2025+ Boulder ranking similarity
  from 0.756 to 0.765 for men and from 0.807 to 0.830 for women. Its probability
  errors also improved across the Boulder women's Top-24/16/8/3 outcomes and
  across most men's outcomes.
- The athlete-volatility latent-skill challenger improved overall next-event
  rank correlation from 0.635 to 0.679 and mean Top-24/16/8/3 Brier error by
  about 3.9%. It centers event updates before carrying them forward.

The strict robustness audit did **not** approve production promotion because
one athlete contributed more than the declared 10% limit of total positive
gain. This is the clearest reason to add a multiple-competition persistence
gate and prospective shadow monitoring rather than immediately replacing Elo.

## Promotion test

Tune only on 2022–2024, evaluate once on 2025 onward, and freeze every forecast
before the predicted competition. Report by discipline and gender:

- rank correlation and placement error;
- Top-24, Top-16, Top-8, Top-3 and win probability calibration;
- improvement with the leading athlete and leading competition removed;
- calibration for new, stable, improving, regressing and inactive athletes;
- sensitivity to one-event, two-event and two-of-three persistence rules.

Production promotion requires positive competition-clustered uncertainty
bounds, no material pool regression and no single athlete dominating the gain.

