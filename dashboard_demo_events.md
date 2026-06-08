# Dashboard Demo Events - Wind Farm A

This note summarizes the five Wind Farm A events currently configured for the
Real-time Monitor demo in `src/config.py` under `REALTIME_REPRESENTATIVE_EVENTS`.
It is written for council/demo explanation: what the data represents, when each
event starts and ends, how long it lasts, and how to explain the 14-day Wind
Farm A labeling rule from the CARE-to-Compare paper.

## Sources Checked

- Dashboard selection: `src/config.py`
- Event metadata: `data/Wind Farm A/event_info.csv`
- Raw event files: `data/Wind Farm A/datasets/{22,40,68,24,14}.csv`
- Paper: `D:\Final Project\Paper\CARE to Compare A real-world dataset for anomaly.md`

## Dataset Context From CARE-to-Compare

The CARE-to-Compare paper frames this dataset as a public benchmark for
wind-turbine SCADA anomaly detection and predictive maintenance. For Wind Farm
A, the original source is the EDP open data and historical fault logbook. The
paper states that Wind Farm A consists of 5 turbines and 22 selected datasets.
Each CSV contains one year of training data and a shorter prediction/demo
period sampled every 10 minutes.

The important labeling rule is:

- Event-level label: a dataset is labeled `anomaly` if its prediction period
  contains an anomaly event; otherwise it is labeled `normal`.
- Event interval: every anomaly has an assigned start timestamp; the anomaly
  end is treated as the start of the real turbine fault.
- Wind Farm A caveat: the EDP logbook only gave fault start timestamps. The
  CARE-to-Compare authors therefore estimated possible anomaly starts by
  analyzing data before each fault. They explicitly warn that the true anomaly
  starts for Wind Farm A may differ from the assigned starts.
- Point-level status rule for Wind Farm A: for each turbine fault, the 14 days
  before the fault timestamp were marked as status-ID `4` (fault), and the 3
  days after the fault timestamp were marked as status-ID `3` (service mode).
  This was a conservative choice to avoid putting suspicious behavior into
  training data.

For the dashboard, this means the council should understand two related but
different ideas:

1. `event_start` to `event_end` is the event interval shown in
   `event_info.csv`.
2. The 14-day rule is the conservative point-level Wind Farm A status labeling
   rule around the real fault timestamp. In some event files, the visible
   prediction period does not include the full 14 days, or the event interval is
   longer than the 14-day status-ID `4` band.

## Current Demo Event Set

The dashboard now normalizes the displayed streaming `time_stamp` for each
selected event so every asset/event stream begins at `2023-01-01 00:00:00`.
This is only a demo-display shift. It preserves each event's internal cadence,
relative timing, ordering, duration, `sequence_id`, labels, and sensor values.
The original anonymized timestamps from the dataset are still shown below for
traceability.

| Demo order | Asset | Event ID | Dashboard label | Official event label | Official event description |
|---:|---:|---:|---|---|---|
| 1 | 21 | 22 | anomaly - hydraulic_group | anomaly | Hydraulic group |
| 2 | 10 | 40 | anomaly - gearbox bearing | anomaly | Generator bearing failure |
| 3 | 11 | 68 | anomaly - Transformer Failure | anomaly | Transformer failure |
| 4 | 0 | 24 | normal | normal | none |
| 5 | 13 | 14 | normal | normal | none |

Note: asset 10 / event 40 has a naming mismatch. The dashboard override uses
the requested demo label `gearbox bearing`, but the local official metadata says
`Generator bearing failure`. If council asks about traceability, the official
CSV metadata should be treated as the dataset source of truth.

## Event Timeline Summary

| Asset | Event ID | Label | Event start | Event end / fault timestamp | Event interval duration | Raw prediction coverage in CSV |
|---:|---:|---|---|---|---|---|
| 21 | 22 | anomaly | 2023-08-12 09:50:00 | 2023-08-19 10:00:00 | 7 days 0 hours 10 minutes | 2023-08-12 09:50:00 to 2023-08-20 09:50:00 |
| 10 | 40 | anomaly | 2022-12-26 00:00:00 | 2023-01-26 13:00:00 | 31 days 13 hours 0 minutes | 2022-12-25 00:00:00 to 2023-01-28 13:00:00 |
| 11 | 68 | anomaly | 2023-07-28 13:20:00 | 2023-08-11 13:10:00 | 13 days 23 hours 50 minutes | 2023-07-28 13:20:00 to 2023-08-13 13:20:00 |
| 0 | 24 | normal | 2023-04-27 15:00:00 | 2023-05-11 11:20:00 | 13 days 20 hours 20 minutes | 2023-04-24 15:00:00 to 2023-05-13 11:20:00 |
| 13 | 14 | normal | 2023-03-05 14:00:00 | 2023-03-12 18:40:00 | 7 days 4 hours 40 minutes | 2023-03-03 14:00:00 to 2023-03-16 18:40:00 |

For anomaly rows, `event_end` is best explained as the fault timestamp. The
period before it is the early-warning/anomaly development period. For normal
events, the start/end interval is a normal-behavior prediction window, not a
fault countdown.

## Point-Level Status Inside the Prediction Period

Status-ID meanings from the paper:

- `0`: normal operation
- `3`: service mode / service team at site
- `4`: asset is down due to fault or other reasons

| Asset | Event ID | Prediction rows | Status-ID counts in prediction | Status-ID `4` window | Status-ID `3` window |
|---:|---:|---:|---|---|---|
| 21 | 22 | 1,148 | `4`: 1,004; `3`: 144 | 2023-08-12 09:50:00 to 2023-08-19 09:40:00 | 2023-08-19 10:00:00 to 2023-08-20 09:50:00 |
| 10 | 40 | 4,939 | `0`: 2,666; `4`: 1,985; `3`: 288 | 2023-01-12 13:00:00 to 2023-01-26 12:50:00 | 2023-01-26 13:00:00 to 2023-01-28 13:00:00 |
| 11 | 68 | 2,295 | `4`: 2,014; `3`: 281 | 2023-07-28 13:20:00 to 2023-08-11 13:10:00 | 2023-08-11 14:40:00 to 2023-08-13 13:20:00 |
| 0 | 24 | 2,714 | `0`: 2,714 | none | none |
| 13 | 14 | 1,901 | `0`: 1,901 | none | none |

This is useful for the live demo:

- Asset 21 / event 22 and asset 11 / event 68 are clean anomaly demo cases:
  the prediction period is almost entirely fault-status or service-status
  data, with the fault timestamp at the event end.
- Asset 10 / event 40 is a longer anomaly event. The event interval begins on
  2022-12-26, but the conservative Wind Farm A status-ID `4` band begins on
  2023-01-12 13:00, exactly 14 days before the real fault timestamp
  2023-01-26 13:00. This makes it a good example to explain the paper's
  distinction between event-level anomaly timing and point-level 14-day
  fault-status labeling.
- Asset 0 / event 24 and asset 13 / event 14 are normal comparison windows:
  their prediction periods contain only status-ID `0`, so they are suitable for
  showing that the dashboard does not only stream fault cases.

## Per-Event Demo Notes

### Asset 21 - Event 22 - Hydraulic Group Anomaly

- Configured dashboard role: anomaly case.
- Official metadata: Hydraulic group.
- Event interval: 2023-08-12 09:50:00 to 2023-08-19 10:00:00.
- Duration: 7 days 0 hours 10 minutes.
- Interpretation: this is a hydraulic-group fault case. The prediction stream
  starts directly at the anomaly interval and continues about one day into
  service mode after the fault timestamp. Use this case to show a compact,
  obvious anomaly episode.

### Asset 10 - Event 40 - Bearing-Related Anomaly

- Configured dashboard role: anomaly case.
- Dashboard label: gearbox bearing.
- Official metadata: Generator bearing failure.
- Event interval: 2022-12-26 00:00:00 to 2023-01-26 13:00:00.
- Duration: 31 days 13 hours 0 minutes.
- Status-ID `4` fault band: 2023-01-12 13:00:00 to 2023-01-26 12:50:00.
- Interpretation: this is the clearest demo case for explaining the 14-day
  Wind Farm A status rule. The broader event interval is over one month, but
  the point-level fault-status band starts 14 days before the real fault
  timestamp. In council discussion, say that the paper uses conservative
  status labels around the fault because the original EDP logbook did not give
  detailed anomaly onset times.

### Asset 11 - Event 68 - Transformer Failure

- Configured dashboard role: anomaly case.
- Official metadata: Transformer failure.
- Event interval: 2023-07-28 13:20:00 to 2023-08-11 13:10:00.
- Duration: 13 days 23 hours 50 minutes.
- Interpretation: this is almost exactly a 14-day pre-fault anomaly window.
  It is the most direct example for explaining "the anomaly period is about
  two weeks before the fault." The prediction stream then continues into
  service mode after the fault timestamp.

### Asset 0 - Event 24 - Normal Behavior

- Configured dashboard role: normal comparison case.
- Official metadata: normal.
- Event interval: 2023-04-27 15:00:00 to 2023-05-11 11:20:00.
- Duration: 13 days 20 hours 20 minutes.
- Prediction status: all status-ID `0`.
- Interpretation: this is a clean normal window. Use it to show stable turbine
  behavior and to demonstrate that alerts are not expected for every stream.

### Asset 13 - Event 14 - Normal Behavior

- Configured dashboard role: normal comparison case.
- Official metadata: normal.
- Event interval: 2023-03-05 14:00:00 to 2023-03-12 18:40:00.
- Duration: 7 days 4 hours 40 minutes.
- Prediction status: all status-ID `0`.
- Interpretation: this is the second normal comparison window. It gives the
  dashboard a mixed fleet story: three fault/anomaly turbines and two normal
  turbines at the same time.

## Suggested Council Explanation

Use this wording during the demo:

> Wind Farm A comes from the CARE-to-Compare benchmark, which adapts the EDP
> SCADA and fault-logbook data into event-level anomaly/normal datasets. For
> Wind Farm A, the original logbook only gave fault start timestamps, so the
> authors estimated anomaly event starts from the data and used a conservative
> status rule: the 14 days before a fault are marked as fault-related status,
> and the following service period is marked separately. In our dashboard demo,
> we selected three anomaly events and two normal events so the council can see
> both failure-development behavior and normal turbine behavior in one live
> monitor.
