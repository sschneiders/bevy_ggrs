# bevy_ggrs fork TODO

## Migration: Back to MultiThreaded Executor

**When:** After Bevy merges `schedule_v3` (per-system tracing in multi-threaded executor)

**Why:** We currently force `SingleThreaded` on GgrsSchedule and ReadInputs to get per-system `check_conditions` spans for PerfLayer profiling. The upcoming schedule_v3 emits `system` and `system_task` spans natively in the multi-threaded executor, making this workaround unnecessary.

**What to revert in `src/lib.rs`:**
```rust
// Remove these blocks:
.edit_schedule(ReadInputs, |schedule| {
    schedule.set_executor_kind(ExecutorKind::SingleThreaded);
})
// And inside .edit_schedule(GgrsSchedule, ...):
schedule.set_executor_kind(ExecutorKind::SingleThreaded);
```

**PerfLayer changes needed (Zomslop side):**
- Update `register_callsite` to accept `system` and `system_task` span names
- `system` spans wrap the actual system body → use directly for exec time
- `system_task` spans wrap the full async task (overhead + system) → ignore or treat as overhead
- Remove `check_conditions` gap computation logic; use `system` span duration directly
- Test with multi-threaded executor and verify tracking accuracy

**Tracking issues:**
- Bevy schedule_v3: https://github.com/bevyengine/bevy/blob/main/crates/bevy_ecs/src/schedule_v3/
- Related: https://github.com/bevyengine/bevy/issues/23450

**Current state:** SingleThreaded executor, 99.7% PerfLayer tracking via check_conditions gaps.
