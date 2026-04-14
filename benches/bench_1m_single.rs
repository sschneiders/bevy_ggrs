//! Minimal benchmark: 1M entities × 1 Clone component with 20 i32 fields.
//! Measures baseline snapshot overhead for a single large Clone component.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy_ggrs::prelude::*;
use bevy_ggrs::{AdvanceWorld, LoadWorld, RollbackFrameCount, SaveWorld, SnapshotPlugin};
use criterion::{Criterion, criterion_group, criterion_main};

#[derive(Component, Clone)]
struct BigData {
    f0: i32, f1: i32, f2: i32, f3: i32, f4: i32,
    f5: i32, f6: i32, f7: i32, f8: i32, f9: i32,
    f10: i32, f11: i32, f12: i32, f13: i32, f14: i32,
    f15: i32, f16: i32, f17: i32, f18: i32, f19: i32,
}

const ENTITY_COUNT: usize = 1_000_000;

fn spawn_entities(mut commands: Commands) {
    for i in 0..ENTITY_COUNT {
        let v = i as i32;
        commands.spawn((
            BigData {
                f0: v, f1: v, f2: v, f3: v, f4: v,
                f5: v, f6: v, f7: v, f8: v, f9: v,
                f10: v, f11: v, f12: v, f13: v, f14: v,
                f15: v, f16: v, f17: v, f18: v, f19: v,
            },
            Rollback,
        ));
    }
}

fn update_all(mut query: Query<&mut BigData>) {
    for mut d in &mut query {
        d.f0 += 1;
        d.f1 += 1;
        d.f2 += 1;
        d.f3 += 1;
        d.f4 += 1;
    }
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, SnapshotPlugin));
    app.rollback_component_with_clone::<BigData>();
    app
}

fn advance_and_save(app: &mut App) {
    app.world_mut().run_schedule(AdvanceWorld);
    app.world_mut().run_schedule(SaveWorld);
}

fn bench_save_all(c: &mut Criterion) {
    let mut app = build_app();
    app.add_systems(AdvanceWorld, update_all);
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);
    advance_and_save(&mut app);

    c.bench_function("1m_1component_20fields_all_mutation", |b| {
        b.iter(|| advance_and_save(&mut app))
    });
}

fn bench_save_none(c: &mut Criterion) {
    let mut app = build_app();
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);
    advance_and_save(&mut app);

    c.bench_function("1m_1component_20fields_no_mutation", |b| {
        b.iter(|| advance_and_save(&mut app))
    });
}

fn bench_load(c: &mut Criterion) {
    let mut app = build_app();
    app.add_systems(AdvanceWorld, update_all);
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);
    for _ in 0..5 { advance_and_save(&mut app); }

    c.bench_function("1m_1component_20fields_load", |b| {
        b.iter(|| {
            app.world_mut().run_schedule(AdvanceWorld);
            app.insert_resource(RollbackFrameCount(0));
            app.world_mut().run_schedule(LoadWorld);
        })
    });
}

fn bench_rollback_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("1m_1component_rollback_cycle");
    group.sample_size(30);

    for &depth in &[3, 7, 12] {
        group.bench_function(format!("depth_{depth}"), |b| {
            b.iter(|| {
                let mut app = build_app();
                app.add_systems(AdvanceWorld, update_all);
                app.update();
                app.world_mut().run_system_once(spawn_entities).unwrap();
                app.world_mut().run_schedule(SaveWorld);
                for _ in 0..depth { advance_and_save(&mut app); }
                app.insert_resource(RollbackFrameCount(0));
                app.world_mut().run_schedule(LoadWorld);
                for _ in 0..depth {
                    app.world_mut().run_schedule(AdvanceWorld);
                    app.world_mut().run_schedule(SaveWorld);
                }
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_save_all, bench_save_none, bench_load, bench_rollback_cycle
);
criterion_main!(benches);
