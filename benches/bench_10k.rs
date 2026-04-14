//! Performance benchmark: 10k entities × 10 rollback components.
//!
//! Tests SaveWorld and LoadWorld performance with a realistic entity layout
//! where all 10 components live on every entity.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy_ggrs::prelude::*;
use bevy_ggrs::{AdvanceWorld, LoadWorld, RollbackFrameCount, SaveWorld, SnapshotPlugin};
use criterion::{Criterion, criterion_group, criterion_main};

// 10 Copy components (cheap to snapshot)
#[derive(Component, Clone, Copy)]
struct Pos(f32, f32);

#[derive(Component, Clone, Copy)]
struct Vel(f32, f32);

#[derive(Component, Clone, Copy)]
struct Health(i32);

#[derive(Component, Clone, Copy)]
struct Armor(i32);

#[derive(Component, Clone, Copy)]
struct Level(u16);

#[derive(Component, Clone, Copy)]
struct Flags(u8);

#[derive(Component, Clone, Copy)]
struct Timer(f32);

#[derive(Component, Clone, Copy)]
struct Energy(f32);

#[derive(Component, Clone, Copy)]
struct Ammo(i16);

#[derive(Component, Clone, Copy)]
struct Score(i64);

// 2 Clone components (heap-allocated — more expensive to snapshot)
#[derive(Component, Clone)]
struct Inventory(Vec<u8>);

#[derive(Component, Clone)]
struct NameTag(String);

const ENTITY_COUNT: usize = 10_000;

fn spawn_entities(mut commands: Commands) {
    for i in 0..ENTITY_COUNT {
        commands.spawn((
            // 10 Copy components
            Pos(i as f32, i as f32),
            Vel(1.0, 0.5),
            Health(100),
            Armor(50),
            Level(1),
            Flags(0),
            Timer(0.0),
            Energy(100.0),
            Ammo(30),
            Score(0),
            // 2 Clone components
            Inventory(vec![0; 8]),
            NameTag(format!("entity_{i}")),
            // Rollback marker
            Rollback,
        ));
    }
}

/// Mutate only a subset of components (simulates typical frame)
fn update_movement(mut pos: Query<&mut Pos>, mut vel: Query<&mut Vel>) {
    for mut p in &mut pos {
        p.0 += 1.0;
    }
    for mut v in &mut vel {
        v.1 += 0.01;
    }
}

/// Mutate ALL components (worst case — nothing skipped)
fn update_all(
    mut pos: Query<&mut Pos>,
    mut vel: Query<&mut Vel>,
    mut health: Query<&mut Health>,
    mut armor: Query<&mut Armor>,
    mut level: Query<&mut Level>,
    mut flags: Query<&mut Flags>,
    mut timer: Query<&mut Timer>,
    mut energy: Query<&mut Energy>,
    mut ammo: Query<&mut Ammo>,
    mut score: Query<&mut Score>,
) {
    for mut p in &mut pos { p.0 += 1.0; }
    for mut v in &mut vel { v.1 += 0.01; }
    for mut h in &mut health { h.0 -= 1; }
    for mut a in &mut armor { a.0 -= 1; }
    for mut l in &mut level { l.0 += 1; }
    for mut f in &mut flags { f.0 = f.0.wrapping_add(1); }
    for mut t in &mut timer { t.0 += 0.016; }
    for mut e in &mut energy { e.0 -= 0.1; }
    for mut a in &mut ammo { a.0 -= 1; }
    for mut s in &mut score { s.0 += 1; }
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, SnapshotPlugin));

    // Register 10 Copy components + 2 Clone components for rollback
    app.rollback_component_with_copy::<Pos>();
    app.rollback_component_with_copy::<Vel>();
    app.rollback_component_with_copy::<Health>();
    app.rollback_component_with_copy::<Armor>();
    app.rollback_component_with_copy::<Level>();
    app.rollback_component_with_copy::<Flags>();
    app.rollback_component_with_copy::<Timer>();
    app.rollback_component_with_copy::<Energy>();
    app.rollback_component_with_copy::<Ammo>();
    app.rollback_component_with_copy::<Score>();
    app.rollback_component_with_clone::<Inventory>();
    app.rollback_component_with_clone::<NameTag>();

    app
}

fn advance_and_save(app: &mut App) {
    app.world_mut().run_schedule(AdvanceWorld);
    app.world_mut().run_schedule(SaveWorld);
}

fn advance_and_load(app: &mut App) {
    app.world_mut().run_schedule(AdvanceWorld);
    app.insert_resource(RollbackFrameCount(0));
    app.world_mut().run_schedule(LoadWorld);
}

/// Benchmark: SaveWorld with partial mutation (only Pos + Vel change)
fn bench_save_partial_mutation(c: &mut Criterion) {
    let mut app = build_app();
    app.add_systems(AdvanceWorld, update_movement);
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);

    // Warm up one more frame so we have a previous snapshot for the fast path
    advance_and_save(&mut app);

    c.bench_function("save_10k_entities_12_components_partial_mutation", |b| {
        b.iter(|| advance_and_save(&mut app))
    });
}

/// Benchmark: SaveWorld with all components mutating (worst case)
fn bench_save_all_mutation(c: &mut Criterion) {
    let mut app = build_app();
    app.add_systems(AdvanceWorld, update_all);
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);

    advance_and_save(&mut app);

    c.bench_function("save_10k_entities_12_components_all_mutation", |b| {
        b.iter(|| advance_and_save(&mut app))
    });
}

/// Benchmark: SaveWorld with NO mutation (fast path — clone only)
fn bench_save_no_mutation(c: &mut Criterion) {
    let mut app = build_app();
    // No update systems — nothing ever changes
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);

    advance_and_save(&mut app);

    c.bench_function("save_10k_entities_12_components_no_mutation", |b| {
        b.iter(|| advance_and_save(&mut app))
    });
}

/// Benchmark: LoadWorld (rollback + restore)
fn bench_load(c: &mut Criterion) {
    let mut app = build_app();
    app.add_systems(AdvanceWorld, update_movement);
    app.update();
    app.world_mut().run_system_once(spawn_entities).unwrap();
    app.world_mut().run_schedule(SaveWorld);

    // Advance a few frames so there's something to roll back to
    for _ in 0..5 {
        advance_and_save(&mut app);
    }

    c.bench_function("load_10k_entities_12_components", |b| {
        b.iter(|| advance_and_load(&mut app))
    });
}

/// Benchmark: Full rollback cycle — save 7 frames, rollback to frame 0, resimulate to 7
/// This is the real-world scenario: load → advance → save → advance → save → ...
fn bench_rollback_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("rollback_cycle_10k_12_components");
    group.sample_size(30);

    // Test with different rollback depths
    for &depth in &[3, 7, 12] {
        group.bench_function(format!("depth_{depth}"), |b| {
            b.iter(|| {
                let mut app = build_app();
                app.add_systems(AdvanceWorld, update_movement);
                app.update();
                app.world_mut().run_system_once(spawn_entities).unwrap();
                app.world_mut().run_schedule(SaveWorld); // frame 0

                // Save `depth` frames forward
                for _ in 0..depth {
                    advance_and_save(&mut app);
                }

                // Rollback to frame 0
                app.insert_resource(RollbackFrameCount(0));
                app.world_mut().run_schedule(LoadWorld);

                // Resimulate `depth` frames forward
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
    targets = bench_save_no_mutation, bench_save_partial_mutation, bench_save_all_mutation, bench_load, bench_rollback_cycle
);
criterion_main!(benches);
