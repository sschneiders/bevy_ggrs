//! Snapshot and restore of [`Component`] data on rollback entities.
//!
//! [`ComponentSnapshotPlugin`] saves all instances of a component type each frame and
//! restores them during rollback using a configurable [`Strategy`].
//! [`ImmutableComponentSnapshotPlugin`] provides the same behaviour for components
//! marked `#[component(immutable)]`, which must be re-inserted rather than mutated in place.

use crate::{
    GgrsComponentSnapshot, GgrsComponentSnapshots, LoadWorld, LoadWorldSystems, RollbackFrameCount,
    RollbackId, SaveWorld, SaveWorldSystems, Strategy,
};
use bevy::{
    ecs::component::{Immutable, Mutable},
    prelude::*,
};
use std::marker::PhantomData;

/// A [`Plugin`] which manages snapshots for a [`Component`] using a provided [`Strategy`].
///
/// # Examples
/// ```rust
/// # use bevy::prelude::*;
/// # use bevy_ggrs::{prelude::*, ComponentSnapshotPlugin, CloneStrategy};
/// #
/// # const FPS: usize = 60;
/// #
/// # type MyInputType = u8;
/// #
/// # fn read_local_inputs() {}
/// #
/// # fn start(session: Session<GgrsConfig<MyInputType>>) {
/// # let mut app = App::new();
/// // The Transform component is a good candidate for Clone-based rollback
/// app.add_plugins(ComponentSnapshotPlugin::<CloneStrategy<Transform>>::default());
/// # }
/// ```
pub struct ComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    _phantom: PhantomData<S>,
}

impl<S> Default for ComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            _phantom: default(),
        }
    }
}

impl<S> ComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    /// Save system for types where `Stored: Clone`.
    ///
    /// Three paths:
    /// - 0 changed → Arc clone (refcount bump, O(1))
    /// - few changed (<50%) → Arc COW + patch k entries
    /// - many changed (>=50%) → full rebuild (cheaper than clone+patch)
    pub fn save_cloneable(
        mut snapshots: ResMut<GgrsComponentSnapshots<S::Target, S::Stored>>,
        frame: Res<RollbackFrameCount>,
        changed_query: Query<(&RollbackId, &S::Target), (With<RollbackId>, Changed<S::Target>)>,
        full_query: Query<(&RollbackId, &S::Target)>,
    ) where
        S::Stored: Clone,
    {
        let frame_val = frame.0;

        // Try incremental path when we have a previous snapshot
        if let Some(prev) = snapshots.peek_latest() {
            let prev_len = prev.len();
            let changes: Vec<_> = changed_query
                .iter()
                .map(|(&rollback, component)| (rollback, S::store(component)))
                .collect();

            if changes.is_empty() {
                // Nothing changed → Arc clone (refcount bump only)
                let reused = GgrsComponentSnapshot::share_arc_from(prev);
                snapshots.push(frame_val, reused);
                return;
            }

            // If most entities changed, full rebuild is cheaper than clone+patch
            // (avoids Arc overhead + n×binary_search in patch)
            if changes.len() * 2 >= prev_len {
                let components = full_query
                    .iter()
                    .map(|(&rollback, component)| (rollback, S::store(component)));
                let snapshot = GgrsComponentSnapshot::new(components);
                snapshots.push(frame_val, snapshot);
                return;
            }

            // Few changed → Arc clone + COW patch
            let mut snapshot = GgrsComponentSnapshot::share_arc_from(prev);
            snapshot.patch(changes);
            snapshots.push(frame_val, snapshot);
            return;
        }

        // No previous snapshot (first frame ever) → full rebuild
        let components = full_query
            .iter()
            .map(|(&rollback, component)| (rollback, S::store(component)));
        let snapshot = GgrsComponentSnapshot::new(components);
        snapshots.push(frame_val, snapshot);
    }

    /// Save system for types where `Stored` is NOT Clone (e.g. ReflectStrategy).
    /// Always rebuilds the full snapshot.
    pub fn save_reflect(
        mut snapshots: ResMut<GgrsComponentSnapshots<S::Target, S::Stored>>,
        frame: Res<RollbackFrameCount>,
        query: Query<(&RollbackId, &S::Target)>,
    ) {
        let components = query
            .iter()
            .map(|(&rollback, component)| (rollback, S::store(component)));
        let snapshot = GgrsComponentSnapshot::new(components);

        trace!(
            "Snapshot {} {} component(s)",
            snapshot.iter().count(),
            disqualified::ShortName::of::<S::Target>()
        );

        snapshots.push(frame.0, snapshot);
    }
}

impl<S> ComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component<Mutability = Mutable>,
    S::Stored: Send + Sync + 'static,
{
    /// System that restores the component to its snapshotted state for the target frame,
    /// inserting or removing it as required.
    pub fn load(
        mut commands: Commands,
        mut snapshots: ResMut<GgrsComponentSnapshots<S::Target, S::Stored>>,
        frame: Res<RollbackFrameCount>,
        mut query: Query<(Entity, &RollbackId, Option<&mut S::Target>)>,
    ) {
        let snapshot = snapshots.rollback(frame.0).get();

        for (entity, rollback, component) in query.iter_mut() {
            let snapshot = snapshot.get(rollback);

            match (component, snapshot) {
                (Some(mut component), Some(snapshot)) => S::update(component.as_mut(), snapshot),
                (Some(_), None) => {
                    commands.entity(entity).remove::<S::Target>();
                }
                (None, Some(snapshot)) => {
                    commands.entity(entity).insert(S::load(snapshot));
                }
                (None, None) => {}
            }
        }

        trace!(
            "Rolled back {} {} component(s)",
            snapshot.iter().count(),
            disqualified::ShortName::of::<S::Target>()
        );
    }
}

// --- Plugin for Cloneable stored types ---

impl<S> Plugin for ComponentSnapshotPlugin<S>
where
    S: Send + Sync + 'static + Strategy,
    S::Target: Component<Mutability = Mutable>,
    S::Stored: Send + Sync + Clone + 'static,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<GgrsComponentSnapshots<S::Target, S::Stored>>()
            .add_systems(
                SaveWorld,
                (
                    GgrsComponentSnapshots::<S::Target, S::Stored>::sync_depth,
                    GgrsComponentSnapshots::<S::Target, S::Stored>::discard_old_snapshots,
                    Self::save_cloneable,
                )
                    .chain()
                    .in_set(SaveWorldSystems::Snapshot),
            );
        app.add_systems(LoadWorld, Self::load.in_set(LoadWorldSystems::Data));
    }
}

// --- Plugin for non-Cloneable stored types (ReflectStrategy) ---

/// A separate plugin for non-Clone strategies (e.g. ReflectStrategy) that always rebuilds snapshots.
pub struct ComponentSnapshotReflectPlugin<S>(PhantomData<S>);

impl<S> Default for ComponentSnapshotReflectPlugin<S> {
    fn default() -> Self {
        Self(default())
    }
}

impl<S> Plugin for ComponentSnapshotReflectPlugin<S>
where
    S: Send + Sync + 'static + Strategy,
    S::Target: Component<Mutability = Mutable>,
    S::Stored: Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<GgrsComponentSnapshots<S::Target, S::Stored>>()
            .add_systems(
                SaveWorld,
                (
                    GgrsComponentSnapshots::<S::Target, S::Stored>::sync_depth,
                    GgrsComponentSnapshots::<S::Target, S::Stored>::discard_old_snapshots,
                    ComponentSnapshotPlugin::<S>::save_reflect,
                )
                    .chain()
                    .in_set(SaveWorldSystems::Snapshot),
            );
        app.add_systems(
            LoadWorld,
            ComponentSnapshotPlugin::<S>::load.in_set(LoadWorldSystems::Data),
        );
    }
}

/// A [`Plugin`] which manages snapshots for a [`Component`] using a provided [`Strategy`] that works with immutable components.
///
/// # Examples
/// ```rust
/// # use bevy::prelude::*;
/// # use bevy_ggrs::{prelude::*, ImmutableComponentSnapshotPlugin, CloneStrategy};
/// #
/// # fn start() {
/// # let mut app = App::new();
/// #[derive(Component, Clone)]
/// #[component(immutable)]
/// struct MyComponent(String);
///
/// app.add_plugins(ImmutableComponentSnapshotPlugin::<CloneStrategy<MyComponent>>::default());
/// # }
/// ```
pub struct ImmutableComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    _phantom: PhantomData<S>,
}

impl<S> Default for ImmutableComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            _phantom: default(),
        }
    }
}

impl<S> Plugin for ImmutableComponentSnapshotPlugin<S>
where
    S: Send + Sync + 'static + Strategy,
    S::Target: Component<Mutability = Immutable>,
    S::Stored: Send + Sync + Clone + 'static,
{
    /// Registers snapshot storage and the save/load systems for this immutable component type.
    fn build(&self, app: &mut App) {
        app.init_resource::<GgrsComponentSnapshots<S::Target, S::Stored>>()
            .add_systems(
                SaveWorld,
                (
                    GgrsComponentSnapshots::<S::Target, S::Stored>::sync_depth,
                    GgrsComponentSnapshots::<S::Target, S::Stored>::discard_old_snapshots,
                    ComponentSnapshotPlugin::<S>::save_cloneable,
                )
                    .chain()
                    .in_set(SaveWorldSystems::Snapshot),
            )
            .add_systems(LoadWorld, Self::load.in_set(LoadWorldSystems::Data));
    }
}

impl<S> ImmutableComponentSnapshotPlugin<S>
where
    S: Strategy,
    S::Target: Component,
    S::Stored: Send + Sync + 'static,
{
    /// System that restores this immutable component to its snapshotted state for the target frame,
    /// re-inserting it (triggering hooks) or removing it as required.
    pub fn load(
        mut commands: Commands,
        mut snapshots: ResMut<GgrsComponentSnapshots<S::Target, S::Stored>>,
        frame: Res<RollbackFrameCount>,
        mut query: Query<(Entity, &RollbackId, Has<S::Target>)>,
    ) {
        let snapshot = snapshots.rollback(frame.0).get();

        for (entity, rollback, has_component) in query.iter_mut() {
            let snapshot = snapshot.get(rollback);

            match (has_component, snapshot) {
                (true, None) => {
                    commands.entity(entity).remove::<S::Target>();
                }
                (_, Some(snapshot)) => {
                    commands.entity(entity).insert(S::load(snapshot));
                }
                (false, None) => {}
            }
        }

        trace!(
            "Rolled back {} {} component(s)",
            snapshot.iter().count(),
            disqualified::ShortName::of::<S::Target>()
        );
    }
}

/// A separate plugin for immutable components with non-Clone strategies (e.g. ReflectStrategy).
pub struct ImmutableComponentSnapshotReflectPlugin<S>(PhantomData<S>);

impl<S> Default for ImmutableComponentSnapshotReflectPlugin<S> {
    fn default() -> Self {
        Self(default())
    }
}

impl<S> Plugin for ImmutableComponentSnapshotReflectPlugin<S>
where
    S: Send + Sync + 'static + Strategy,
    S::Target: Component<Mutability = Immutable>,
    S::Stored: Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<GgrsComponentSnapshots<S::Target, S::Stored>>()
            .add_systems(
                SaveWorld,
                (
                    GgrsComponentSnapshots::<S::Target, S::Stored>::sync_depth,
                    GgrsComponentSnapshots::<S::Target, S::Stored>::discard_old_snapshots,
                    ComponentSnapshotPlugin::<S>::save_reflect,
                )
                    .chain()
                    .in_set(SaveWorldSystems::Snapshot),
            )
            .add_systems(
                LoadWorld,
                ImmutableComponentSnapshotPlugin::<S>::load.in_set(LoadWorldSystems::Data),
            );
    }
}
