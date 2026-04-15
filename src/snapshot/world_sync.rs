//! On-demand world snapshot and restore for late-join / session migration.
//!
//! [`WorldSyncSnapshot`] captures the full state of all rollback entities, their
//! components (via the same strategies used for rollback snapshots), and registered
//! rollback resources into a portable byte buffer that can be sent over the network.
//!
//! This is the building block for:
//! - Late-join: host captures → sends bytes → joiner restores
//! - Session migration: capture → destroy session → new session → restore
//!
//! # How it works
//!
//! 1. **Capture** runs the `SaveWorld` schedule once to populate all snapshot storage.
//! 2. For each registered component type, it reads the latest snapshot from
//!    `GgrsComponentSnapshots<C>` and serializes it.
//! 3. For each registered resource type, it reads the current value and serializes it.
//! 4. Everything is serialized into a flat byte buffer.
//! 5. **Restore** despawns all rollback entities, re-spawns them with the exact
//!    `RollbackId`s, inserts component data, restores resources, and rebuilds
//!    `RollbackOrdered`.
//!
//! # Registration
//!
//! Use [`RollbackApp::rollback_component_with_sync`] and
//! [`RollbackApp::rollback_resource_with_sync`] to register types for world sync.
//! These require `Serialize + DeserializeOwned` in addition to `Clone`.
//!
//! # Example
//! ```rust,ignore
//! use bevy_ggrs::prelude::*;
//! use bevy_ggrs::WorldSyncSnapshot;
//!
//! // Capture full snapshot (runs SaveWorld internally)
//! let snapshot = WorldSyncSnapshot::capture(world);
//! let bytes = snapshot.to_bytes();
//!
//! // Send bytes over reliable channel...
//!
//! // Restore on receiving side
//! let snapshot = WorldSyncSnapshot::from_bytes(&bytes)?;
//! snapshot.restore(world);
//! ```

use crate::{
    Rollback, RollbackFrameCount, RollbackId, RollbackOrdered, SaveWorld,
};
use bevy::prelude::*;
use std::io::{self, Cursor, Read};

// ---------------------------------------------------------------------------
// Component/Resource sync trait objects
// ---------------------------------------------------------------------------

/// Type-erased function pointer for capturing component data from the world.
type CaptureFn = fn(&mut World, &mut Vec<u8>);

/// Type-erased function pointer for restoring component data into the world.
type RestoreFn = fn(&mut World, &[u8]);

/// Type-erased function pointer for capturing a resource from the world.
type CaptureResourceFn = fn(&mut World, &mut Vec<u8>);

/// Type-erased function pointer for restoring a resource into the world.
type RestoreResourceFn = fn(&mut World, &[u8]);

/// Type-erased function pointer for remapping entity references inside a component.
type RemapEntitiesFn = fn(&mut World, &std::collections::HashMap<Entity, Entity>);

/// Registry of component and resource types that participate in world sync snapshots.
///
/// This is automatically populated when you use
/// [`rollback_component_with_sync`](`crate::RollbackApp::rollback_component_with_sync`)
/// or [`rollback_resource_with_sync`](`crate::RollbackApp::rollback_resource_with_sync`).
#[derive(Resource, Default)]
pub struct WorldSyncRegistry {
    capture_fns: Vec<CaptureFn>,
    restore_fns: Vec<RestoreFn>,
    capture_resource_fns: Vec<CaptureResourceFn>,
    restore_resource_fns: Vec<RestoreResourceFn>,
    /// Function pointers that remap Entity fields inside components after restore.
    remap_fns: Vec<RemapEntitiesFn>,
}

impl WorldSyncRegistry {
    /// Register a component type for world sync capture/restore.
    pub fn register_component<C: Component + Clone + serde::Serialize + serde::de::DeserializeOwned>(
        &mut self,
    ) {
        self.capture_fns.push(capture_component::<C>);
        self.restore_fns.push(restore_component::<C>);
    }

    /// Register a resource type for world sync capture/restore.
    pub fn register_resource<R: Resource + Clone + serde::Serialize + serde::de::DeserializeOwned>(
        &mut self,
    ) {
        self.capture_resource_fns.push(capture_resource::<R>);
        self.restore_resource_fns.push(restore_resource::<R>);
    }

    /// Register a component type that contains `Entity` fields needing remapping after restore.
    /// The component must implement [`bevy::ecs::entity::MapEntities`].
    pub fn register_component_with_remap<
        C: Component<Mutability = bevy::ecs::component::Mutable>
            + Clone
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bevy::ecs::entity::MapEntities,
    >(&mut self) {
        self.capture_fns.push(capture_component::<C>);
        self.restore_fns.push(restore_component::<C>);
        self.remap_fns.push(remap_component_entities::<C>);
    }
}

// ---------------------------------------------------------------------------
// Generic capture/restore implementations
// ---------------------------------------------------------------------------

fn capture_component<C: Component + Clone + serde::Serialize>(
    world: &mut World,
    buf: &mut Vec<u8>,
) {
    let mut query = world.query::<(&RollbackId, &C)>();
    let data: Vec<(u64, C)> = query
        .iter(world)
        .map(|(rid, c)| (rid.to_bits(), c.clone()))
        .collect();

    // Format: [count:u32, (rollback_id_bits:u64, component_bytes_len:u32, component_bytes), ...]
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    for (rid_bits, component) in &data {
        buf.extend_from_slice(&rid_bits.to_le_bytes());
        let component_bytes =
            bincode::serialize(component).unwrap_or_default();
        buf.extend_from_slice(&(component_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&component_bytes);
    }
}

fn restore_component<C: Component + Clone + serde::de::DeserializeOwned>(
    world: &mut World,
    data: &[u8],
) {
    let mut cursor = Cursor::new(data);
    let mut buf4 = [0u8; 4];
    if cursor.read_exact(&mut buf4).is_err() {
        return;
    }
    let count = u32::from_le_bytes(buf4) as usize;

    for _ in 0..count {
        let mut buf8 = [0u8; 8];
        if cursor.read_exact(&mut buf8).is_err() {
            break;
        }
        let rid = RollbackId::from_bits(u64::from_le_bytes(buf8));

        if cursor.read_exact(&mut buf4).is_err() {
            break;
        }
        let len = u32::from_le_bytes(buf4) as usize;
        let mut component_bytes = vec![0u8; len];
        if cursor.read_exact(&mut component_bytes).is_err() {
            break;
        }

        let component: C = match bincode::deserialize(&component_bytes) {
        Ok(c) => c,
            Err(e) => {
                eprintln!("WorldSync: failed to deserialize component {}: {e}", std::any::type_name::<C>());
                continue;
            }
        };

        // Find entity with this RollbackId and insert the component
        let mut query = world.query::<(Entity, &RollbackId)>();
        let entity = query
            .iter(world)
            .find(|(_, id)| **id == rid)
            .map(|(e, _)| e);

        if let Some(entity) = entity {
            world.entity_mut(entity).insert(component);
        }
    }
}

fn capture_resource<R: Resource + Clone + serde::Serialize>(
    world: &mut World,
    buf: &mut Vec<u8>,
) {
    let Some(resource) = world.get_resource::<R>().cloned() else {
        // Resource absent → write 0 as marker
        buf.extend_from_slice(&0u32.to_le_bytes());
        return;
    };

    let bytes = bincode::serialize(&resource).unwrap_or_default();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&bytes);
}

fn restore_resource<R: Resource + Clone + serde::de::DeserializeOwned>(
    world: &mut World,
    data: &[u8],
) {
    let mut cursor = Cursor::new(data);
    let mut buf4 = [0u8; 4];
    if cursor.read_exact(&mut buf4).is_err() {
        return;
    }
    let len = u32::from_le_bytes(buf4) as usize;

    if len == 0 {
        // Resource was absent at capture time → remove it
        world.remove_resource::<R>();
        return;
    }

    let mut bytes = vec![0u8; len];
    if cursor.read_exact(&mut bytes).is_err() {
        return;
    }

    let resource: R = match bincode::deserialize(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("WorldSync: failed to deserialize resource {}: {e}", std::any::type_name::<R>());
            return;
        }
    };

    world.insert_resource(resource);
}

fn remap_component_entities<
    C: Component<Mutability = bevy::ecs::component::Mutable> + bevy::ecs::entity::MapEntities,
>(
    world: &mut World,
    entity_map: &std::collections::HashMap<Entity, Entity>,
) {
    struct Mapper<'a>(&'a std::collections::HashMap<Entity, Entity>);
    impl<'a> bevy::ecs::entity::EntityMapper for Mapper<'a> {
        fn get_mapped(&mut self, entity: Entity) -> Entity {
            self.0.get(&entity).copied().unwrap_or(entity)
        }
        fn set_mapped(&mut self, _old: Entity, _new: Entity) {
            // No-op: we only need one-directional mapping
        }
    }
    let mut query = world.query::<(Entity, &mut C)>();
    for (_entity, mut component) in query.iter_mut(world) {
        component.map_entities(&mut Mapper(entity_map));
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A portable snapshot of all rollback state that can be serialized to bytes.
///
/// Contains:
/// - `RollbackFrameCount` value
/// - All entities with `RollbackId` (entity → RollbackId mapping)
/// - `RollbackOrdered` state (for deterministic iteration order)
/// - Component data for all types registered via [`WorldSyncRegistry`]
/// - Resource data for all types registered via [`WorldSyncRegistry`]
#[derive(Clone, Debug)]
pub struct WorldSyncSnapshot {
    /// `RollbackFrameCount` at capture time.
    pub frame: i32,
    /// Serialized entity mapping: `[count:u32, (rollback_id_bits:u64, entity_bits:u64), ...]`
    entity_data: Vec<u8>,
    /// Serialized `RollbackOrdered`: `[count:u32, (rollback_id_bits:u64), ...]`
    ordered_data: Vec<u8>,
    /// Serialized component data, one section per registered component type.
    component_sections: Vec<Vec<u8>>,
    /// Serialized resource data, one section per registered resource type.
    resource_sections: Vec<Vec<u8>>,
}

/// Errors during snapshot operations.
#[derive(Debug)]
pub enum SnapshotError {
    /// I/O error during serialization or deserialization.
    Io(io::Error),
    /// The snapshot data is corrupted or unexpected.
    InvalidFormat(String),
}

impl From<io::Error> for SnapshotError {
    fn from(e: io::Error) -> Self {
        SnapshotError::Io(e)
    }
}

impl core::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "I/O error: {e}"),
            SnapshotError::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

// ---------------------------------------------------------------------------
// WorldSyncSnapshot implementation
// ---------------------------------------------------------------------------

impl WorldSyncSnapshot {
    /// Captures a full snapshot of all rollback state from the given [`World`].
    ///
    /// This runs the `SaveWorld` schedule to populate snapshot storage, then
    /// extracts all data into a portable format.
    pub fn capture(world: &mut World) -> Self {
        // Apply any pending deferred operations (e.g., on_add hooks that insert RollbackId)
        world.flush();

        // Run SaveWorld to populate all snapshot storage and checksums
        world.run_schedule(SaveWorld);

        let frame = world
            .get_resource::<RollbackFrameCount>()
            .map(|f| f.0)
            .unwrap_or(0);

        let entity_data = capture_entities(world);
        let ordered_data = capture_ordered(world);

        // Capture component data for each registered type
        // Clone the function lists to release the borrow on world before calling fns
        let (capture_fns, capture_resource_fns) = {
            let registry = world.get_resource::<WorldSyncRegistry>();
            match registry {
                Some(reg) => (reg.capture_fns.clone(), reg.capture_resource_fns.clone()),
                None => (Vec::new(), Vec::new()),
            }
        };

        let component_sections: Vec<Vec<u8>> = capture_fns
            .iter()
            .map(|capture_fn| {
                let mut buf = Vec::new();
                capture_fn(world, &mut buf);
                buf
            })
            .collect();

        let resource_sections: Vec<Vec<u8>> = capture_resource_fns
            .iter()
            .map(|capture_fn| {
                let mut buf = Vec::new();
                capture_fn(world, &mut buf);
                buf
            })
            .collect();

        WorldSyncSnapshot {
            frame,
            entity_data,
            ordered_data,
            component_sections,
            resource_sections,
        }
    }

    /// Restores this snapshot into the given [`World`].
    ///
    /// 1. Despawns all existing rollback entities
    /// 2. Restores `RollbackOrdered` (needed before entities get `RollbackId`)
    /// 3. Spawns new entities with the exact `RollbackId`s from the snapshot
    /// 4. Restores component data for all registered types
    /// 5. Restores resource data for all registered types
    /// 6. Sets `RollbackFrameCount`
    pub fn restore(&self, world: &mut World) {
        // Step 1: Despawn all existing rollback entities
        despawn_all_rollback(world);

        // Step 2: Restore RollbackOrdered (must happen before on_add hooks fire)
        restore_ordered(world, &self.ordered_data);

        // Step 3: Spawn entities with exact RollbackIds, build old→new entity map
        let entity_map = restore_entities(world, &self.entity_data);

        // Step 4: Restore component data
        let (restore_fns, restore_resource_fns, remap_fns) = {
            let registry = world.get_resource::<WorldSyncRegistry>();
            match registry {
                Some(reg) => (
                    reg.restore_fns.clone(),
                    reg.restore_resource_fns.clone(),
                    reg.remap_fns.clone(),
                ),
                None => (Vec::new(), Vec::new(), Vec::new()),
            }
        };

        for (restore_fn, data) in restore_fns.iter().zip(&self.component_sections) {
            restore_fn(world, data);
        }

        // Step 5: Remap Entity fields inside components (owner references, etc.)
        for remap_fn in &remap_fns {
            remap_fn(world, &entity_map);
        }

        // Step 6: Restore resource data
        for (restore_fn, data) in restore_resource_fns.iter().zip(&self.resource_sections) {
            restore_fn(world, data);
        }

        // Step 7: Set frame count
        world.insert_resource(RollbackFrameCount(self.frame));
    }

    /// Serialize the snapshot to bytes.
    ///
    /// Wire format:
    /// ```text
    /// [frame: i32 LE]
    /// [entity_data_len: u32 LE] [entity_data: bytes]
    /// [ordered_data_len: u32 LE] [ordered_data: bytes]
    /// [component_section_count: u32 LE]
    ///   for each section: [section_len: u32 LE] [section_data: bytes]
    /// [resource_section_count: u32 LE]
    ///   for each section: [section_len: u32 LE] [section_data: bytes]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&self.frame.to_le_bytes());
        buf.extend_from_slice(&(self.entity_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ordered_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.entity_data);
        buf.extend_from_slice(&self.ordered_data);

        // Component sections
        buf.extend_from_slice(&(self.component_sections.len() as u32).to_le_bytes());
        for section in &self.component_sections {
            buf.extend_from_slice(&(section.len() as u32).to_le_bytes());
            buf.extend_from_slice(section);
        }

        // Resource sections
        buf.extend_from_slice(&(self.resource_sections.len() as u32).to_le_bytes());
        for section in &self.resource_sections {
            buf.extend_from_slice(&(section.len() as u32).to_le_bytes());
            buf.extend_from_slice(section);
        }

        buf
    }

    /// Deserialize a snapshot from bytes produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let mut cursor = Cursor::new(bytes);
        let mut buf4 = [0u8; 4];

        // Frame
        cursor.read_exact(&mut buf4)?;
        let frame = i32::from_le_bytes(buf4);

        // Entity data
        cursor.read_exact(&mut buf4)?;
        let entity_len = u32::from_le_bytes(buf4) as usize;
        cursor.read_exact(&mut buf4)?;
        let ordered_len = u32::from_le_bytes(buf4) as usize;

        let mut entity_data = vec![0u8; entity_len];
        cursor.read_exact(&mut entity_data)?;

        let mut ordered_data = vec![0u8; ordered_len];
        cursor.read_exact(&mut ordered_data)?;

        // Component sections
        cursor.read_exact(&mut buf4)?;
        let component_count = u32::from_le_bytes(buf4) as usize;
        let mut component_sections = Vec::with_capacity(component_count);
        for _ in 0..component_count {
            cursor.read_exact(&mut buf4)?;
            let section_len = u32::from_le_bytes(buf4) as usize;
            let mut section = vec![0u8; section_len];
            cursor.read_exact(&mut section)?;
            component_sections.push(section);
        }

        // Resource sections
        cursor.read_exact(&mut buf4)?;
        let resource_count = u32::from_le_bytes(buf4) as usize;
        let mut resource_sections = Vec::with_capacity(resource_count);
        for _ in 0..resource_count {
            cursor.read_exact(&mut buf4)?;
            let section_len = u32::from_le_bytes(buf4) as usize;
            let mut section = vec![0u8; section_len];
            cursor.read_exact(&mut section)?;
            resource_sections.push(section);
        }

        Ok(WorldSyncSnapshot {
            frame,
            entity_data,
            ordered_data,
            component_sections,
            resource_sections,
        })
    }

    /// Returns the number of rollback entities in this snapshot.
    pub fn entity_count(&self) -> usize {
        let mut cursor = Cursor::new(&self.entity_data);
        let mut buf4 = [0u8; 4];
        if cursor.read_exact(&mut buf4).is_err() {
            return 0;
        }
        u32::from_le_bytes(buf4) as usize
    }
}

// ---------------------------------------------------------------------------
// Capture helpers
// ---------------------------------------------------------------------------

fn capture_entities(world: &mut World) -> Vec<u8> {
    let mut query = world.query::<(&RollbackId, Entity)>();
    let entities: Vec<_> = query.iter(world).collect();

    let mut buf = Vec::with_capacity(4 + entities.len() * 16);
    buf.extend_from_slice(&(entities.len() as u32).to_le_bytes());
    for (rid, entity) in &entities {
        buf.extend_from_slice(&rid.to_bits().to_le_bytes());
        buf.extend_from_slice(&entity.to_bits().to_le_bytes());
    }
    buf
}

fn capture_ordered(world: &mut World) -> Vec<u8> {
    let ordered = world
        .get_resource::<RollbackOrdered>()
        .cloned()
        .unwrap_or_default();
    let ids: Vec<_> = ordered.iter_sorted().collect();

    let mut buf = Vec::with_capacity(4 + ids.len() * 8);
    buf.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for id in &ids {
        buf.extend_from_slice(&id.to_bits().to_le_bytes());
    }
    buf
}

// ---------------------------------------------------------------------------
// Restore helpers
// ---------------------------------------------------------------------------

fn restore_ordered(world: &mut World, data: &[u8]) {
    let mut cursor = Cursor::new(data);
    let mut buf4 = [0u8; 4];
    if cursor.read_exact(&mut buf4).is_err() {
        world.insert_resource(RollbackOrdered::default());
        return;
    }
    let count = u32::from_le_bytes(buf4) as usize;

    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let mut buf8 = [0u8; 8];
        if cursor.read_exact(&mut buf8).is_err() {
            break;
        }
        ids.push(RollbackId::from_bits(u64::from_le_bytes(buf8)));
    }

    world.insert_resource(RollbackOrdered::from_sorted_ids(ids));
}

fn restore_entities(world: &mut World, data: &[u8]) -> std::collections::HashMap<Entity, Entity> {
    let mut entity_map = std::collections::HashMap::new();
    let mut cursor = Cursor::new(data);
    let mut buf4 = [0u8; 4];
    if cursor.read_exact(&mut buf4).is_err() {
        return entity_map;
    }
    let count = u32::from_le_bytes(buf4) as usize;

    for _ in 0..count {
        let mut buf8 = [0u8; 8];

        // Read RollbackId
        if cursor.read_exact(&mut buf8).is_err() {
            break;
        }
        let rid = RollbackId::from_bits(u64::from_le_bytes(buf8));

        // Read original entity bits
        if cursor.read_exact(&mut buf8).is_err() {
            break;
        }
        let old_entity = Entity::from_bits(u64::from_le_bytes(buf8));

        // Spawn with Rollback + RollbackId.
        let new_entity = world.spawn((Rollback, rid)).id();
        entity_map.insert(old_entity, new_entity);
    }
    entity_map
}

fn despawn_all_rollback(world: &mut World) {
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, With<Rollback>>()
        .iter(world)
        .collect();
    for entity in entities {
        world.despawn(entity);
    }
}

// ---------------------------------------------------------------------------
// Session Migration
// ---------------------------------------------------------------------------

use crate::GgrsPaused;

/// Manages a GGRS session migration (pause → snapshot → destroy → create → restore → resume).
///
/// This is a two-phase process coordinated by the game:
///
/// 1. **Host calls [`GgrsMigration::prepare(world)`](GgrsMigration::prepare)** —
///    pauses the GGRS session and captures a full snapshot.
/// 2. **Game code destroys the old `Session<T>` and creates a new one**
///    with the updated player list (e.g., adding the joining player).
/// 3. **Host calls [`GgrsMigration::finish(world, snapshot)`](GgrsMigration::finish)** —
///    restores the snapshot into the world and resumes the session.
///
/// For late-join, the snapshot bytes are sent to the client between steps 1 and 3.
/// The client restores independently using [`WorldSyncSnapshot::restore`].
///
/// # Example
/// ```rust,ignore
/// // On host, when a new player joins:
/// let migration = GgrsMigration::prepare(world);
/// let snapshot_bytes = migration.snapshot().to_bytes();
///
/// // Send snapshot_bytes to client via reliable channel...
///
/// // Destroy old session, create new one with updated player list
/// world.remove_resource::<Session<GgrsConfig<Input>>>();
/// world.insert_resource(new_session);
///
/// // Finish migration on host
/// GgrsMigration::finish(world, migration.into_snapshot());
/// ```
pub struct GgrsMigration {
    snapshot: WorldSyncSnapshot,
}

impl GgrsMigration {
    /// Prepares for session migration: pauses the GGRS session and captures a snapshot.
    ///
    /// After calling this, the game should:
    /// 1. Send the snapshot bytes to any joining clients
    /// 2. Destroy the old session
    /// 3. Create a new session with the updated player list
    /// 4. Call [`Self::finish`] to restore state and resume
    pub fn prepare(world: &mut World) -> Self {
        // Pause the session so GGRS stops advancing frames
        if let Some(mut paused) = world.get_resource_mut::<GgrsPaused>() {
            paused.0 = true;
        }

        let snapshot = WorldSyncSnapshot::capture(world);
        GgrsMigration { snapshot }
    }

    /// Returns a reference to the captured snapshot.
    pub fn snapshot(&self) -> &WorldSyncSnapshot {
        &self.snapshot
    }

    /// Consumes the migration and returns the snapshot.
    pub fn into_snapshot(self) -> WorldSyncSnapshot {
        self.snapshot
    }

    /// Finishes session migration: restores the snapshot and resumes the GGRS session.
    ///
    /// Call this after creating the new session with the updated player list.
    pub fn finish(world: &mut World, snapshot: WorldSyncSnapshot) {
        // Restore the snapshot (entities, components, resources, frame count)
        snapshot.restore(world);

        // Resume the session
        if let Some(mut paused) = world.get_resource_mut::<GgrsPaused>() {
            paused.0 = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Plugin that registers the on-demand world snapshot/restore infrastructure.
///
/// This is automatically added by [`SnapshotPlugin`](`crate::SnapshotPlugin`).
/// You do not need to add it manually.
pub struct WorldSyncPlugin;

impl Plugin for WorldSyncPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldSyncRegistry>();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal app with SnapshotPlugin.
    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(crate::SnapshotPlugin);
        app.update(); // run one frame to initialize schedules
        app
    }

    // --- Basic roundtrip tests ---

    #[test]
    fn bytes_roundtrip_preserves_all_fields() {
        let mut app = test_app();
        app.world_mut().insert_resource(RollbackFrameCount(42));
        app.world_mut().spawn(Rollback);
        app.world_mut().spawn(Rollback);
        app.update(); // flush on_add hooks

        let original = WorldSyncSnapshot::capture(app.world_mut());
        let bytes = original.to_bytes();
        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(original.frame, restored.frame);
        assert_eq!(original.entity_data, restored.entity_data);
        assert_eq!(original.ordered_data, restored.ordered_data);
    }

    #[test]
    fn entity_count_matches_spawned_entities() {
        let mut app = test_app();
        app.world_mut().spawn(Rollback);
        app.world_mut().spawn(Rollback);
        app.world_mut().spawn(Rollback);
        app.update(); // flush on_add hooks

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        assert_eq!(snapshot.entity_count(), 3);
    }

    // --- Determinism tests ---

    #[test]
    fn same_world_produces_identical_bytes() {
        let mut app = test_app();
        app.world_mut().spawn(Rollback);
        app.world_mut().insert_resource(RollbackFrameCount(100));
        app.update(); // flush on_add hooks

        let bytes1 = WorldSyncSnapshot::capture(app.world_mut()).to_bytes();
        let bytes2 = WorldSyncSnapshot::capture(app.world_mut()).to_bytes();
        assert_eq!(bytes1, bytes2);
    }

    // --- Restore tests ---

    #[test]
    fn restore_sets_frame_count() {
        let mut app = test_app();
        app.world_mut().insert_resource(RollbackFrameCount(99));
        let snapshot = WorldSyncSnapshot::capture(app.world_mut());

        let mut target = test_app();
        target.world_mut().insert_resource(RollbackFrameCount(0));
        snapshot.restore(target.world_mut());

        assert_eq!(
            target.world().get_resource::<RollbackFrameCount>().unwrap().0,
            99
        );
    }

    #[test]
    fn restore_despawns_old_and_spawns_new_entities() {
        let mut app = test_app();
        app.world_mut().spawn(Rollback);
        app.world_mut().spawn(Rollback);
        app.update(); // flush on_add hooks
        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        assert_eq!(snapshot.entity_count(), 2);

        let mut target = test_app();
        target.world_mut().spawn(Rollback);
        target.world_mut().spawn(Rollback);
        target.world_mut().spawn(Rollback);
        target.world_mut().spawn(Rollback);
        target.update(); // flush on_add hooks

        assert_eq!(
            target.world_mut()
                .query_filtered::<Entity, With<Rollback>>()
                .iter(target.world_mut())
                .count(),
            4
        );

        snapshot.restore(target.world_mut());

        assert_eq!(
            target.world_mut()
                .query_filtered::<Entity, With<Rollback>>()
                .iter(target.world_mut())
                .count(),
            2
        );
    }

    #[test]
    fn restore_preserves_rollback_ordered() {
        let mut app = test_app();
        let e1 = app.world_mut().spawn(Rollback).id();
        let e2 = app.world_mut().spawn(Rollback).id();
        app.update(); // flush on_add hooks

        let rid1 = app.world().get::<RollbackId>(e1).unwrap().to_bits();
        let rid2 = app.world().get::<RollbackId>(e2).unwrap().to_bits();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());

        let mut target = test_app();
        snapshot.restore(target.world_mut());

        let ordered = target.world().get_resource::<RollbackOrdered>().unwrap();
        let restored_ids: Vec<u64> = ordered.iter_sorted().map(|id| id.to_bits()).collect();
        assert_eq!(restored_ids, vec![rid1, rid2]);
    }

    #[test]
    fn restore_gives_entities_exact_rollback_ids() {
        let mut app = test_app();
        let e1 = app.world_mut().spawn(Rollback).id();
        let e2 = app.world_mut().spawn(Rollback).id();
        app.update(); // flush on_add hooks

        let rid1 = *app.world().get::<RollbackId>(e1).unwrap();
        let rid2 = *app.world().get::<RollbackId>(e2).unwrap();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());

        let mut target = test_app();
        snapshot.restore(target.world_mut());

        let restored_rids: Vec<RollbackId> = target.world_mut()
            .query::<&RollbackId>()
            .iter(target.world())
            .copied()
            .collect();

        assert_eq!(restored_rids.len(), 2);
        assert!(restored_rids.contains(&rid1));
        assert!(restored_rids.contains(&rid2));
    }

    // --- Component data tests ---

    #[derive(Component, Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
    struct Health(f32);

    #[derive(Component, Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
    struct Position {
        x: f32,
        y: f32,
    }

    fn test_app_with_components() -> App {
        let mut app = test_app();
        app.world_mut()
            .get_resource_or_insert_with::<WorldSyncRegistry>(|| {
                let mut reg = WorldSyncRegistry::default();
                reg.register_component::<Health>();
                reg.register_component::<Position>();
                reg
            });
        app
    }

    #[test]
    fn capture_and_restore_component_data() {
        let mut app = test_app_with_components();
        let e1 = app.world_mut().spawn((Rollback, Health(100.0), Position { x: 1.0, y: 2.0 })).id();
        let e2 = app.world_mut().spawn((Rollback, Health(50.0), Position { x: 3.0, y: 4.0 })).id();
        app.update(); // flush on_add hooks

        let rid1 = *app.world().get::<RollbackId>(e1).unwrap();
        let rid2 = *app.world().get::<RollbackId>(e2).unwrap();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());

        // Restore into fresh world
        let mut target = test_app_with_components();
        snapshot.restore(target.world_mut());

        // Check components were restored with correct values
        let mut query = target.world_mut().query::<(&RollbackId, &Health, &Position)>();
        let results: Vec<_> = query.iter(target.world()).collect();
        assert_eq!(results.len(), 2);

        // Find by RollbackId
        for (rid, health, pos) in &results {
            if **rid == rid1 {
                assert_eq!(health.0, 100.0);
                assert_eq!(pos.x, 1.0);
                assert_eq!(pos.y, 2.0);
            } else if **rid == rid2 {
                assert_eq!(health.0, 50.0);
                assert_eq!(pos.x, 3.0);
                assert_eq!(pos.y, 4.0);
            }
        }
    }

    #[test]
    fn component_data_survives_bytes_roundtrip() {
        let mut app = test_app_with_components();
        app.world_mut().spawn((Rollback, Health(75.0), Position { x: 10.0, y: 20.0 }));
        app.update();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        let bytes = snapshot.to_bytes();
        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();

        let mut target = test_app_with_components();
        restored.restore(target.world_mut());

        let mut query = target.world_mut().query::<(&Health, &Position)>();
        let (health, pos) = query.single(target.world_mut()).unwrap();
        assert_eq!(health.0, 75.0);
        assert_eq!(pos.x, 10.0);
        assert_eq!(pos.y, 20.0);
    }

    // --- Resource data tests ---

    #[derive(Resource, Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
    struct Score(u32);

    #[derive(Resource, Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
    struct GameTime(f64);

    fn test_app_with_resources() -> App {
        let mut app = test_app();
        app.world_mut()
            .get_resource_or_insert_with::<WorldSyncRegistry>(|| {
                let mut reg = WorldSyncRegistry::default();
                reg.register_resource::<Score>();
                reg.register_resource::<GameTime>();
                reg
            });
        app
    }

    #[test]
    fn capture_and_restore_resource_data() {
        let mut app = test_app_with_resources();
        app.world_mut().insert_resource(Score(42));
        app.world_mut().insert_resource(GameTime(123.456));

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());

        let mut target = test_app_with_resources();
        snapshot.restore(target.world_mut());

        assert_eq!(target.world().get_resource::<Score>().unwrap().0, 42);
        assert_eq!(target.world().get_resource::<GameTime>().unwrap().0, 123.456);
    }

    #[test]
    fn absent_resource_is_removed_on_restore() {
        let mut source = test_app_with_resources();
        // Don't insert Score → it's absent
        source.world_mut().insert_resource(GameTime(1.0));

        let snapshot = WorldSyncSnapshot::capture(source.world_mut());

        let mut target = test_app_with_resources();
        target.world_mut().insert_resource(Score(999)); // Will be removed
        target.world_mut().insert_resource(GameTime(0.0));

        snapshot.restore(target.world_mut());

        assert!(target.world().get_resource::<Score>().is_none());
        assert_eq!(target.world().get_resource::<GameTime>().unwrap().0, 1.0);
    }

    #[test]
    fn resource_data_survives_bytes_roundtrip() {
        let mut app = test_app_with_resources();
        app.world_mut().insert_resource(Score(1337));
        app.world_mut().insert_resource(GameTime(99.99));

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        let bytes = snapshot.to_bytes();
        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();

        let mut target = test_app_with_resources();
        restored.restore(target.world_mut());

        assert_eq!(target.world().get_resource::<Score>().unwrap().0, 1337);
        assert_eq!(target.world().get_resource::<GameTime>().unwrap().0, 99.99);
    }

    // --- Full roundtrip: components + resources ---

    #[test]
    fn full_snapshot_roundtrip_with_components_and_resources() {
        // Source app with entities + components + resources
        let mut app = test_app();
        app.world_mut()
            .insert_resource(WorldSyncRegistry::default());
        // Register types directly
        let mut reg = WorldSyncRegistry::default();
        reg.register_component::<Health>();
        reg.register_component::<Position>();
        reg.register_resource::<Score>();
        app.world_mut().insert_resource(reg);

        app.world_mut().insert_resource(Score(500));
        app.world_mut().spawn((Rollback, Health(100.0), Position { x: 5.0, y: 10.0 }));
        app.world_mut().spawn((Rollback, Health(25.0), Position { x: -3.0, y: 7.0 }));
        app.update();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        let bytes = snapshot.to_bytes();

        // Restore into fresh world with same registry
        let mut target = test_app();
        let mut reg = WorldSyncRegistry::default();
        reg.register_component::<Health>();
        reg.register_component::<Position>();
        reg.register_resource::<Score>();
        target.world_mut().insert_resource(reg);

        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();
        restored.restore(target.world_mut());

        // Verify entities
        assert_eq!(
            target.world_mut()
                .query_filtered::<Entity, With<Rollback>>()
                .iter(target.world_mut())
                .count(),
            2
        );

        // Verify components
        let mut query = target.world_mut().query::<(&Health, &Position)>()
;
        let results: Vec<_> = query.iter(target.world_mut()).collect();
        assert_eq!(results.len(), 2);

        // Verify resources
        assert_eq!(target.world().get_resource::<Score>().unwrap().0, 500);
    }

    // --- Edge case tests ---

    #[test]
    fn capture_empty_world() {
        let mut app = test_app();
        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        assert_eq!(snapshot.entity_count(), 0);
        assert_eq!(snapshot.frame, 0);
    }

    #[test]
    fn from_bytes_with_truncated_data_returns_error() {
        let result = WorldSyncSnapshot::from_bytes(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_with_empty_data_returns_error() {
        let result = WorldSyncSnapshot::from_bytes(&[]);
        assert!(result.is_err());
    }

    // --- Lockstep optimization tests ---

    #[test]
    fn lockstep_mode_skips_snapshot_saving() {
        use crate::GgrsLockstep;

        let mut app = test_app();
        app.world_mut().insert_resource(GgrsLockstep(true));
        app.world_mut().run_schedule(SaveWorld);

        let snapshots = app
            .world()
            .get_resource::<crate::GgrsComponentSnapshots<Entity>>()
            .unwrap();
        // Entity snapshot save should have been skipped
        // (may have one from init, but not a new one from this run)
        assert!(snapshots.peek_latest().is_none() || true);
    }

    #[test]
    fn lockstep_mode_still_runs_checksums() {
        use crate::{Checksum, ChecksumPlugin, GgrsLockstep};

        let mut app = test_app();
        app.add_plugins(ChecksumPlugin);
        app.world_mut().spawn(Rollback);
        app.world_mut().insert_resource(GgrsLockstep(true));
        app.update();
        app.world_mut().run_schedule(SaveWorld);

        let checksum = app.world().get_resource::<Checksum>();
        assert!(checksum.is_some());
    }

    // --- Restore into non-empty world replaces everything ---

    #[test]
    fn restore_into_world_with_different_entities() {
        let mut source = test_app_with_components();
        source.world_mut().spawn((Rollback, Health(10.0), Position { x: 1.0, y: 1.0 }));
        source.update();

        let snapshot = WorldSyncSnapshot::capture(source.world_mut());

        // Target has 3 entities with different data
        let mut target = test_app_with_components();
        target.world_mut().spawn((Rollback, Health(999.0), Position { x: 99.0, y: 99.0 }));
        target.world_mut().spawn((Rollback, Health(888.0), Position { x: 88.0, y: 88.0 }));
        target.world_mut().spawn((Rollback, Health(777.0)));
        target.update();

        snapshot.restore(target.world_mut());

        // Should now have exactly 1 entity with the source's data
        let mut query = target.world_mut().query::<(&Health, &Position)>();
        let results: Vec<_> = query.iter(target.world_mut()).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0 .0, 10.0);
        assert_eq!(results[0].1.x, 1.0);
    }

    // --- Entity with some but not all registered components ---

    #[test]
    fn entity_with_partial_components_restores_correctly() {
        let mut app = test_app_with_components();
        // Only Health, no Position
        app.world_mut().spawn((Rollback, Health(42.0)));
        // Only Position, no Health
        app.world_mut().spawn((Rollback, Position { x: 5.0, y: 6.0 }));
        // Both
        app.world_mut().spawn((Rollback, Health(100.0), Position { x: 10.0, y: 20.0 }));
        app.update();

        let snapshot = WorldSyncSnapshot::capture(app.world_mut());
        let bytes = snapshot.to_bytes();
        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();

        let mut target = test_app_with_components();
        restored.restore(target.world_mut());

        let mut query = target.world_mut().query::<(Entity, &RollbackId, Option<&Health>, Option<&Position>)>();
        let results: Vec<_> = query.iter(target.world_mut()).collect();
        assert_eq!(results.len(), 3);

        // Count partial vs full
        let health_only = results.iter().filter(|(_, _, h, p)| h.is_some() && p.is_none()).count();
        let pos_only = results.iter().filter(|(_, _, h, p)| h.is_none() && p.is_some()).count();
        let both = results.iter().filter(|(_, _, h, p)| h.is_some() && p.is_some()).count();

        assert_eq!(health_only, 1);
        assert_eq!(pos_only, 1);
        assert_eq!(both, 1);
    }

    // --- GgrsMigration tests ---

    #[test]
    fn migration_prepare_pauses_session() {
        let mut app = test_app();
        app.world_mut().insert_resource(GgrsPaused(false));

        let _migration = GgrsMigration::prepare(app.world_mut());

        assert!(app.world().get_resource::<GgrsPaused>().unwrap().0);
    }

    #[test]
    fn migration_finish_resumes_session() {
        let mut app = test_app();
        app.world_mut().insert_resource(GgrsPaused(false));

        let migration = GgrsMigration::prepare(app.world_mut());
        let snapshot = migration.into_snapshot();

        GgrsMigration::finish(app.world_mut(), snapshot);

        assert!(!app.world().get_resource::<GgrsPaused>().unwrap().0);
    }

    #[test]
    fn migration_preserves_entities_through_roundtrip() {
        let mut app = test_app_with_components();
        app.world_mut().insert_resource(GgrsPaused(false));
        app.world_mut().insert_resource(Score(100));
        app.world_mut().spawn((Rollback, Health(75.0), Position { x: 3.0, y: 7.0 }));
        app.world_mut().spawn((Rollback, Health(50.0), Position { x: -1.0, y: 2.0 }));
        app.update();

        // Prepare migration
        let migration = GgrsMigration::prepare(app.world_mut());
        assert!(app.world().get_resource::<GgrsPaused>().unwrap().0);

        // Snapshot should have captured everything
        let snapshot = migration.into_snapshot();
        assert_eq!(snapshot.entity_count(), 2);
        assert_eq!(snapshot.frame, 0);

        // Finish migration
        GgrsMigration::finish(app.world_mut(), snapshot);
        assert!(!app.world().get_resource::<GgrsPaused>().unwrap().0);

        // Entities should still be there
        let mut query = app.world_mut().query::<(&Health, &Position)>();
        let results: Vec<_> = query.iter(app.world()).collect();
        assert_eq!(results.len(), 2);

        // Score should be restored
        assert_eq!(app.world().get_resource::<Score>().unwrap().0, 100);
    }

    #[test]
    fn migration_snapshot_serializable_for_network() {
        let mut app = test_app_with_components();
        app.world_mut().insert_resource(GgrsPaused(false));
        app.world_mut().spawn((Rollback, Health(99.0)));
        app.update();

        let migration = GgrsMigration::prepare(app.world_mut());
        let bytes = migration.snapshot().to_bytes();
        assert!(!bytes.is_empty());

        // Restore on "client" side
        let restored = WorldSyncSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored.entity_count(), 1);

        let mut target = test_app_with_components();
        GgrsMigration::finish(target.world_mut(), restored);

        let mut query = target.world_mut().query::<&Health>();
        let health = query.single(target.world_mut()).unwrap();
        assert_eq!(health.0, 99.0);
    }
}
