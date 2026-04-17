//! Mid-session player sync for lockstep GGRS sessions.
//!
//! Handles adding new players to an existing session without restarting the app.
//! The socket always listens for connections. When a new peer is detected:
//!
//! 1. Host pauses → existing clients auto-pause (lockstep waits for host inputs)
//! 2. Host notifies existing clients (`TAG_PAUSE`)
//! 3. Host serializes world via [`SyncSerialize`] schedule → sends to new client
//! 4. New client deserializes via [`SyncDeserialize`] schedule → sends `TAG_READY`
//! 5. Host runs [`RebuildSession`] schedule (game creates new session with added player)
//! 6. Host sends `TAG_RESUME` with session config to all clients
//! 7. All clients run [`RebuildSession`] → unpause
//!
//! # Channel 16 (reliable data)
//!
//! Each peer-to-peer WebRTC connection has its own set of data channels.
//! Channel 0 is unreliable (ggrs inputs), channel 16 is reliable (sync data).
//! The host has a separate channel 16 to each peer — `data_send(data, peer_id)`
//! targets a specific peer's channel. All clients can use channel 16 because
//! each connection has its own.

use std::collections::HashSet;
use std::hash::Hash;

use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;

use ggrs::Config;

use crate::{reset_timestep_accumulator, GgrsPaused, RollbackFrameCount, Session};

// ── Protocol Tags ──────────────────────────────────────────────────────────

/// Host → existing clients: "new player joining, pause"
pub const TAG_PAUSE: u8 = 0x10;
/// Host → new client: "here's the world snapshot"
pub const TAG_SYNC_DATA: u8 = 0x11;
/// New client → host: "snapshot applied, ready to play"
pub const TAG_READY: u8 = 0x12;
/// Host → all clients: "rebuild session and resume"
pub const TAG_RESUME: u8 = 0x13;

// ── Schedule Labels ────────────────────────────────────────────────────────

/// Game systems: serialize world state into [`WorldSnapshot`].
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SyncSerialize;

/// Game systems: deserialize [`WorldSnapshot`] into the ECS world.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SyncDeserialize;

/// Game systems: rebuild the ggrs session using [`SessionConfigData`].
///
/// On the **host**: create a new `P2PSession` with the additional player,
/// then write the serialized session config to `SessionConfigData`
/// (bevy_ggrs will send it to all clients via `TAG_RESUME`).
///
/// On **clients**: read `SessionConfigData` and create a new `P2PSession`.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RebuildSession;

// ── Resources ──────────────────────────────────────────────────────────────

/// Serialized world snapshot. Populated by game's [`SyncSerialize`] systems,
/// consumed by game's [`SyncDeserialize`] systems.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct WorldSnapshot(pub Vec<u8>);

/// Serialized session config (handle map, player info, etc.).
///
/// On the host: populated by the game's [`RebuildSession`] systems.
/// On clients: populated from the `TAG_RESUME` message payload.
/// Consumed by the game's [`RebuildSession`] systems.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct SessionConfigData(pub Vec<u8>);

/// Reliable message inbox. Game adapter system populates from the socket
/// before the sync system runs each frame.
#[derive(Resource, Deref, DerefMut)]
pub struct SyncInbox<A>(pub Vec<(A, Vec<u8>)>);

impl<A> Default for SyncInbox<A> {
    fn default() -> Self { Self(Vec::new()) }
}

/// Reliable message outbox. Game adapter system flushes to the socket
/// after the sync system runs each frame.
#[derive(Resource, Deref, DerefMut)]
pub struct SyncOutbox<A>(pub Vec<(A, Vec<u8>)>);

impl<A> Default for SyncOutbox<A> {
    fn default() -> Self { Self(Vec::new()) }
}

/// Currently connected peer addresses. Game adapter system updates this
/// before the sync system runs each frame.
#[derive(Resource, Deref, DerefMut)]
pub struct ConnectedPeers<A>(pub Vec<A>);

impl<A> Default for ConnectedPeers<A> {
    fn default() -> Self { Self(Vec::new()) }
}

/// Whether this peer is the session host (handle 0).
/// The game sets this when creating the initial session.
#[derive(Resource, Default)]
pub struct IsHost(pub bool);

// ── Internal State ─────────────────────────────────────────────────────────

#[derive(Resource)]
pub(crate) struct GgrsSyncState<A: Eq + Hash + Clone> {
    phase: Phase,
    /// Peers currently in the ggrs session (not counting the new one being synced).
    session_peers: HashSet<A>,
    /// The new peer being synced (host only).
    new_peer: Option<A>,
    /// Host address (for clients to reply to).
    host_addr: Option<A>,
}

impl<A: Eq + Hash + Clone> Default for GgrsSyncState<A> {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            session_peers: HashSet::new(),
            new_peer: None,
            host_addr: None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
enum Phase {
    #[default]
    Idle,
    /// Host: sent snapshot, waiting for `TAG_READY` from new client.
    HostWaitingReady,
    /// Client: received snapshot, running `SyncDeserialize`.
    ClientDeserializing,
    /// Client: deserialized, sent `TAG_READY`, waiting for `TAG_RESUME`.
    ClientWaitingResume,
    /// Existing client: received `TAG_PAUSE`, waiting for `TAG_RESUME`.
    ExistingWaitingResume,
}

// ── Sync System ────────────────────────────────────────────────────────────

/// Main sync orchestration system. Runs in PreUpdate before `run_ggrs_schedules`.
pub(crate) fn run_mid_session_sync<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    // Ensure resources exist
    if world.get_resource::<GgrsSyncState<T::Address>>().is_none() {
        world.insert_resource(GgrsSyncState::<T::Address>::default());
    }
    world.get_resource_or_insert_with(WorldSnapshot::default);
    world.get_resource_or_insert_with(SessionConfigData::default);
    world.get_resource_or_insert_with(SyncInbox::<T::Address>::default);
    world.get_resource_or_insert_with(SyncOutbox::<T::Address>::default);
    world.get_resource_or_insert_with(ConnectedPeers::<T::Address>::default);

    let phase = world.resource::<GgrsSyncState<T::Address>>().phase.clone();

    match phase {
        Phase::Idle => handle_idle::<T>(world),
        Phase::HostWaitingReady => handle_host_waiting::<T>(world),
        Phase::ClientDeserializing => handle_client_deserializing::<T>(world),
        Phase::ClientWaitingResume => handle_client_waiting_resume::<T>(world),
        Phase::ExistingWaitingResume => handle_existing_waiting::<T>(world),
    }
}

fn handle_idle<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    // Drain inbox
    let inbox: Vec<(T::Address, Vec<u8>)> = world.resource::<SyncInbox<T::Address>>().0.clone();
    world.resource_mut::<SyncInbox<T::Address>>().0.clear();

    let has_session = world.get_resource::<Session<T>>().is_some();
    let is_host = world
        .get_resource::<IsHost>()
        .map(|h| h.0)
        .unwrap_or(false);

    // ── Host: check for new peers ──────────────────────────────────────
    if has_session && is_host {
        let connected = world.resource::<ConnectedPeers<T::Address>>().0.clone();
        let session_peers = world
            .resource::<GgrsSyncState<T::Address>>()
            .session_peers
            .clone();

        // First session with existing peers: populate session_peers
        // (handles the case where the host creates a session after clients connect)
        let session_peers = if session_peers.is_empty() && !connected.is_empty() {
            // If we already have a session with players, assume all connected
            // peers are session peers (e.g., after initial P2P session creation).
            // New peer detection will work correctly from here.
            let initial: HashSet<_> = connected.iter().cloned().collect();
            world
                .resource_mut::<GgrsSyncState<T::Address>>()
                .session_peers = initial.clone();
            initial
        } else {
            session_peers
        };

        let new_peer = connected
            .iter()
            .find(|p| !session_peers.contains(p))
            .cloned();

        if let Some(new_peer) = new_peer {
            info!("[SYNC] New peer detected, starting sync");

            // 1. Pause
            *world.resource_mut::<GgrsPaused>() = GgrsPaused(true);
            reset_timestep_accumulator(world);

            // 2. Notify existing clients
            for peer in &session_peers {
                world
                    .resource_mut::<SyncOutbox<T::Address>>()
                    .0
                    .push((peer.clone(), vec![TAG_PAUSE]));
            }

            // 3. Serialize world
            run_schedule_extract(world, SyncSerialize);
            let snapshot = world.resource::<WorldSnapshot>().0.clone();

            // 4. Send snapshot to new peer
            let mut data = vec![TAG_SYNC_DATA];
            data.extend_from_slice(&snapshot);
            world
                .resource_mut::<SyncOutbox<T::Address>>()
                .0
                .push((new_peer.clone(), data));

            // 5. Update state
            {
                let mut state = world.resource_mut::<GgrsSyncState<T::Address>>();
                state.phase = Phase::HostWaitingReady;
                state.new_peer = Some(new_peer);
            }
            return;
        }
    }

    // ── Existing client: check inbox for host messages ─────────────────
    if has_session && !is_host {
        for (peer, msg) in &inbox {
            if msg.is_empty() {
                continue;
            }
            match msg[0] {
                TAG_PAUSE => {
                    info!("[SYNC] Received pause from host");
                    *world.resource_mut::<GgrsPaused>() = GgrsPaused(true);
                    reset_timestep_accumulator(world);
                    let mut state = world.resource_mut::<GgrsSyncState<T::Address>>();
                    state.phase = Phase::ExistingWaitingResume;
                    state.host_addr = Some(peer.clone());
                    return;
                }
                TAG_RESUME => {
                    info!("[SYNC] Received resume from host");
                    apply_resume::<T>(world, msg);
                    return;
                }
                _ => {}
            }
        }
    }

    // ── New client (no session yet): wait for snapshot ─────────────────
    if !has_session {
        for (peer, msg) in &inbox {
            if msg.is_empty() {
                continue;
            }
            if msg[0] == TAG_SYNC_DATA {
                info!("[SYNC] Received snapshot from host");
                *world.resource_mut::<WorldSnapshot>() = WorldSnapshot(msg[1..].to_vec());
                *world.resource_mut::<GgrsPaused>() = GgrsPaused(true);
                let mut state = world.resource_mut::<GgrsSyncState<T::Address>>();
                state.host_addr = Some(peer.clone());
                state.phase = Phase::ClientDeserializing;
                return;
            }
        }
    }
}

fn handle_host_waiting<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    let inbox: Vec<(T::Address, Vec<u8>)> = world.resource::<SyncInbox<T::Address>>().0.clone();
    world.resource_mut::<SyncInbox<T::Address>>().0.clear();

    let new_peer = world
        .resource::<GgrsSyncState<T::Address>>()
        .new_peer
        .clone();

    for (peer, msg) in &inbox {
        if msg.is_empty() {
            continue;
        }
        if msg[0] == TAG_READY {
            if let Some(ref expected) = new_peer {
                if peer == expected {
                    info!("[SYNC] New client ready, rebuilding session");

                    // Run RebuildSession (game creates new session + populates SessionConfigData)
                    run_schedule_extract(world, RebuildSession);

                    // Send TAG_RESUME with session config to ALL clients (including the new one)
                    let config = world.resource::<SessionConfigData>().0.clone();
                    let mut resume_msg = vec![TAG_RESUME];
                    resume_msg.extend_from_slice(&config);

                    // Send to all connected peers (session peers + new peer)
                    let all_peers: Vec<_> = world
                        .resource::<ConnectedPeers<T::Address>>()
                        .0
                        .iter()
                        .cloned()
                        .collect();
                    for peer in &all_peers {
                        world
                            .resource_mut::<SyncOutbox<T::Address>>()
                            .0
                            .push((peer.clone(), resume_msg.clone()));
                    }

                    // Update state
                    let connected = world.resource::<ConnectedPeers<T::Address>>().0.clone();
                    let mut state = world.resource_mut::<GgrsSyncState<T::Address>>();
                    state.session_peers = connected.into_iter().collect();
                    state.phase = Phase::Idle;
                    state.new_peer = None;

                    // Unpause self
                    *world.resource_mut::<RollbackFrameCount>() = RollbackFrameCount(0);
                    reset_timestep_accumulator(world);
                    *world.resource_mut::<GgrsPaused>() = GgrsPaused(false);

                    info!("[SYNC] Session rebuilt, all resumed");
                    return;
                }
            }
        }
    }
}

fn handle_client_deserializing<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    info!("[SYNC] Deserializing world snapshot");

    // Run SyncDeserialize schedule
    run_schedule_extract(world, SyncDeserialize);

    // Send TAG_READY to host
    let host_addr = world
        .resource::<GgrsSyncState<T::Address>>()
        .host_addr
        .clone();
    if let Some(host) = host_addr {
        world
            .resource_mut::<SyncOutbox<T::Address>>()
            .0
            .push((host, vec![TAG_READY]));
    }

    // Move to waiting for resume
    world.resource_mut::<GgrsSyncState<T::Address>>().phase = Phase::ClientWaitingResume;
    info!("[SYNC] Snapshot applied, sent ready to host");
}

fn handle_client_waiting_resume<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    let inbox: Vec<(T::Address, Vec<u8>)> = world.resource::<SyncInbox<T::Address>>().0.clone();
    world.resource_mut::<SyncInbox<T::Address>>().0.clear();

    for (_peer, msg) in &inbox {
        if msg.is_empty() {
            continue;
        }
        if msg[0] == TAG_RESUME {
            info!("[SYNC] Received resume from host");
            apply_resume::<T>(world, msg);
            return;
        }
    }
}

fn handle_existing_waiting<T: Config>(world: &mut World)
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    let inbox: Vec<(T::Address, Vec<u8>)> = world.resource::<SyncInbox<T::Address>>().0.clone();
    world.resource_mut::<SyncInbox<T::Address>>().0.clear();

    for (_peer, msg) in &inbox {
        if msg.is_empty() {
            continue;
        }
        if msg[0] == TAG_RESUME {
            info!("[SYNC] Received resume from host");
            apply_resume::<T>(world, msg);
            return;
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Apply a TAG_RESUME message: store config, run RebuildSession, unpause.
fn apply_resume<T: Config>(world: &mut World, msg: &[u8])
where
    T::Address: Clone + Eq + Hash + Send + Sync,
{
    // Store config data from message payload (after tag byte)
    *world.resource_mut::<SessionConfigData>() = SessionConfigData(msg[1..].to_vec());

    // Run RebuildSession (game creates new session)
    run_schedule_extract(world, RebuildSession);

    // Unpause
    *world.resource_mut::<RollbackFrameCount>() = RollbackFrameCount(0);
    reset_timestep_accumulator(world);
    *world.resource_mut::<GgrsPaused>() = GgrsPaused(false);

    // Update session peers from connected peers
    let connected = world.resource::<ConnectedPeers<T::Address>>().0.clone();
    let mut state = world.resource_mut::<GgrsSyncState<T::Address>>();
    state.session_peers = connected.into_iter().collect();
    state.phase = Phase::Idle;

    info!("[SYNC] Session rebuilt, resumed");
}

/// Extract a schedule by label, run it, and put it back.
/// If the schedule doesn't exist, silently returns (game hasn't configured it).
fn run_schedule_extract(world: &mut World, label: impl ScheduleLabel + Clone) {
    let mut schedules = world.resource_mut::<Schedules>();
    let Some((_, mut schedule)) = schedules.remove_entry(label) else {
        return;
    };
    drop(schedules);

    schedule.run(world);

    let mut schedules = world.resource_mut::<Schedules>();
    schedules.insert(schedule);
}
