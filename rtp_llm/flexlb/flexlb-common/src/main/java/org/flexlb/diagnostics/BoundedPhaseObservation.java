package org.flexlb.diagnostics;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.dao.BalanceContext;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.SecureDirectoryStream;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.nio.file.attribute.PosixFileAttributeView;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;

/** Isolated diagnostic only. No property/no active original performance case means no capture. */
public final class BoundedPhaseObservation {
    public static final String DIRECTORY_PROPERTY = "flexlb.diagnostic.phase-directory";
    static final int MAX_REQUESTS = 1024;
    static final int MAX_POINTS = 32768;
    static final int MAX_CASES = 16;
    static final int MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
    static final long CASE_NANOS = TimeUnit.SECONDS.toNanos(120);
    static final long SERIALIZE_NANOS = TimeUnit.MILLISECONDS.toNanos(200);
    private static final AtomicInteger CASE_SEQUENCE = new AtomicInteger();
    private static volatile Capture current;

    public enum Phase {
        SERVICE_START, GLOBAL_OFFER, PLANNER_START, PLANNER_END,
        DECISION_START, DECISION_END, WORKER_ADMITTED, WORKER_CYCLE,
        HANDOFF_START, HANDOFF_END, EXECUTOR_START, PROTO_START, PROTO_END, RPC_START, DELIVERY_PUBLISHED
    }

    public enum SlotPhase { CLAIMED, SUBMITTED, OUTER_CLOSE_DONE, RELEASED }

    /** Read only while the caller already owns the original queue lock. */
    public record SlotState(int plannerCount, int inFlight, int completed,
                            int queueDepth, boolean unscanned, boolean ready) { }

    private record SlotPoint(Point point, int attempt, SlotState state) { }

    public record Point(long wallMillis, long nanos, long threadId, long cpuNanos, long batchId, int caseSequence) { }

    private BoundedPhaseObservation() { }

    public static boolean enabled() {
        return current != null;
    }

    public static void beginCase(String method, String displayName) {
        try {
            String directory = System.getProperty(DIRECTORY_PROPERTY, "");
            if (directory.isBlank()) { return; }
            if (!Set.of("batchScheduleRemainsFastAcrossRealGrpcBoundaries",
                    "masterMeetsRateSloAcrossEngineScaleMatrix").contains(method)) { return; }
            if (displayName.length() > 256 || current != null) { return; }
            int sequence = CASE_SEQUENCE.incrementAndGet();
            if (sequence > MAX_CASES) { return; }
            current = new Capture(Path.of(directory), method, displayName, sequence);
        } catch (Throwable ignored) {
            // Observation cannot replace the original test/setup failure.
        }
    }

    public static Point snapshot() {
        Capture capture = current;
        if (capture == null) { return null; }
        try {
            return capture.point(-1L);
        } catch (Throwable ignored) {
            capture.errors.incrementAndGet();
            return null;
        }
    }

    public static void mark(BalanceContext context, Phase phase) {
        mark(context, phase, null, -1L);
    }

    public static void mark(BalanceContext context, Phase phase, Point observed, long batchId) {
        Capture capture = current;
        if (capture == null || context == null) { return; }
        long requestId;
        try {
            requestId = context.getRequestId();
            if ((requestId & 15L) != 0L) { return; }
        } catch (Throwable ignored) { capture.errors.incrementAndGet(); return; }
        capture.writers.incrementAndGet();
        try {
            if (capture.closed
                    || (observed != null && observed.caseSequence() != capture.sequence)) { return; }
            if (System.nanoTime() - capture.startedNanos >= CASE_NANOS) {
                capture.expired.incrementAndGet();
                return;
            }
            if (capture.points.getAndIncrement() >= MAX_POINTS) {
                capture.pointOmissions.incrementAndGet();
                return;
            }
            Trace trace = capture.traces.get(requestId);
            // Only the actual service-entry point creates a trace. Weak identity binds
            // later callbacks without retaining request payloads or crossing case epochs.
            if (trace == null && phase == Phase.SERVICE_START) {
                if (capture.slots.incrementAndGet() > MAX_REQUESTS) {
                    capture.slots.decrementAndGet();
                    capture.requestOmissions.incrementAndGet();
                    return;
                }
                Trace candidate = new Trace(requestId, context);
                Trace existing = capture.traces.putIfAbsent(requestId, candidate);
                if (existing != null) { capture.slots.decrementAndGet(); }
                trace = existing == null ? candidate : existing;
            }
            if (trace == null || trace.context.get() != context) { return; }
            Point point = observed == null ? capture.point(batchId)
                    : new Point(observed.wallMillis(), observed.nanos(), observed.threadId(), observed.cpuNanos(), batchId, capture.sequence);
            if (point.cpuNanos() < 0L) { capture.cpuUnavailable.incrementAndGet(); }
            int index = phase.ordinal();
            trace.counts.incrementAndGet(index);
            trace.first.compareAndSet(index, null, point);
            trace.last.set(index, point);
        } catch (Throwable ignored) {
            capture.errors.incrementAndGet();
        } finally {
            capture.writers.decrementAndGet();
        }
    }

    /** Sidechannel only: never creates a request trace or changes the queue. */
    public static void markSlot(BalanceContext context, SlotPhase phase, SlotState state) {
        Capture capture = current;
        if (capture == null || context == null) { return; }
        capture.writers.incrementAndGet();
        try {
            long requestId = context.getRequestId();
            if ((requestId & 15L) != 0L || capture.closed) { return; }
            Trace trace = capture.traces.get(requestId);
            if (trace == null || trace.context.get() != context) { return; }
            if (System.nanoTime() - capture.startedNanos >= CASE_NANOS) {
                capture.expired.incrementAndGet();
                return;
            }
            if (capture.points.getAndIncrement() >= MAX_POINTS) {
                capture.pointOmissions.incrementAndGet();
                return;
            }
            int attempt = phase == SlotPhase.CLAIMED
                    ? trace.slotAttempt.incrementAndGet() : trace.slotAttempt.get();
            Point point = capture.point(-1L);
            if (point.cpuNanos() < 0L) { capture.cpuUnavailable.incrementAndGet(); }
            SlotPoint value = new SlotPoint(point, attempt, state);
            int index = phase.ordinal();
            trace.slotCounts.incrementAndGet(index);
            trace.slotFirst.compareAndSet(index, null, value);
            trace.slotLast.set(index, value);
        } catch (Throwable ignored) {
            capture.errors.incrementAndGet();
        } finally {
            capture.writers.decrementAndGet();
        }
    }

    public static void observationError() {
        Capture capture = current;
        if (capture != null) { capture.errors.incrementAndGet(); }
    }

    /** One bounded write after the original case cleanup; never a request-path write. */
    public static void finishCase() {
        Capture capture = current;
        current = null;
        if (capture == null) { return; }
        capture.closed = true;
        try {
            capture.write();
        } catch (Throwable ignored) {
            System.err.println("[phase-observation] output unavailable (" + ignored.getClass().getSimpleName()
                    + "); original test outcome unchanged");
        }
    }

    static void resetForTests() {
        current = null;
        CASE_SEQUENCE.set(0);
    }

    private static final class Trace {
        final long requestId;
        final WeakReference<BalanceContext> context;
        final AtomicReferenceArray<Point> first = new AtomicReferenceArray<>(Phase.values().length);
        final AtomicReferenceArray<Point> last = new AtomicReferenceArray<>(Phase.values().length);
        final AtomicIntegerArray counts = new AtomicIntegerArray(Phase.values().length);
        final AtomicInteger slotAttempt = new AtomicInteger();
        final AtomicReferenceArray<SlotPoint> slotFirst = new AtomicReferenceArray<>(SlotPhase.values().length);
        final AtomicReferenceArray<SlotPoint> slotLast = new AtomicReferenceArray<>(SlotPhase.values().length);
        final AtomicIntegerArray slotCounts = new AtomicIntegerArray(SlotPhase.values().length);
        Trace(long requestId, BalanceContext context) { this.requestId = requestId; this.context = new WeakReference<>(context); }
    }

    private static final class Capture {
        final Path directory;
        final String method;
        final String displayName;
        final int sequence;
        final long startedNanos = System.nanoTime();
        final long startedWallMillis = System.currentTimeMillis();
        final ThreadMXBean threads = ManagementFactory.getThreadMXBean();
        final ConcurrentHashMap<Long, Trace> traces = new ConcurrentHashMap<>();
        final AtomicInteger slots = new AtomicInteger();
        final AtomicInteger points = new AtomicInteger();
        final AtomicInteger writers = new AtomicInteger();
        final AtomicLong requestOmissions = new AtomicLong();
        final AtomicLong pointOmissions = new AtomicLong();
        final AtomicLong expired = new AtomicLong();
        final AtomicLong errors = new AtomicLong();
        final AtomicLong cpuUnavailable = new AtomicLong();
        volatile boolean closed;

        Capture(Path directory, String method, String displayName, int sequence) {
            this.directory = directory;
            this.method = method;
            this.displayName = displayName;
            this.sequence = sequence;
        }

        Point point(long batchId) {
            long cpu = threads.isCurrentThreadCpuTimeSupported() && threads.isThreadCpuTimeEnabled()
                    ? threads.getCurrentThreadCpuTime() : -1L;
            return new Point(System.currentTimeMillis(), System.nanoTime(), Thread.currentThread().threadId(), cpu, batchId, sequence);
        }

        byte[] encode() throws IOException {
            long deadline = System.nanoTime() + SERIALIZE_NANOS;
            Map<String, Object> report = new LinkedHashMap<>();
            report.put("purpose", "DIAGNOSTIC_ONLY_NOT_ACCEPTANCE");
            report.put("class", "org.flexlb.httpserver.MasterBatchEndToEndPerformanceTest");
            report.put("method", method);
            report.put("display_name", displayName);
            report.put("sequence", sequence);
            report.put("schema_version", 2);
            report.put("point_encoding", "last omitted only if count=1 and first.equals(last); reconstruct last=first exactly; repeated events retain both, not a full retry trace");
            report.put("slot_rule", "CLAIMED increments per-request observer attempt_id; compare only same nonzero attempt, count1 points; state is instantaneous existing-lock snapshot, not whole wait occupancy; OUTER_CLOSE_DONE excludes prior commit-owned close");
            report.put("jvm_pid", ProcessHandle.current().pid());
            report.put("cpu_unavailable_points", cpuUnavailable.get());
            report.put("started_wall_millis", startedWallMillis);
            report.put("started_nanos", startedNanos);
            report.put("sample_rule", "requestId low4bits==0; no payload/token data");
            report.put("request_cap", MAX_REQUESTS);
            report.put("point_cap", MAX_POINTS);
            report.put("case_capture_nanos", CASE_NANOS);
            report.put("request_omissions", requestOmissions.get());
            report.put("point_omissions", pointOmissions.get());
            report.put("expired_marks", expired.get());
            report.put("observation_errors", errors.get());
            report.put("writers_at_finish", writers.get());
            report.put("cpu_rule", "difference only same thread; -1 unsupported; batch-shared costs repeated for members, never sum members or overlapping phases");
            List<Object> rows = new ArrayList<>();
            List<Trace> ordered = new ArrayList<>(traces.values());
            ordered.sort(Comparator.comparingLong(trace -> trace.requestId));
            for (Trace trace : ordered) {
                if (System.nanoTime() >= deadline) { break; }
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("request_id", trace.requestId);
                Map<String, Object> phases = new LinkedHashMap<>();
                for (Phase phase : Phase.values()) {
                    Point first = trace.first.get(phase.ordinal());
                    if (first == null) { continue; }
                    Point last = trace.last.get(phase.ordinal());
                    int count = trace.counts.get(phase.ordinal());
                    phases.put(phase.name(), count == 1 && first.equals(last)
                            ? Map.of("count", count, "first", pointMap(first))
                            : Map.of("count", count, "first", pointMap(first), "last", pointMap(last)));
                }
                row.put("phases", phases);
                Map<String, Object> slots = new LinkedHashMap<>();
                for (SlotPhase phase : SlotPhase.values()) {
                    SlotPoint first = trace.slotFirst.get(phase.ordinal());
                    if (first == null) { continue; }
                    int count = trace.slotCounts.get(phase.ordinal());
                    SlotPoint last = trace.slotLast.get(phase.ordinal());
                    slots.put(phase.name(), count == 1 && first.equals(last)
                            ? Map.of("count", count, "first", slotMap(first))
                            : Map.of("count", count, "first", slotMap(first),
                                    "last", slotMap(trace.slotLast.get(phase.ordinal()))));
                }
                row.put("slot_lifecycle", slots);
                rows.add(row);
            }
            report.put("serialization_omissions", ordered.size() - rows.size());
            report.put("requests", rows);
            BoundedBytes bytes = new BoundedBytes();
            try {
                new ObjectMapper().writeValue(bytes, report);
            } catch (IOException | OutputLimit limit) {
                report.remove("requests");
                report.put("output_limit_or_serialization_error", true);
                bytes = new BoundedBytes();
                new ObjectMapper().writeValue(bytes, report);
            }
            return bytes.toByteArray();
        }

        void write() throws IOException {
            byte[] bytes = encode();
            // Start at the stable filesystem root and traverse each component through
            // the preceding directory handle. No checked pathname is reopened later.
            Path normalized = directory.toAbsolutePath().normalize();
            List<DirectoryStream<Path>> ownedStreams = new ArrayList<>();
            try {
                DirectoryStream<Path> root = Files.newDirectoryStream(normalized.getRoot());
                ownedStreams.add(root);
                if (!(root instanceof SecureDirectoryStream<Path> secureRoot)) {
                    throw new IOException("secure observation directory unavailable");
                }
                SecureDirectoryStream<Path> secure = secureRoot;
                for (Path component : normalized) {
                    secure = secure.newDirectoryStream(component, LinkOption.NOFOLLOW_LINKS);
                    ownedStreams.add(secure);
                }
                var attributes = secure.getFileAttributeView(PosixFileAttributeView.class).readAttributes();
                if (!attributes.isDirectory()
                        || !attributes.permissions().equals(PosixFilePermissions.fromString("rwx------"))) {
                    throw new IOException("observation directory is not private");
                }
                Set<OpenOption> options = Set.of(StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE, LinkOption.NOFOLLOW_LINKS);
                try (SeekableByteChannel out = secure.newByteChannel(Path.of(String.format("phase-%03d.json", sequence)), options,
                        PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")))) {
                    ByteBuffer buffer = ByteBuffer.wrap(bytes);
                    while (buffer.hasRemaining()) { out.write(buffer); }
                }
            } finally {
                IOException closeError = null;
                for (int index = ownedStreams.size() - 1; index >= 0; index--) {
                    try { ownedStreams.get(index).close(); }
                    catch (IOException error) { if (closeError == null) { closeError = error; } }
                }
                if (closeError != null) { throw closeError; }
            }
        }
    }

    private static Map<String, Long> pointMap(Point point) {
        return Map.of("wall_millis", point.wallMillis(), "nanos", point.nanos(),
                "thread_id", point.threadId(), "cpu_nanos", point.cpuNanos(), "batch_id", point.batchId());
    }

    private static Map<String, Object> slotMap(SlotPoint value) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("point", pointMap(value.point()));
        result.put("attempt_id", value.attempt());
        SlotState state = value.state();
        if (state != null) {
            result.put("state", Map.of("planner_count", state.plannerCount(),
                    "in_flight", state.inFlight(), "completed", state.completed(),
                    "queue_depth", state.queueDepth(), "unscanned", state.unscanned(), "ready", state.ready()));
        }
        return result;
    }

    private static final class BoundedBytes extends ByteArrayOutputStream {
        final long deadline = System.nanoTime() + SERIALIZE_NANOS;
        @Override public synchronized void write(int value) {
            if (count >= MAX_OUTPUT_BYTES || System.nanoTime() >= deadline) { throw new OutputLimit(); }
            super.write(value);
        }
        @Override public synchronized void write(byte[] values, int offset, int length) {
            if (length > MAX_OUTPUT_BYTES - count || System.nanoTime() >= deadline) { throw new OutputLimit(); }
            super.write(values, offset, length);
        }
    }

    private static final class OutputLimit extends RuntimeException { }
}
