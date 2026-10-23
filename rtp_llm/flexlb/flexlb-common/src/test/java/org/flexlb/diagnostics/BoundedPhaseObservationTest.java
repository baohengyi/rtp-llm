package org.flexlb.diagnostics;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SecureDirectoryStream;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class BoundedPhaseObservationTest {
    private static final String METHOD = "batchScheduleRemainsFastAcrossRealGrpcBoundaries";
    @TempDir Path temporary;
    private Path output;

    @BeforeEach void prepare() throws Exception {
        BoundedPhaseObservation.resetForTests();
        output = temporary.toRealPath();
        Files.setPosixFilePermissions(output, PosixFilePermissions.fromString("rwx------"));
        System.setProperty(BoundedPhaseObservation.DIRECTORY_PROPERTY, output.toString());
    }

    @AfterEach void cleanup() {
        BoundedPhaseObservation.finishCase();
        BoundedPhaseObservation.resetForTests();
        System.clearProperty(BoundedPhaseObservation.DIRECTORY_PROPERTY);
    }

    private static BalanceContext context(long id) {
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        Request request = new Request();
        request.setRequestId(id);
        context.setRequest(request);
        return context;
    }

    private void start() { BoundedPhaseObservation.beginCase(METHOD, METHOD + "()"); }
    private JsonNode finish(int sequence) throws Exception {
        // Exercise the exact production serializer even where this OS lacks secure openat.
        Field current = BoundedPhaseObservation.class.getDeclaredField("current");
        current.setAccessible(true);
        Object capture = current.get(null);
        Method encode = capture.getClass().getDeclaredMethod("encode");
        encode.setAccessible(true);
        byte[] encoded = (byte[]) encode.invoke(capture);
        boolean secure;
        try (var stream = Files.newDirectoryStream(output)) {
            secure = stream instanceof SecureDirectoryStream<?>;
        }
        BoundedPhaseObservation.finishCase();
        Path file = output.resolve(String.format("phase-%03d.json", sequence));
        if (secure) {
            assertTrue(Files.isRegularFile(file));
            return new ObjectMapper().readTree(Files.readAllBytes(file));
        }
        assertFalse(Files.exists(file), "unsupported secure writer must fail closed");
        System.out.println("phase_contract_secure_directory_stream=false; serializer only, no Linux file-output qualification");
        return new ObjectMapper().readTree(encoded);
    }

    @Test void disabledDoesNotCaptureOrCreateOutput() throws Exception {
        System.clearProperty(BoundedPhaseObservation.DIRECTORY_PROPERTY);
        start();
        BoundedPhaseObservation.mark(context(16L), BoundedPhaseObservation.Phase.SERVICE_START);
        recordRound(context(16L));
        BoundedPhaseObservation.finishCase();
        assertFalse(BoundedPhaseObservation.enabled());
        try (var files = Files.list(output)) { assertEquals(0, files.count()); }
    }

    @Test void exactMethodIdentityAndRealRequestSampleAreRecorded() throws Exception {
        start();
        BoundedPhaseObservation.mark(context(16L), BoundedPhaseObservation.Phase.SERVICE_START);
        BoundedPhaseObservation.mark(context(17L), BoundedPhaseObservation.Phase.SERVICE_START);
        JsonNode report = finish(1);
        assertEquals(ProcessHandle.current().pid(), report.path("jvm_pid").asLong());
        assertEquals(METHOD, report.path("method").asText());
        assertEquals(METHOD + "()", report.path("display_name").asText());
        assertEquals(1, report.path("requests").size());
        assertEquals(16L, report.path("requests").get(0).path("request_id").asLong());
        if (Files.exists(output.resolve("phase-001.json"))) {
            assertEquals("rw-------", PosixFilePermissions.toString(Files.getPosixFilePermissions(output.resolve("phase-001.json"))));
        }
    }

    @Test void cpuPointsBelongToActualThreadsAndBatch() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        Thread thread = new Thread(() -> {
            BoundedPhaseObservation.Point point = BoundedPhaseObservation.snapshot();
            BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.EXECUTOR_START, point, 77L);
        });
        thread.start();
        thread.join(5000);
        assertFalse(thread.isAlive());
        JsonNode phases = finish(1).path("requests").get(0).path("phases");
        assertNotEquals(phases.path("SERVICE_START").path("first").path("thread_id").asLong(), phases.path("EXECUTOR_START").path("first").path("thread_id").asLong());
        assertEquals(77L, phases.path("EXECUTOR_START").path("first").path("batch_id").asLong());
        assertTrue(phases.path("SERVICE_START").path("first").path("cpu_nanos").asLong() >= -1L);
    }

    @Test void lateForeignContextWithReusedIdCannotPolluteNewCase() throws Exception {
        start();
        BalanceContext old = context(16L);
        BoundedPhaseObservation.mark(old, BoundedPhaseObservation.Phase.SERVICE_START);
        BoundedPhaseObservation.Point stale = BoundedPhaseObservation.snapshot();
        finish(1);
        start();
        BalanceContext fresh = context(16L);
        BoundedPhaseObservation.mark(fresh, BoundedPhaseObservation.Phase.SERVICE_START);
        BoundedPhaseObservation.mark(old, BoundedPhaseObservation.Phase.PLANNER_START);
        BoundedPhaseObservation.mark(fresh, BoundedPhaseObservation.Phase.PROTO_START, stale, 9L);
        JsonNode phases = finish(2).path("requests").get(0).path("phases");
        assertFalse(phases.has("PLANNER_START"));
        assertFalse(phases.has("PROTO_START"));
    }

    @Test void callbacksWithoutActualServiceEntryNeverInventRequestTraces() throws Exception {
        start();
        BoundedPhaseObservation.mark(context(16L), BoundedPhaseObservation.Phase.PLANNER_END);
        assertEquals(0, finish(1).path("requests").size());
    }

    @Test void unknownMethodDoesNotCreateAQualificationFile() throws Exception {
        BoundedPhaseObservation.beginCase("unrelatedMethod", "unrelated");
        assertFalse(BoundedPhaseObservation.enabled());
        try (var files = Files.list(output)) { assertEquals(0, files.count()); }
    }

    @Test void requestCardinalityIsBoundedAndOmissionsAreExplicit() throws Exception {
        start();
        List<BalanceContext> references = new ArrayList<>();
        for (int index = 0; index < BoundedPhaseObservation.MAX_REQUESTS + 20; index++) {
            BalanceContext context = context(index * 16L);
            references.add(context);
            BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        }
        JsonNode report = finish(1);
        assertEquals(20, report.path("request_omissions").asInt());
        assertTrue(report.path("requests").size() <= BoundedPhaseObservation.MAX_REQUESTS);
        if (Files.exists(output.resolve("phase-001.json"))) {
            assertTrue(Files.size(output.resolve("phase-001.json")) <= BoundedPhaseObservation.MAX_OUTPUT_BYTES);
        }
        assertEquals(BoundedPhaseObservation.MAX_REQUESTS + 20, references.size());
    }

    @Test void pointCardinalityIsBoundedAndOmissionsAreExplicit() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        for (int index = 0; index < BoundedPhaseObservation.MAX_POINTS + 20; index++) {
            BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.WORKER_CYCLE);
        }
        JsonNode report = finish(1);
        assertTrue(report.path("point_omissions").asLong() > 0);
        assertTrue(report.path("requests").get(0).path("phases").path("WORKER_CYCLE").path("count").asInt() < BoundedPhaseObservation.MAX_POINTS);
    }

    @Test void existingOutputIsNeverOverwritten() throws Exception {
        Path destination = output.resolve("phase-001.json");
        Files.writeString(destination, "original");
        start();
        BoundedPhaseObservation.finishCase();
        assertEquals("original", Files.readString(destination));
    }

    @Test void symlinkOutputDirectoryCannotEscape() throws Exception {
        Path real = Files.createDirectory(output.resolve("real"));
        Path link = output.resolve("linked");
        Files.createSymbolicLink(link, real);
        System.setProperty(BoundedPhaseObservation.DIRECTORY_PROPERTY, link.toString());
        start();
        BoundedPhaseObservation.finishCase();
        try (var files = Files.list(real)) { assertEquals(0, files.count()); }
    }

    @Test void expiredCaptureStopsObservationWithoutThrowing() throws Exception {
        start();
        Field current = BoundedPhaseObservation.class.getDeclaredField("current");
        current.setAccessible(true);
        Object capture = current.get(null);
        Field started = capture.getClass().getDeclaredField("startedNanos");
        started.setAccessible(true);
        started.setLong(capture, System.nanoTime() - BoundedPhaseObservation.CASE_NANOS - 1L);
        BoundedPhaseObservation.mark(context(16L), BoundedPhaseObservation.Phase.SERVICE_START);
        JsonNode report = finish(1);
        assertEquals(1L, report.path("expired_marks").asLong());
        assertEquals(0, report.path("requests").size());
    }

    @Test void concurrentRequestsRemainWithinFixedLimits() throws Exception {
        start();
        var pool = Executors.newFixedThreadPool(4);
        try {
            for (int index = 0; index < 2000; index++) {
                final long id = index * 16L;
                pool.submit(() -> BoundedPhaseObservation.mark(context(id), BoundedPhaseObservation.Phase.SERVICE_START));
            }
        } finally {
            pool.shutdown();
            assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));
        }
        JsonNode report = finish(1);
        assertTrue(report.path("requests").size() <= BoundedPhaseObservation.MAX_REQUESTS);
        assertEquals(0, report.path("writers_at_finish").asInt());
        assertTrue(report.path("request_omissions").asLong() > 0L);
    }
    @Test void fullBurstSampleFitsOutputAndRecordsAllPhases() throws Exception {
        start();
        List<BalanceContext> references = new ArrayList<>();
        for (int index = 0; index < 512; index++) {
            BalanceContext context = context(1000000L + index * 16L);
            references.add(context);
            BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
            for (BoundedPhaseObservation.Phase phase : BoundedPhaseObservation.Phase.values()) {
                if (phase != BoundedPhaseObservation.Phase.SERVICE_START) {
                    BoundedPhaseObservation.mark(context, phase);
                }
            }
            recordRound(context);
        }
        JsonNode report = finish(1);
        assertFalse(report.path("output_limit_or_serialization_error").asBoolean(), "complete original8192 request1/16 sample must fit the declared output bound");
        assertEquals(512, report.path("requests").size());
        assertEquals(0, report.path("serialization_omissions").asInt());
        for (JsonNode request : report.path("requests")) {
            assertEquals(BoundedPhaseObservation.Phase.values().length, request.path("phases").size());
            assertEquals(4, request.path("slot_lifecycle").size());
            assertEquals(last(request.path("phases").path("SERVICE_START")),
                    request.path("phases").path("SERVICE_START").path("first"));
        }
        assertEquals(512, references.size());
    }

    private static JsonNode last(JsonNode event) {
        return event.has("last") ? event.path("last") : event.path("first");
    }

    private static void recordRound(BalanceContext context) {
        BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.CLAIMED,
                new BoundedPhaseObservation.SlotState(8, 8, 0, 8192, true, false));
        BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.SUBMITTED, null);
        BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.OUTER_CLOSE_DONE, null);
        BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.RELEASED,
                new BoundedPhaseObservation.SlotState(8, 7, 1, 8191, true, false));
    }

    @Test void slotStateAndAttemptBelongToActualRequestAndThread() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        recordRound(context);
        JsonNode report = finish(1);
        assertEquals(2, report.path("schema_version").asInt());
        JsonNode slots = report.path("requests").get(0).path("slot_lifecycle");
        assertEquals(4, slots.size());
        assertEquals(8, slots.path("CLAIMED").path("first").path("state").path("in_flight").asInt());
        assertEquals(7, slots.path("RELEASED").path("first").path("state").path("in_flight").asInt());
        for (JsonNode event : slots) {
            assertEquals(1, event.path("count").asInt());
            assertEquals(1, event.path("first").path("attempt_id").asInt());
            assertEquals(Thread.currentThread().threadId(), event.path("first").path("point").path("thread_id").asLong());
            assertFalse(event.has("last"));
            assertEquals(event.path("first"), last(event));
        }
    }

    @Test void retryCountsRetainFirstAndLastDistinctAttemptIdentity() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        recordRound(context);
        recordRound(context);
        JsonNode slots = finish(1).path("requests").get(0).path("slot_lifecycle");
        for (JsonNode event : slots) {
            assertEquals(2, event.path("count").asInt());
            assertTrue(event.has("last"));
            assertEquals(1, event.path("first").path("attempt_id").asInt());
            assertEquals(2, event.path("last").path("attempt_id").asInt());
        }
    }

    @Test void slotEventsNeverCreateUnsampledOrForeignRequestTrace() throws Exception {
        start();
        BalanceContext owned = context(16L);
        BoundedPhaseObservation.mark(owned, BoundedPhaseObservation.Phase.SERVICE_START);
        recordRound(context(16L));
        recordRound(context(17L));
        JsonNode rows = finish(1).path("requests");
        assertEquals(1, rows.size());
        assertEquals(0, rows.get(0).path("slot_lifecycle").size());
    }

    @Test void slotAndOriginalPointsShareUnchangedOverallCap() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        for (int index = 0; index < BoundedPhaseObservation.MAX_POINTS + 20; index++) {
            BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.SUBMITTED, null);
        }
        JsonNode report = finish(1);
        assertEquals(32768, report.path("point_cap").asInt());
        assertTrue(report.path("point_omissions").asLong() > 0L);
        assertTrue(report.path("requests").get(0).path("slot_lifecycle").path("SUBMITTED").path("count").asInt() < 32768);
    }

    @Test void unavailableVirtualThreadCpuRemainsUnknownAndPartialCounter() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        Thread thread = Thread.ofVirtual().start(() ->
                BoundedPhaseObservation.markSlot(context, BoundedPhaseObservation.SlotPhase.SUBMITTED, null));
        thread.join(5000);
        assertFalse(thread.isAlive());
        JsonNode report = finish(1);
        assertTrue(report.path("cpu_unavailable_points").asLong() > 0);
        assertEquals(-1L, report.path("requests").get(0).path("slot_lifecycle").path("SUBMITTED").path("first").path("point").path("cpu_nanos").asLong());
    }

    @Test void expiredSlotStopsOnlyObservationAndKeepsOriginalLimit() throws Exception {
        start();
        BalanceContext context = context(16L);
        BoundedPhaseObservation.mark(context, BoundedPhaseObservation.Phase.SERVICE_START);
        Field current = BoundedPhaseObservation.class.getDeclaredField("current");
        current.setAccessible(true);
        Object capture = current.get(null);
        Field started = capture.getClass().getDeclaredField("startedNanos");
        started.setAccessible(true);
        started.setLong(capture, System.nanoTime() - BoundedPhaseObservation.CASE_NANOS - 1L);
        recordRound(context);
        JsonNode report = finish(1);
        assertEquals(4, report.path("expired_marks").asInt());
        assertEquals(0, report.path("requests").get(0).path("slot_lifecycle").size());
        assertEquals(TimeUnit.SECONDS.toNanos(120), report.path("case_capture_nanos").asLong());
    }

    @Test void outputByteCapStillRejectsLargerWrite() throws Exception {
        Class<?> type = Class.forName("org.flexlb.diagnostics.BoundedPhaseObservation$BoundedBytes");
        var constructor = type.getDeclaredConstructor();
        constructor.setAccessible(true);
        Object bytes = constructor.newInstance();
        Method write = type.getDeclaredMethod("write", byte[].class, int.class, int.class);
        write.setAccessible(true);
        var error = assertThrows(java.lang.reflect.InvocationTargetException.class,
                () -> write.invoke(bytes, new byte[2097153], 0, 2097153));
        assertEquals("OutputLimit", error.getCause().getClass().getSimpleName());
        assertEquals(2097152, BoundedPhaseObservation.MAX_OUTPUT_BYTES);
    }

}
