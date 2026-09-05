package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Test-only {@link EngineCancelChannel} backed by the in-process mock engine
 * cluster. Resolves the target {@link JavaMockEngineCluster.FastRpcService} by
 * the endpoint's gRPC port and drives the mock cancel behaviour: a live
 * request is removed and a CANCELLED completion surfaces in the next
 * WorkerStatus finished list, exactly like a real engine would report.
 * Mirrors the engine contract: a live request and its accepted-cancel
 * tombstone return ACCEPTED; a never-seen request installs an absent fence and
 * returns TOMBSTONED; a naturally completed request returns NOT_FOUND; Decode
 * rejects this RPC as unsupported.
 *
 * <p><b>Wiring:</b> this class is NOT a Spring component. Production contexts
 * keep {@code UnsupportedEngineCancelChannel}; tests inject this channel
 * explicitly, e.g.:
 * <pre>{@code
 *   Map<Integer, FastRpcService> services = ...; // port -> mock engine
 *   EngineCancelChannel channel = new MockEngineCancelChannel(services);
 * }</pre>
 */
public final class MockEngineCancelChannel implements EngineCancelChannel {

    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    public MockEngineCancelChannel(Map<Integer, JavaMockEngineCluster.FastRpcService> services) {
        this.services = services;
    }

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return endpoint != null && services.containsKey(endpoint.getGrpcPort());
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target, long requestId,
                                                   long timeoutMs) {
        JavaMockEngineCluster.FastRpcService service = target == null
                ? null : services.get(target.prefillGrpcPort());
        if (service == null) {
            return CompletableFuture.completedFuture(CancelOutcome.unsupported());
        }
        try {
            // Deliberately inspect only the addressed Prefill. Scanning other
            // workers would hide an incorrect Prefill route in tests.
            JavaMockEngineCluster.CancelResult result = service.cancelRequest(requestId);
            return CompletableFuture.completedFuture(switch (result.status()) {
                case ACCEPTED -> CancelOutcome.accepted();
                case NOT_FOUND -> CancelOutcome.notFound();
                case TOMBSTONED -> CancelOutcome.tombstoned();
            });
        } catch (UnsupportedOperationException e) {
            return CompletableFuture.completedFuture(CancelOutcome.failed());
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }
}
