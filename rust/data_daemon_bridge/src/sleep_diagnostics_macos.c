#include <errno.h>
#include <os/signpost.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    uint64_t wait_start_ns;
    uint64_t wait_return_ns;
    int32_t wait_error;
    uint32_t cpu_before;
    uint32_t cpu_after;
    uint32_t qos_class;
    int32_t qos_relative_priority;
} nc_sleep_result_t;

static os_log_t sleep_log;
static pthread_once_t sleep_log_once = PTHREAD_ONCE_INIT;

static void initialize_sleep_log(void) {
    sleep_log =
        os_log_create("com.neuracore.data-daemon-tests", "sleep-latency");
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return ((uint64_t)value.tv_sec * 1000000000ULL) + (uint64_t)value.tv_nsec;
}

static void read_qos(uint32_t *qos_class, int32_t *relative_priority) {
    qos_class_t qos = QOS_CLASS_UNSPECIFIED;
    int relative = 0;
    int error =
        pthread_get_qos_class_np(pthread_self(), &qos, &relative);
    if (error != 0) {
        qos = QOS_CLASS_UNSPECIFIED;
        relative = 0;
    }
    *qos_class = (uint32_t)qos;
    *relative_priority = (int32_t)relative;
}

int32_t nc_diagnostic_sleep_ns(
    uint64_t requested_ns,
    uint64_t sleep_id,
    const char *label,
    const char *correlation_id,
    uint64_t anomaly_threshold_ns,
    nc_sleep_result_t *result
) {
    (void)anomaly_threshold_ns;
    pthread_once(&sleep_log_once, initialize_sleep_log);
    os_signpost_id_t signpost_id = (os_signpost_id_t)sleep_id;
    result->cpu_before = UINT32_MAX;
    result->cpu_after = UINT32_MAX;
    read_qos(&result->qos_class, &result->qos_relative_priority);
    uint64_t native_thread_id = 0;
    pthread_threadid_np(pthread_self(), &native_thread_id);

    result->wait_start_ns = monotonic_ns();
    uint64_t deadline_ns = result->wait_start_ns + requested_ns;
    os_signpost_interval_begin(
        sleep_log,
        signpost_id,
        "DIAGNOSTIC_SLEEP",
        "sleep_id=%{public}llu label=%{public}s native_tid=%{public}llu "
        "requested_ns=%{public}llu correlation=%{public}s",
        (unsigned long long)sleep_id,
        label,
        (unsigned long long)native_thread_id,
        (unsigned long long)requested_ns,
        correlation_id
    );
    os_signpost_event_emit(
        sleep_log,
        signpost_id,
        "SLEEP_DEADLINE",
        "sleep_id=%{public}llu deadline_monotonic_ns=%{public}llu",
        (unsigned long long)sleep_id,
        (unsigned long long)deadline_ns
    );

    struct timespec remaining = {
        .tv_sec = (time_t)(requested_ns / 1000000000ULL),
        .tv_nsec = (long)(requested_ns % 1000000000ULL),
    };
    int wait_error = 0;
    while (nanosleep(&remaining, &remaining) != 0) {
        if (errno != EINTR) {
            wait_error = errno;
            break;
        }
    }

    result->wait_return_ns = monotonic_ns();
    result->wait_error = wait_error;
    uint64_t actual_ns = result->wait_return_ns - result->wait_start_ns;
    int64_t overshoot_ns = (int64_t)actual_ns - (int64_t)requested_ns;
    os_signpost_interval_end(
        sleep_log,
        signpost_id,
        "DIAGNOSTIC_SLEEP",
        "sleep_id=%{public}llu actual_ns=%{public}llu "
        "overshoot_ns=%{public}lld error=%{public}d correlation=%{public}s",
        (unsigned long long)sleep_id,
        (unsigned long long)actual_ns,
        (long long)overshoot_ns,
        wait_error,
        correlation_id
    );
    return wait_error;
}

void nc_diagnostic_sleep_anomaly(
    uint64_t sleep_id,
    const char *label,
    const char *correlation_id,
    uint64_t requested_ns,
    uint64_t actual_ns,
    int64_t overshoot_ns,
    uint64_t native_thread_id
) {
    pthread_once(&sleep_log_once, initialize_sleep_log);
    os_signpost_event_emit(
        sleep_log,
        (os_signpost_id_t)sleep_id,
        "SLEEP_OVERSHOOT",
        "sleep_id=%{public}llu label=%{public}s native_tid=%{public}llu "
        "requested_ns=%{public}llu actual_ns=%{public}llu "
        "overshoot_ns=%{public}lld correlation=%{public}s",
        (unsigned long long)sleep_id,
        label,
        (unsigned long long)native_thread_id,
        (unsigned long long)requested_ns,
        (unsigned long long)actual_ns,
        (long long)overshoot_ns,
        correlation_id
    );
}
