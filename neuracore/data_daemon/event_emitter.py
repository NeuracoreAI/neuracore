"""Shared event emitter for cross-component signaling."""

from pyee import EventEmitter


class Emitter(EventEmitter):
    """Shared event emitter for cross-component signaling."""

    # Comms manager to State manager
    STOP_RECORDING = "STOP_RECORDING"

    # State manager -> RDM
    STOP_ALL_TRACES_FOR_RECORDING = "STOP_ALL_TRACES_FOR_RECORDING"
    # (recording_id, trace_id)

    # RDM -> State manager
    TRACE_WRITTEN = "TRACE_WRITTEN"
    # (trace_id, recording_id, bytes_written)

    # RDM -> State manager
    START_TRACE = "START_TRACE"

    # State manager -> Uploader
    READY_FOR_UPLOAD = "READY_FOR_UPLOAD"
    # (trace_id, recording_id, path, data_type, data_type_name, bytes_uploaded)

    # Connection manager -> Uploader
    IS_CONNECTED = "IS_CONNECTED"
    # (Event will trigger a state change in the consumers,
    #  this will enable tasks with
    # internet requirements to operate)

    # Upload manager -> State manager
    UPLOADED_BYTES = "UPLOADED_BYTES"
    # (trace_id, bytes_uploaded(total))

    # State manager -> Progress reporter
    PROGRESS_REPORT = "PROGRESS_REPORT"
    # (start_time:float, end_time:float, traces:List[TraceRecord])

    # Progress reporter -> State manager
    PROGRESS_REPORTED = "PROGRESS_REPORTED"
    # (recording_id:str)

    # Progress reporter -> State manager
    PROGRESS_REPORT_FAILED = "PROGRESS_REPORT_FAILED"
    # (recording_id:str, error_message:str)

    # Uploader -> State manager / RDM
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"
    # (delete this file/db entry)

    # Uploader -> state manager
    UPLOAD_FAILED = "UPLOAD_FAILED"
    # (Trace_id, bytes_uploaded, status, error_code, error_message)

    # State manager -> RDM
    DELETE_TRACE = "DELETE_TRACE"
    # (recording_id, trace_id, data_type)

    def __init__(self) -> None:
        """Initialize the event emitter."""
        super().__init__()


emitter = Emitter()

# Progress report service job is to send progress report
# updates to the backend (bytes uploaded)


# Example usage:

# from neuracore.data_daemon.event_emitter import emitter, Emitter

# @emitter.on(Emitter.UPLOAD_COMPLETE)
# def on_upload_complete(trace_id: str) -> None:
#     print("uploaded", trace_id)

# emitter.emit(Emitter.UPLOAD_COMPLETE, "trace-123")

# Use a simple decorator with the .on method to subscribe to the events,
# arguments must match the function being called
