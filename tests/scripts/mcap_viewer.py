# cspell:ignore selectbox Scattergl vline hovermode
"""Minimal viewer for Neuracore JSON MCAP exports.

Install: pip install streamlit plotly 'mcap>=1.3.1,<2' av
Run: streamlit run tests/scripts/mcap_viewer.py -- /path/to/recording.mcap
Omit the path to upload a file in the browser.
"""

import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import av
import plotly.graph_objects as go
import streamlit as st
from mcap.reader import make_reader


def numeric_fields(value, prefix=""):
    """Flatten nested scalar/vector values into individually selectable traces."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_fields(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_fields(child, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            yield prefix, value


@st.cache_data(show_spinner="Reading MCAP…", max_entries=1)
def read_mcap(payload):
    """Read JSON channels and embedded media without loading the SDK."""
    reader = make_reader(io.BytesIO(payload))
    topics = {}
    skipped = defaultdict(int)
    for schema, channel, message in reader.iter_messages():
        if channel.message_encoding != "json":
            skipped[f"{channel.topic} ({channel.message_encoding})"] += 1
            continue
        try:
            data = json.loads(message.data)
        except (ValueError, UnicodeDecodeError):
            skipped[f"{channel.topic} (invalid JSON)"] += 1
            continue
        topic = topics.setdefault(
            channel.topic, {"metadata": channel.metadata, "samples": []}
        )
        topic["samples"].append((message.log_time, data))
    attachments = [
        {"name": item.name, "type": item.media_type, "data": item.data}
        for item in reader.iter_attachments()
    ]
    return topics, attachments, dict(skipped)


def show_media(attachment, sample_index=None):
    """Preview attached images or inspect individual video frames."""
    st.caption(attachment["name"])
    payload = attachment["data"]
    media_type = attachment["type"]
    if media_type.startswith("image/"):
        st.image(payload)
    elif media_type.startswith("video/"):
        st.video(payload, format=media_type)
        if sample_index is not None and st.checkbox("Show frame for selected sample"):
            try:
                with av.open(io.BytesIO(payload)) as container:
                    for index, frame in enumerate(container.decode(video=0)):
                        if index == sample_index:
                            # Decode explicitly as RGB; never pass BGR/YUV pixels
                            # to Streamlit's RGB image renderer.
                            st.image(frame.to_ndarray(format="rgb24"), channels="RGB")
                            break
                    else:
                        st.info("No video frame at this sample index.")
            except Exception as exc:
                st.warning(f"Could not decode video frame: {exc}")
    else:
        st.caption(f"{media_type} · {len(payload):,} bytes (no preview)")


def main():
    """Render the single-page viewer."""
    st.set_page_config(page_title="MCAP viewer", layout="wide")
    st.title("MCAP viewer")
    upload = st.sidebar.file_uploader("Open a recording", type="mcap")
    path = st.sidebar.text_input(
        "Or local file path", sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if upload is None and not path:
        st.info("Open an MCAP file to view its traces and camera media.")
        return
    try:
        payload = (
            upload.getvalue()
            if upload is not None
            else Path(path).expanduser().read_bytes()
        )
        topics, attachments, skipped = read_mcap(payload)
    except Exception as exc:
        st.error(f"Could not read MCAP: {exc}")
        return

    st.caption(
        f"{len(topics)} JSON topics · "
        f"{sum(len(t['samples']) for t in topics.values()):,} samples · "
        f"{len(attachments)} attachments"
    )
    if skipped:
        with st.expander("Skipped messages (this viewer supports JSON traces)"):
            st.json(skipped)

    selected_attachment = None
    if topics:
        name = st.sidebar.selectbox("Topic", sorted(topics))
        topic = topics[name]
        samples = topic["samples"]
        origin = min(t["samples"][0][0] for t in topics.values())
        rows = [dict(numeric_fields(data)) for _, data in samples]
        fields = sorted({key for row in rows for key in row if key != "timestamp"})
        selected = st.multiselect("Traces", fields, default=fields[:6])
        index = (
            st.slider("Sample", 0, len(samples) - 1, 0, key=f"sample:{name}")
            if len(samples) > 1
            else 0
        )
        times = [(timestamp - origin) / 1e9 for timestamp, _ in samples]
        if selected:
            figure = go.Figure()
            for field in selected:
                figure.add_trace(
                    go.Scattergl(
                        x=times,
                        y=[row.get(field) for row in rows],
                        name=field,
                        mode="lines+markers",
                        marker_size=3,
                    )
                )
            figure.add_vline(x=times[index], line_dash="dot", line_color="gray")
            figure.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Seconds from recording's first message",
                yaxis_title="Value",
                hovermode="x unified",
            )
            st.plotly_chart(figure, width="stretch")
        elif not fields:
            st.info("This topic has no numeric fields to plot.")

        trace, media = st.columns(2)
        with trace:
            st.subheader("Trace sample")
            st.caption(f"Sample {index} · {times[index]:.6f} s")
            st.json(samples[index][1])
        with media:
            selected_attachment = topic["metadata"].get("attachment")
            attachment = next(
                (a for a in attachments if a["name"] == selected_attachment), None
            )
            if attachment:
                st.subheader("Camera / media")
                show_media(attachment, index)
            elif selected_attachment:
                st.warning(f"Missing attachment: {selected_attachment}")
    else:
        st.info("No readable JSON traces found.")

    remaining = [a for a in attachments if a["name"] != selected_attachment]
    if remaining:
        with st.expander("Other attachments"):
            choice = st.selectbox(
                "Attachment",
                range(len(remaining)),
                format_func=lambda i: remaining[i]["name"],
            )
            show_media(remaining[choice])


if __name__ == "__main__":
    main()
