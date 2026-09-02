"""
Football Computer Vision Analytics Interactive Dashboard.
Presents match statistics, gameplay coverage, player tracking trajectories,
movement leaderboards, camera view breakdown, and scoreboard OCR data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Page Configuration
st.set_page_config(
    page_title="Football CV Analytics | Panda FC vs Joga FC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .tag-wide {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .tag-side {
        background-color: #1e3a8a;
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .tag-other {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
DATA_DIR = Path("data")
SEGMENTS_FILE = DATA_DIR / "scoreboard_segments.csv"
FRAMES_FILE = DATA_DIR / "scoreboard_frames.csv"
OCR_FILE = DATA_DIR / "scoreboard_data.csv"
CAMERA_FILE = DATA_DIR / "camera_segments.csv"
TRACKING_FILE = DATA_DIR / "clean_tracking.csv"
POSITIONS_FILE = DATA_DIR / "player_positions.csv"
STATS_FILE = DATA_DIR / "player_statistics.csv"
PLAYERS_FILE = DATA_DIR / "players.csv"


@st.cache_data
def load_data():
    data = {}
    if SEGMENTS_FILE.exists():
        data["segments"] = pd.read_csv(SEGMENTS_FILE)
    if FRAMES_FILE.exists():
        data["frames"] = pd.read_csv(FRAMES_FILE)
    if OCR_FILE.exists():
        data["ocr"] = pd.read_csv(OCR_FILE)
    if CAMERA_FILE.exists():
        data["camera"] = pd.read_csv(CAMERA_FILE)
    if TRACKING_FILE.exists():
        data["tracking"] = pd.read_csv(TRACKING_FILE)
    if POSITIONS_FILE.exists():
        data["positions"] = pd.read_csv(POSITIONS_FILE)
    if STATS_FILE.exists():
        data["stats"] = pd.read_csv(STATS_FILE)
    if PLAYERS_FILE.exists():
        data["players"] = pd.read_csv(PLAYERS_FILE)
    return data


data = load_data()

# Sidebar
st.sidebar.title("⚽ Match Navigation")
st.sidebar.markdown("**Match:** Panda FC vs Joga FC")
st.sidebar.markdown("**Input Video:** `game2.mp4` (1920x1080 @ 59.94 FPS)")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Dashboard View",
    ["📊 Match Overview", "🏃 Player Analytics & Trajectories", "🎥 Camera & Coverage", "⏱ Scoreboard & OCR", "📁 Data Integrity Audit"]
)

# Header Banner
st.title("⚽ Football Computer Vision Analytics Pipeline")
st.caption("AI-Powered Player Tracking, Gameplay Filtering & Movement Analytics")

# TAB 1: Match Overview
if app_mode == "📊 Match Overview":
    st.subheader("Match Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    total_video_dur = 0.0
    gameplay_dur = 0.0
    gameplay_pct = 0.0
    num_segments = 0

    if "segments" in data and len(data["segments"]) > 0:
        num_segments = len(data["segments"])
        gameplay_dur = data["segments"]["duration"].sum()
        if "frames" in data and len(data["frames"]) > 0:
            total_video_dur = data["frames"]["video_time"].max()
            gameplay_pct = (gameplay_dur / total_video_dur * 100) if total_video_dur > 0 else 0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Match Video</div>
            <div class="metric-value">{total_video_dur/60:.1f} min</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Gameplay</div>
            <div class="metric-value">{gameplay_dur/60:.1f} min</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gameplay Coverage</div>
            <div class="metric-value">{gameplay_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_tracks = len(data["stats"]) if "stats" in data else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cleaned Player Tracks</div>
            <div class="metric-value">{total_tracks}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Match Timeline Breakdown
    st.subheader("Match Broadcast Coverage Timeline")
    segments = data.get("segments", pd.DataFrame())
    if not segments.empty:
        # Build timeline chart
        fig_time = go.Figure()
        
        # Base non-gameplay background
        fig_time.add_trace(go.Bar(
            y=["Match Video"],
            x=[total_video_dur],
            orientation="h",
            marker=dict(color="#ef4444"),
            name="Non-Gameplay / Ads / Breaks",
            hoverinfo="x+name"
        ))

        # Add active segments
        for i, (_, s) in enumerate(segments.iterrows()):
            fig_time.add_trace(go.Bar(
                y=["Match Video"],
                x=[s["duration"]],
                base=s["start_time"],
                orientation="h",
                marker=dict(color="#10b981"),
                name=f"Segment {i + 1}",
                hovertemplate=f"Segment {i + 1}: %{{base:.1f}}s - %{{x:.1f}}s dur<extra></extra>",
                showlegend=False
            ))

        fig_time.update_layout(
            barmode="overlay",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            height=180,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Video Time (seconds)"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning("Scoreboard segments data (`data/scoreboard_segments.csv`) is missing or empty.")

    # Leaderboard Preview
    if "stats" in data and len(data["stats"]) > 0:
        st.subheader("Top Active Players by Distance")
        st_df = data["stats"].copy()
        st.dataframe(
            st_df[["track_id", "player_name", "team", "jersey_number", "duration_seconds", "distance", "distance_unit", "average_speed", "max_speed", "coordinate_system"]].head(10),
            use_container_width=True
        )


# TAB 2: Player Analytics & Trajectories
elif app_mode == "🏃 Player Analytics & Trajectories":
    st.subheader("Player Movement Trajectories & Tracking")

    if "positions" in data and len(data["positions"]) > 0:
        pos_df = data["positions"].copy()
        stats_df = data.get("stats", pd.DataFrame())

        # Track selection
        all_tracks = sorted(pos_df["track_id"].unique())
        
        # Determine top tracks
        top_tracks = pos_df["track_id"].value_counts().head(10).index.tolist()

        col_sel, col_mode = st.columns([3, 1])
        with col_sel:
            selected_tracks = st.multiselect(
                "Select Players / Tracks to Visualize",
                options=all_tracks,
                default=top_tracks[:5]
            )
        with col_mode:
            coord_view = st.selectbox(
                "Coordinate Mode",
                ["Image Space (1920x1080)", "Normalized Image (0-1)", "Pitch World Coordinates (Meters)"]
            )

        if selected_tracks:
            filtered_pos = pos_df[pos_df["track_id"].isin(selected_tracks)].sort_values(["track_id", "frame"])

            fig_traj = go.Figure()

            # Pitch background if Pitch World
            if coord_view == "Pitch World Coordinates (Meters)" and filtered_pos["pitch_x"].notna().any():
                x_col, y_col = "pitch_x", "pitch_y"
                x_title, y_title = "Pitch Length (Metres)", "Pitch Width (Metres)"
                # Outer boundaries
                fig_traj.add_shape(type="rect", x0=0, y0=0, x1=60, y1=35, line=dict(color="white", width=2), fillcolor="#1b4d3e")
                fig_traj.add_shape(type="line", x0=30, y0=0, x1=30, y1=35, line=dict(color="white", width=2))
                fig_traj.add_shape(type="circle", x0=24, y0=11.5, x1=36, y1=23.5, line=dict(color="white", width=2))
            elif coord_view == "Normalized Image (0-1)":
                x_col, y_col = "norm_x", "norm_y"
                x_title, y_title = "Normalized X [0 - 1]", "Normalized Y [0 - 1]"
            else:
                x_col, y_col = "player_x", "player_y"
                x_title, y_title = "Image X (Pixels)", "Image Y (Pixels)"

            for t_id in selected_tracks:
                sub = filtered_pos[filtered_pos["track_id"] == t_id]
                if len(sub) > 0:
                    fig_traj.add_trace(go.Scatter(
                        x=sub[x_col],
                        y=sub[y_col],
                        mode="lines+markers",
                        name=f"Track {t_id}",
                        marker=dict(size=4),
                        line=dict(width=2.5),
                        hovertemplate=f"Track {t_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Frame: %{{customdata}}<extra></extra>",
                        customdata=sub["frame"]
                    ))

            fig_traj.update_layout(
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                font=dict(color="#f8fafc"),
                height=650,
                xaxis_title=x_title,
                yaxis_title=y_title,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            if coord_view != "Pitch World Coordinates (Meters)":
                fig_traj.update_yaxes(autorange="reversed")

            st.plotly_chart(fig_traj, use_container_width=True)

        # Full Leaderboard Table
        st.subheader("Player Statistics Leaderboard")
        if len(stats_df) > 0:
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No tracking positions available. Run the tracking stage first.")


# TAB 3: Camera & Coverage
elif app_mode == "🎥 Camera & Coverage":
    st.subheader("Broadcast Camera View Breakdown")

    if "camera" in data and len(data["camera"]) > 0:
        cam_df = data["camera"].copy()
        
        view_counts = cam_df["camera_view"].value_counts().reset_index()
        view_counts.columns = ["Camera View", "Segment Count"]

        col1, col2 = st.columns([1, 2])
        with col1:
            fig_pie = px.pie(
                view_counts,
                values="Segment Count",
                names="Camera View",
                color="Camera View",
                color_discrete_map={"WIDE": "#10b981", "SIDE": "#3b82f6", "OTHER": "#ef4444"},
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor="#1e293b",
                font=dict(color="#f8fafc"),
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### Camera Angle Guidelines")
            st.markdown("""
            - **WIDE Angle (Green)**: Main tactical broadcast view covering wide pitch and lines. Calibrated for 2D homography.
            - **SIDE Angle (Blue)**: Lateral broadcast view with pitch ground visible.
            - **OTHER (Red)**: Close-ups, replays, adverts, and transitions. Excluded from spatial mapping.
            """)

        st.subheader("Camera Segments Table")
        st.dataframe(cam_df, use_container_width=True)
    else:
        st.info("Camera segmentation not loaded.")


# TAB 4: Scoreboard & OCR
elif app_mode == "⏱ Scoreboard & OCR":
    st.subheader("Scoreboard Detection & OCR Match Clocks")

    if "ocr" in data and len(data["ocr"]) > 0:
        ocr_df = data["ocr"].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total OCR Samples", f"{len(ocr_df):,}")
        with col2:
            valid_scores = ocr_df["score"].dropna().nunique()
            st.metric("Unique Scores Detected", valid_scores)
        with col3:
            avg_ocr_conf = ocr_df["ocr_confidence"].mean() if "ocr_confidence" in ocr_df.columns else 0.0
            st.metric("Average OCR Confidence", f"{avg_ocr_conf:.1f}%")

        st.subheader("Scoreboard Samples & OCR Readings")
        st.dataframe(ocr_df, use_container_width=True)
    else:
        st.info("Scoreboard OCR data not loaded.")


# TAB 5: Data Integrity Audit
elif app_mode == "📁 Data Integrity Audit":
    st.subheader("Pipeline Validation & Integrity Status")
    
    components = [
        ("Scoreboard Frames", FRAMES_FILE),
        ("Scoreboard Segments", SEGMENTS_FILE),
        ("Scoreboard OCR", OCR_FILE),
        ("Camera Segments", CAMERA_FILE),
        ("Clean Tracking", TRACKING_FILE),
        ("Player Positions", POSITIONS_FILE),
        ("Movement Statistics", STATS_FILE),
        ("Player Roster", PLAYERS_FILE)
    ]

    audit_records = []
    for name, path in components:
        exists = path.exists()
        if exists:
            df = pd.read_csv(path)
            status = "✅ Valid & Populated" if len(df) > 0 else "⚠️ Empty File"
            count = f"{len(df):,} records"
        else:
            status = "❌ Missing"
            count = "0 records"
        audit_records.append({
            "Component": name,
            "File Path": str(path),
            "Status": status,
            "Records": count
        })

    st.dataframe(pd.DataFrame(audit_records), use_container_width=True)
