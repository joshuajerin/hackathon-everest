#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080
FPS = 50
FOOT_LABELS = ("LEFT FOOT", "RIGHT FOOT")
SURFACE_LABELS = {
    "hard_glacier_ice": "HARD GLACIER ICE",
    "fractured_blue_ice": "FRACTURED BLUE ICE",
    "polished_wind_ice": "POLISHED WIND ICE",
    "thin_snow_over_ice": "THIN SNOW OVER ICE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,pix_fmt,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def telemetry_values(report: dict) -> list[list[list[float]]]:
    return [
        record["packet_values"]
        for surface in report["surfaces"]
        for record in surface["telemetry"]
        if record is not None
    ]


def rounded_scale(value: float) -> float:
    if value <= 0.0:
        return 1.0
    exponent = 10.0 ** math.floor(math.log10(value))
    normalized = value / exponent
    step = (
        1.0
        if normalized <= 1.0
        else 2.0
        if normalized <= 2.0
        else 5.0
        if normalized <= 5.0
        else 10.0
    )
    return step * exponent


def draw_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    scale: float,
    colors: tuple[str, ...],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill="#142337", outline="#31506b", width=1)
    gap = 4
    width = (x1 - x0 - gap * (len(values) + 1)) / len(values)
    for index, value in enumerate(values):
        bx0 = x0 + gap + index * (width + gap)
        bx1 = bx0 + width
        ratio = min(1.0, abs(value) / max(scale, 1.0e-12))
        by0 = y1 - gap - ratio * (y1 - y0 - 2 * gap)
        draw.rectangle((bx0, by0, bx1, y1 - gap), fill=colors[index % len(colors)])


def draw_trace(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    scale: float,
    color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill="#0e1928", outline="#294158", width=1)
    if len(values) < 2:
        return
    points = []
    for index, value in enumerate(values):
        x = x0 + index * (x1 - x0) / (len(values) - 1)
        y = y1 - min(1.0, max(0.0, value / max(scale, 1.0e-12))) * (y1 - y0)
        points.append((x, y))
    draw.line(points, fill=color, width=3)


def fmt(values: list[float], digits: int) -> str:
    return "  ".join(f"{value:.{digits}f}" for value in values)


def compose_frame(
    source: Image.Image,
    surface_id: str,
    phase_index: int,
    phase_count: int,
    frame_index: int,
    record: dict | None,
    histories: dict[str, list[float]],
    scales: dict[str, float],
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#07111f")
    # The raw native frame is preserved separately. This crop enlarges the lower body and
    # bilateral crampons for the sensor-focused annotated view.
    source = source.crop((320, 300, 960, 660)).resize((1280, 720), Image.Resampling.LANCZOS)
    canvas.paste(source, (0, 92))
    draw = ImageDraw.Draw(canvas)
    white, muted, cyan, green, orange, red = (
        "#f4f8ff",
        "#91a4b8",
        "#38d6d0",
        "#47e683",
        "#ffba55",
        "#ff5f72",
    )
    draw.rectangle((0, 0, WIDTH, 92), fill="#050c17")
    draw.text((28, 18), "CRAMPON SENSOR LAB", font=font(34, bold=True), fill=white)
    draw.text(
        (28, 59),
        f"{SURFACE_LABELS[surface_id]}   |   SEGMENT {phase_index}/{phase_count}   |   t={frame_index / FPS:05.2f}s",
        font=font(18, bold=True),
        fill=cyan,
    )
    draw.rounded_rectangle((1550, 18, 1888, 72), radius=12, outline=cyan, width=2)
    draw.text((1590, 34), "VISIBLE SENSOR PACKET", font=font(17, bold=True), fill=cyan)
    draw.rectangle((1280, 92, WIDTH, 812), fill="#091525")
    draw.line((1280, 92, 1280, 812), fill=cyan, width=2)
    draw.text(
        (1310, 112),
        "EXACT BILATERAL ABI  [2 feet, 19 channels/foot]",
        font=font(19, bold=True),
        fill=white,
    )
    draw.text(
        (1310, 144),
        "force 4 | penetration 4 | accel 3 | gyro 3 | radar 5",
        font=font(16),
        fill=muted,
    )

    if record is None:
        draw.text(
            (1310, 220), "NO SENSOR PACKET AT THIS VIDEO STEP", font=font(22, bold=True), fill=red
        )
    else:
        values = record["packet_values"]
        masks = record["valid_mask"]
        ages = record["sample_age_s"]
        for foot in range(2):
            x0 = 1310
            y0 = 184 + foot * 292
            packet = values[foot]
            valid = masks[foot]
            age = ages[foot]
            draw.text(
                (x0, y0),
                FOOT_LABELS[foot],
                font=font(22, bold=True),
                fill=green if all(valid) else red,
            )
            draw.text(
                (x0 + 175, y0 + 4),
                f"fresh {sum(valid)}/19 | age max {1000 * max(age):.1f} ms | sensor t {record['timestamp_s'][foot]:.3f} s",
                font=font(14),
                fill=muted,
            )
            draw.text((x0, y0 + 40), "FORCE probes [N]", font=font(15, bold=True), fill=orange)
            draw.text((x0 + 185, y0 + 40), fmt(packet[0:4], 1), font=font(15), fill=white)
            draw_bar(
                draw,
                (x0, y0 + 64, x0 + 548, y0 + 105),
                packet[0:4],
                scales["force"],
                (orange, green, cyan, "#9c8cff"),
            )
            draw.text(
                (x0, y0 + 119), "PENETRATION probes [mm]", font=font(15, bold=True), fill=cyan
            )
            draw.text(
                (x0 + 220, y0 + 119),
                fmt([1000 * x for x in packet[4:8]], 2),
                font=font(15),
                fill=white,
            )
            draw_bar(
                draw,
                (x0, y0 + 143, x0 + 548, y0 + 184),
                packet[4:8],
                scales["penetration"],
                (cyan, "#7edbff", green, "#9c8cff"),
            )
            draw.text(
                (x0, y0 + 198),
                f"ACCEL world xyz [m/s²]  {fmt(packet[8:11], 2)}",
                font=font(15),
                fill=white,
            )
            draw.text(
                (x0, y0 + 222),
                f"GYRO world xyz [rad/s]   {fmt(packet[11:14], 2)}",
                font=font(15),
                fill=white,
            )
            draw.text(
                (x0, y0 + 246),
                f"RADAR frontend      {fmt(packet[14:19], 3)}",
                font=font(15),
                fill=white,
            )

    draw.rectangle((0, 812, WIDTH, HEIGHT), fill="#06101d")
    draw.line((0, 812, WIDTH, 812), fill=cyan, width=2)
    draw.text(
        (28, 832),
        "TIME-ALIGNED VISIBLE SENSOR TRACES  (rolling 1.0 s)",
        font=font(19, bold=True),
        fill=white,
    )
    draw.text(
        (28, 864), f"total force scale 0–{scales['force_trace']:.0f} N", font=font(14), fill=muted
    )
    draw_trace(draw, (28, 888, 620, 1015), histories["left_force"], scales["force_trace"], orange)
    draw_trace(draw, (646, 888, 1238, 1015), histories["right_force"], scales["force_trace"], green)
    draw.text((44, 988), "LEFT total force", font=font(14, bold=True), fill=orange)
    draw.text((662, 988), "RIGHT total force", font=font(14, bold=True), fill=green)
    draw.text(
        (1268, 832),
        f"max penetration scale 0–{scales['penetration']:.2f} mm",
        font=font(14),
        fill=muted,
    )
    draw_trace(draw, (1268, 864, 1888, 1015), histories["penetration"], scales["penetration"], cyan)
    draw.text((1284, 988), "max probe penetration", font=font(14, bold=True), fill=cyan)
    draw.text(
        (28, 1038),
        "Simulator-only evidence | each video frame shows the latest packet; all higher-rate ticks are preserved in the sidecar",
        font=font(14),
        fill=muted,
    )
    draw.text(
        (1430, 1038), "surface = world metadata, not a sensor channel", font=font(14), fill=muted
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose a time-aligned crampon sensor HUD video")
    parser.add_argument("--sensor-world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.sensor_world.expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report["packet_abi"] != ["B", 2, 19]:
        raise ValueError("sensor-world report does not preserve the bilateral ABI")
    values = telemetry_values(report)
    if not values:
        raise ValueError("sensor-world report contains no packets")
    force_scale = rounded_scale(
        max(abs(channel) for packet in values for foot in packet for channel in foot[0:4])
    )
    total_force_scale = rounded_scale(
        max(sum(max(0.0, x) for x in foot[0:4]) for packet in values for foot in packet)
    )
    penetration_scale_mm = rounded_scale(
        1000 * max(abs(channel) for packet in values for foot in packet for channel in foot[4:8])
    )
    scales = {
        "force": force_scale,
        "force_trace": total_force_scale,
        "penetration": penetration_scale_mm,
    }
    first_video = Path(report["surfaces"][0]["video"]["path"])
    stream = probe(first_video)
    source_width, source_height = int(stream["width"]), int(stream["height"])
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    if encoder.stdin is None:
        raise RuntimeError("failed to open ffmpeg encoder")
    frames_written = 0
    for phase_index, surface in enumerate(report["surfaces"], start=1):
        video = Path(surface["video"]["path"])
        decoder = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        if decoder.stdout is None:
            raise RuntimeError("failed to open ffmpeg decoder")
        frame_size = source_width * source_height * 3
        histories = {"left_force": [], "right_force": [], "penetration": []}
        for frame_index, record in enumerate(surface["telemetry"]):
            raw = decoder.stdout.read(frame_size)
            if len(raw) != frame_size:
                raise RuntimeError(f"short frame read for {surface['surface_id']} at {frame_index}")
            source = Image.frombytes("RGB", (source_width, source_height), raw)
            if record is not None:
                packet = record["packet_values"]
                histories["left_force"].append(sum(max(0.0, x) for x in packet[0][0:4]))
                histories["right_force"].append(sum(max(0.0, x) for x in packet[1][0:4]))
                histories["penetration"].append(
                    1000 * max(abs(x) for foot in packet for x in foot[4:8])
                )
            for history in histories.values():
                del history[:-FPS]
            frame = compose_frame(
                source,
                surface["surface_id"],
                phase_index,
                len(report["surfaces"]),
                frame_index,
                record,
                histories,
                scales,
            )
            encoder.stdin.write(frame.tobytes())
            frames_written += 1
        decoder.stdout.close()
        if decoder.wait() != 0:
            raise RuntimeError(f"ffmpeg decoder failed for {video}")
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg encoder failed")
    output_stream = probe(output)
    if int(output_stream["nb_frames"]) != frames_written:
        raise RuntimeError("output frame count mismatch")
    manifest = {
        "schema_version": "1.0.0",
        "source": {"path": str(report_path), "sha256": sha256(report_path)},
        "output": {"path": str(output), "sha256": sha256(output), "stream": output_stream},
        "frames": frames_written,
        "fps": FPS,
        "scales_from_recorded_data": scales,
        "annotation_semantics": "Every displayed numeric value and trace is aligned to its recorded visible packet. Surface identity is separately labeled world metadata.",
        "visual_crop": {
            "source_pixels": [320, 300, 960, 660],
            "purpose": "Enlarge the lower body and bilateral crampons; raw native clips are preserved.",
        },
        "claim_boundary": report["claim_boundary"],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
